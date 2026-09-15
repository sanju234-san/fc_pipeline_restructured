"""Supervisor ReAct agent loop, parameter resolution, and manifest assembly."""

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel

from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand
from fc_pipeline.schemas.state import GraphState
from fc_pipeline.config.metric_canonicalization import METRIC_LOOKUP
from fc_pipeline.config.thresholds import (
    SUPERVISOR_CONFIDENCE_THRESHOLD,
    TAU_PHASE,
    TAU_ZEROLAG,
)
from fc_pipeline.agentic.supervisor.prompts import SUPERVISOR_SYSTEM_PROMPT
from fc_pipeline.agentic.supervisor.tracer import SupervisorTracer
from fc_pipeline.agentic.supervisor.tool_call_parser import extract_mistral_style_tool_calls
from fc_pipeline.observability.mlflow_tracker import trace_span
from fc_pipeline.agentic.supervisor.tools.dataset_info import get_dataset_info
from fc_pipeline.agentic.supervisor.tools.dataset_conditions import get_dataset_conditions
from fc_pipeline.agentic.supervisor.tools.frequency_band import resolve_frequency_band
from fc_pipeline.agentic.supervisor.tools.channel_selection import resolve_channel_selection
from fc_pipeline.agentic.supervisor.tools.dataset_overview_plot import generate_dataset_overview_plot


# Available Supervisor inspection tools (5 total)
SUPERVISOR_TOOLS = [
    get_dataset_info,
    get_dataset_conditions,
    resolve_frequency_band,
    resolve_channel_selection,
    generate_dataset_overview_plot,
]
TOOL_MAP = {t.name: t for t in SUPERVISOR_TOOLS}

# Tools that can fully satisfy a request without ever touching any of the 3
# mandatory scientific axes (frequency band, channels, condition). A request
# that only calls these — e.g. "show me an overview plot of this dataset",
# "what conditions does this dataset have" — is a complete, terminal
# informational/diagnostic response, not a halted analysis awaiting more
# input. See the axis_resolution_attempted check in supervisor_node().
INFORMATIONAL_TOOLS = {
    "get_dataset_info",
    "get_dataset_conditions",
    "generate_dataset_overview_plot",
}


def canonicalize_metrics_inline(user_request: str) -> Tuple[List[MetricEnum], Optional[str]]:
    """Sub-part 3: Deterministic inline metric canonicalization against METRIC_LOOKUP (Section 6.1 Step 5).
    
    Returns (selected_metrics, error_message). Defaults to all 5 foundational metrics if unspecified.
    """
    cleaned_request = user_request.lower()
    matched_metrics: List[MetricEnum] = []
    
    # Check for matches across aliases in METRIC_LOOKUP
    for alias, info in METRIC_LOOKUP.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, cleaned_request):
            canonical = info["canonical_id"]
            if canonical not in matched_metrics:
                matched_metrics.append(canonical)
                
    # If no metrics were explicitly named, default to all five foundational metrics
    if not matched_metrics:
        matched_metrics = [
            MetricEnum.PLI,
            MetricEnum.WPLI,
            MetricEnum.IMAGINARY_COHERENCE,
            MetricEnum.PLV,
            MetricEnum.COHERENCE,
        ]
        
    return matched_metrics, None


# ---------------------------------------------------------------------------
# Condition resolution (Bug 2 fix)
#
# supervisor_node() matches the user's requested experimental condition
# against the dataset's known condition labels via substring search over
# `user_request`. Because `user_request` can be an accumulated multi-turn
# string (e.g. the original request plus a Gate 1 "Request Changes" reply
# concatenated on), a naive "first substring match, in dataset iteration
# order" search silently prefers whichever condition happens to appear
# first in the dataset file — which is very often the OLD condition from
# an earlier turn, not the user's most recent, actual intent. The helpers
# below fix this with a two-layer, recency-aware search.
# ---------------------------------------------------------------------------

# Tags that mark the start of a newer, higher-priority statement of user
# intent within an accumulated multi-turn user_request. Condition matching
# must prefer text after the LAST such tag — otherwise an earlier-mentioned
# condition can win a substring match purely because it appears first in
# the dataset's condition-list iteration order, silently overriding a more
# recent, explicit user correction (e.g. a Gate 1 "Request Changes" reply).
_REVISION_TAGS = (
    "[User Gate 1 Change Request]:",
    "[User follow-up refinement]:",
    "[User clarification reply]:",
)


def _condition_search_window(user_request: str) -> str:
    """Text to search first when resolving the condition: everything after
    the last revision tag present, or the full request if none are present.
    """
    last_tag_end = -1
    for tag in _REVISION_TAGS:
        pos = user_request.rfind(tag)
        if pos != -1:
            last_tag_end = max(last_tag_end, pos + len(tag))
    return user_request[last_tag_end:] if last_tag_end >= 0 else user_request


def resolve_condition_from_request(
    user_request: str, conditions: List[str]
) -> Optional[str]:
    """Determine which dataset condition the accumulated user_request refers to.

    Two-layer strategy:
      1. Search only the most recent turn (text after the last revision tag).
         Within that window, take the LAST substring match by index, so an
         explicit correction always wins over an earlier mention within the
         same turn.
      2. If nothing matches in the recent window (the latest turn didn't
         address condition at all — e.g. a revision that only changed the
         frequency band), fall back to searching the full accumulated
         request, so a condition resolved earlier still carries over
         unchanged.

    Returns the matched condition label exactly as given in `conditions`,
    or None if no condition label appears anywhere in the request.
    """
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())

    def _last_match(haystack_norm: str) -> Optional[str]:
        best_idx, best_cond = -1, None
        for cond in conditions:
            idx = haystack_norm.rfind(norm(cond))
            if idx > best_idx:
                best_idx, best_cond = idx, cond
        return best_cond

    recent = _last_match(norm(_condition_search_window(user_request)))
    if recent is not None:
        return recent
    return _last_match(norm(user_request))


def compile_and_confirm_manifest(
    plan: AnalysisPlan,
    tool_confidences: Dict[str, float],
    trial_count: Optional[int] = None,
) -> Tuple[List[ParameterManifestEntry], Optional[str]]:
    """Sub-part 4: Validates structural invariants and compiles the full Gate 1 checklist (Section 6.1 Step 6)."""
    # 1. Structural Invariant Checks (Single Canonical Home)
    if len(plan.channels) < 2:
        return [], f"Structural validation failed: Functional connectivity requires at least 2 distinct channels. Found: {len(plan.channels)}"
    if plan.freq_band.fmin >= plan.freq_band.fmax:
        return [], f"Structural validation failed: fmin ({plan.freq_band.fmin} Hz) must be strictly less than fmax ({plan.freq_band.fmax} Hz)"
    if not plan.metrics:
        return [], "Structural validation failed: Metric list cannot be empty."
    if not plan.condition:
        return [], "Structural validation failed: Experimental condition must be specified."

    manifest: List[ParameterManifestEntry] = []

    # 2. Compile Scientific Axes (Risk Tier: low)
    fb_conf = tool_confidences.get("frequency_band", 1.0)
    manifest.append(
        ParameterManifestEntry(
            name="freq_band",
            category="scientific_axis",
            proposed_value=f"{plan.freq_band.name} ({plan.freq_band.fmin:.1f} - {plan.freq_band.fmax:.1f} Hz)",
            confidence=fb_conf,
            needs_human_input=(fb_conf < SUPERVISOR_CONFIDENCE_THRESHOLD),
            risk_tier="low",
        )
    )

    ch_conf = tool_confidences.get("channels", 1.0)
    manifest.append(
        ParameterManifestEntry(
            name="channels",
            category="scientific_axis",
            proposed_value=", ".join(plan.channels),
            confidence=ch_conf,
            needs_human_input=(ch_conf < SUPERVISOR_CONFIDENCE_THRESHOLD),
            risk_tier="low",
        )
    )

    cond_conf = tool_confidences.get("condition", 1.0)
    manifest.append(
        ParameterManifestEntry(
            name="condition",
            category="scientific_axis",
            proposed_value=plan.condition,
            confidence=cond_conf,
            needs_human_input=(cond_conf < SUPERVISOR_CONFIDENCE_THRESHOLD),
            risk_tier="low",
        )
    )

    # 3. Metric Selection (Risk Tier: low, non-LLM resolved)
    manifest.append(
        ParameterManifestEntry(
            name="metrics",
            category="metric_selection",
            proposed_value=", ".join([m.value for m in plan.metrics]),
            confidence=None,
            needs_human_input=False,
            risk_tier="low",
        )
    )

    # 4. Trial Adequacy Check (Scientific Completeness)
    if trial_count is not None and trial_count < 3:
        manifest.append(
            ParameterManifestEntry(
                name="trial_adequacy",
                category="advisory",
                proposed_value=f"Only {trial_count} trial(s) available for condition '{plan.condition}'. Connectivity estimates from very few trials have limited statistical reliability.",
                confidence=None,
                needs_human_input=True,
                risk_tier="elevated",
            )
        )

    # 5. Cross-Metric Synthesis Advisory (Section 10: Option b with Guardrails)
    # The 4-category evidence classification requires at least one Phase-Robust
    # metric (PLI/wPLI/ImCoh) AND at least one Zero-Lag metric (PLV/Coh).
    # If either group is unrepresented, the synthesis is mathematically undefined.
    has_phase_robust = any(
        m in {MetricEnum.PLI, MetricEnum.WPLI, MetricEnum.IMAGINARY_COHERENCE}
        for m in plan.metrics
    )
    has_zerolag = any(
        m in {MetricEnum.PLV, MetricEnum.COHERENCE}
        for m in plan.metrics
    )

    if not (has_phase_robust and has_zerolag):
        manifest.append(
            ParameterManifestEntry(
                name="cross_metric_synthesis",
                category="advisory",
                proposed_value=(
                    "DISABLED: Requested metrics lack representation from both "
                    "Phase-Robust (PLI/wPLI/ImCoh) and Zero-Lag (PLV/Coh) groups. "
                    "4-category artifact disambiguation will not run."
                ),
                confidence=None,
                needs_human_input=True,
                risk_tier="elevated",
            )
        )

    # 6. Engineering & Coupling Thresholds (Risk Tier: elevated)
    manifest.append(
        ParameterManifestEntry(
            name="reference",
            category="engineering_threshold",
            proposed_value="average",
            confidence=None,
            needs_human_input=False,
            risk_tier="elevated",
        )
    )

    manifest.append(
        ParameterManifestEntry(
            name="tau_phase",
            category="engineering_threshold",
            proposed_value=str(TAU_PHASE),
            confidence=None,
            needs_human_input=False,
            risk_tier="elevated",
        )
    )

    manifest.append(
        ParameterManifestEntry(
            name="tau_zerolag",
            category="engineering_threshold",
            proposed_value=str(TAU_ZEROLAG),
            confidence=None,
            needs_human_input=False,
            risk_tier="elevated",
        )
    )

    # 7. Dataset Overview Plot (Informational / Diagnostic — not a scientific axis)
    manifest.append(
        ParameterManifestEntry(
            name="dataset_overview_plot",
            category="informational",
            proposed_value="generate_dataset_overview_plot available (optional pre-preprocessing diagnostic)",
            confidence=None,
            needs_human_input=False,
            risk_tier="low",
        )
    )

    # 8. Bad Channel Screening Status (Informational)
    manifest.append(
        ParameterManifestEntry(
            name="bad_channel_screening",
            category="informational",
            proposed_value="Not yet performed. Will run in Data Preparation with before/after variance audit.",
            confidence=None,
            needs_human_input=False,
            risk_tier="low",
        )
    )

    return manifest, None


def supervisor_node(
    state: GraphState,
    llm: BaseChatModel,
    run_id: str = "default_run",
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """Sub-parts 1 & 2: Supervisor ReAct loop and LangGraph node entry point."""
    user_request = state["user_request"]
    raw_data_path = state["raw_data_path"]

    tracer = SupervisorTracer(run_id=run_id)
    tracer.log_event("supervisor_start", {"user_request": user_request, "raw_data_path": raw_data_path})

    # Sub-part 1: Bind tools to LLM
    llm_with_tools = llm.bind_tools(SUPERVISOR_TOOLS)

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Dataset Path: {raw_data_path}\nRun ID: {run_id}\nUser Request: {user_request}"),
    ]

    tool_confidences: Dict[str, float] = {}
    resolved_band_info: Optional[Dict[str, Any]] = None
    resolved_channel_info: Optional[Dict[str, Any]] = None
    resolved_condition: Optional[str] = None
    resolved_trial_count: Optional[int] = None
    resolved_condition_counts: Dict[str, int] = {}
    clarification_question: Optional[str] = None
    attempted_frequency_band: bool = False
    attempted_channel_selection: bool = False
    informational_tool_outputs: Dict[str, Any] = {}

    # Sub-part 2: The ReAct Execution Loop
    for iteration in range(max_iterations):
        response: AIMessage = llm_with_tools.invoke(messages)

        # Extract tool calls (standard OpenAI schema or fallback parser)
        tool_calls = response.tool_calls
        if not tool_calls and isinstance(response.content, str) and "[TOOL_CALLS]" in response.content:
            tool_calls = extract_mistral_style_tool_calls(response.content)
            if tool_calls:
                response = response.model_copy(update={"tool_calls": tool_calls})
                tracer.log_event("fallback_tool_calls_parsed", {"count": len(tool_calls), "calls": tool_calls})

        messages.append(response)
        tracer.log_event("agent_thought", {"iteration": iteration + 1, "content": response.content})

        if tool_calls:
            for tool_call in tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]

                # Defensively populate path & run_id for overview plot if omitted by LLM
                if t_name == "generate_dataset_overview_plot":
                    if not t_args.get("data_path"):
                        t_args["data_path"] = raw_data_path
                    if not t_args.get("run_id") or t_args.get("run_id") == "default":
                        t_args["run_id"] = state.get("run_id") or run_id

                tracer.log_event("tool_call", {"tool": t_name, "args": t_args})

                # Track that an axis-resolution tool was actually invoked,
                # regardless of whether it succeeds — this is what
                # distinguishes "the user asked for something requiring
                # these axes and it's genuinely unresolved" (real
                # clarification needed) from "the request never needed
                # this axis at all" (see axis_resolution_attempted below).
                if t_name == "resolve_frequency_band":
                    attempted_frequency_band = True
                elif t_name == "resolve_channel_selection":
                    attempted_channel_selection = True

                tool_fn = TOOL_MAP.get(t_name)
                # Inject duration_seconds into resolve_frequency_band calls for cycle check
                if t_name == "resolve_frequency_band":
                    if "duration_seconds" not in t_args:
                        # Extract duration_seconds from get_dataset_info's observation if available
                        duration = state.get("dataset_duration_seconds")
                        if not duration:
                            # Fallback to a safe default if get_dataset_info hasn't run yet
                            duration = 60.0
                        t_args["duration_seconds"] = duration

                if tool_fn:
                    with trace_span(
                        name=f"tool:{t_name}",
                        span_type="TOOL",
                        inputs={"tool": t_name, "args": t_args},
                    ):
                        observation = tool_fn.invoke(t_args)
                else:
                    observation = {"error": f"Tool '{t_name}' not found."}

                # Capture per-condition trial counts for manifest compilation.
                # NOTE: "total_trials" is the dataset-wide sum across *all* conditions,
                # not the count for the specific condition the user requested, so it is
                # kept only as a last-resort fallback (see resolution below the loop).
                if t_name == "get_dataset_conditions" and "conditions" in observation:
                    resolved_condition_counts = observation["conditions"]
                    if resolved_trial_count is None and "total_trials" in observation:
                        resolved_trial_count = observation["total_trials"]
                # Capture duration_seconds from get_dataset_info for resolve_frequency_band
                if t_name == "get_dataset_info" and "duration_seconds" in observation:
                    state["dataset_duration_seconds"] = observation["duration_seconds"]

                tracer.log_event("tool_observation", {"tool": t_name, "observation": observation})

                # Capture raw output of successful informational/diagnostic
                # tool calls (e.g. the overview plot's path) so a terminal
                # informational response can carry structured artifact data
                # forward, rather than relying on the plot path only
                # appearing inside the LLM's free-text narration.
                if t_name in INFORMATIONAL_TOOLS and not (
                    isinstance(observation, dict) and observation.get("error")
                ):
                    informational_tool_outputs[t_name] = observation

                # Capture resolution metadata for manifest compilation
                if t_name == "resolve_frequency_band" and "confidence" in observation:
                    tool_confidences["frequency_band"] = observation["confidence"]
                    resolved_band_info = observation
                elif t_name == "resolve_channel_selection" and "confidence" in observation:
                    tool_confidences["channels"] = observation["confidence"]
                    resolved_channel_info = observation
                elif t_name == "get_dataset_conditions" and "conditions" in observation:
                    # Match requested condition against valid dataset conditions.
                    # See resolve_condition_from_request() above: this prefers a
                    # condition stated in the most recent turn (e.g. a Gate 1
                    # "Request Changes" reply) over one merely mentioned earlier
                    # in the accumulated user_request, and only falls back to the
                    # full request when the latest turn doesn't address condition
                    # at all.
                    match = resolve_condition_from_request(user_request, observation["conditions"])
                    if match is not None:
                        resolved_condition = match
                        tool_confidences["condition"] = 1.0

                # Append tool observation back to messages
                call_id = tool_call.get("id", "fallback_id")
                messages.append(ToolMessage(tool_call_id=call_id, content=str(observation)))
        else:
            # No tool calls: LLM finished reasoning
            break

    # Sub-part 3: Structural Guardrail Evaluation across all 3 Scientific Axes
    band_ok = bool(resolved_band_info and not resolved_band_info.get("error"))
    channels_ok = bool(resolved_channel_info and not resolved_channel_info.get("error"))
    condition_ok = bool(resolved_condition)

    # Whether this request even attempted to resolve any of the 3 mandatory
    # axes at all. A bare informational/diagnostic request (e.g. "show me an
    # overview plot", "what conditions does this dataset have") never calls
    # resolve_frequency_band or resolve_channel_selection, and may call
    # get_dataset_conditions purely to list conditions rather than to match
    # one against the request — so resolved_condition stays None too. That
    # is a fully completed request, not a halted one awaiting more input.
    axis_resolution_attempted = (
        attempted_frequency_band or attempted_channel_selection or resolved_condition is not None
    )

    if not (band_ok and channels_ok and condition_ok):
        if not axis_resolution_attempted and informational_tool_outputs:
            # Terminal informational/diagnostic completion — nothing further
            # to ask. Surface the LLM's own final narration as the response
            # text, plus the raw tool output(s) as structured artifact data
            # (e.g. informational_tool_outputs["generate_dataset_overview_plot"]
            # carries whatever path/metadata keys that tool returns) so the
            # frontend can render an image directly instead of parsing it out
            # of free text.
            informational_text = (
                response.content.strip()
                if (response and isinstance(response.content, str))
                else ""
            )
            if not informational_text or informational_text.startswith("[TOOL_CALLS]"):
                informational_text = "Request completed."

            tracer.log_event("supervisor_informational_complete", {
                "response": informational_text,
                "tools": list(informational_tool_outputs.keys()),
            })
            return {
                "clarification_question": None,
                "plan": None,
                "parameter_manifest": None,
                "preflight_confirmed": False,
                "informational_response": informational_text,
                "informational_artifacts": informational_tool_outputs,
            }

        # Genuine halt: an axis was attempted (or partially resolved) and
        # is still missing — ask a precise clarification question.
        last_thought = response.content.strip() if (response and isinstance(response.content, str)) else ""
        if last_thought and not last_thought.startswith("[TOOL_CALLS]"):
            clarification_question = last_thought
        elif not band_ok:
            clarification_question = "Please specify a valid physiological frequency band (e.g., alpha, 8-12 Hz)."
        elif not channels_ok:
            clarification_question = "Please specify at least two EEG channels or a valid brain region."
        else:
            clarification_question = "Could not resolve a valid experimental condition from your request matching the dataset events. Please specify the target condition."

        tracer.log_event("supervisor_halt_unresolved_axes", {
            "question": clarification_question,
            "band_ok": band_ok,
            "channels_ok": channels_ok,
            "condition_ok": condition_ok,
        })
        return {
            "clarification_question": clarification_question,
            "plan": None,
            "parameter_manifest": None,
            "preflight_confirmed": False,
            "informational_response": None,
            "informational_artifacts": None,
        }

    # Resolve the trial count for the *specific* condition that was matched,
    # rather than the dataset-wide total across all conditions.
    if resolved_condition in resolved_condition_counts:
        resolved_trial_count = resolved_condition_counts[resolved_condition]

    # Inline Metric Canonicalization (Sub-part 3)
    metrics, _ = canonicalize_metrics_inline(user_request)

    # Assemble AnalysisPlan
    plan = AnalysisPlan(
        metrics=metrics,
        freq_band=FrequencyBand(
            name=resolved_band_info["name"],
            fmin=resolved_band_info["fmin"],
            fmax=resolved_band_info["fmax"],
        ),
        channels=resolved_channel_info["resolved_channels"],
        condition=resolved_condition,
    )

    # Manifest Compilation & Invariant Validation (Sub-part 4)
    manifest, validation_error = compile_and_confirm_manifest(plan, tool_confidences, resolved_trial_count)
    if validation_error:
        tracer.log_event("structural_validation_failed", {"error": validation_error})
        return {
            "clarification_question": validation_error,
            "plan": None,
            "parameter_manifest": None,
            "preflight_confirmed": False,
            "informational_response": None,
            "informational_artifacts": None,
        }

    tracer.log_event("manifest_compiled", {"plan": plan.model_dump(), "manifest_rows": len(manifest)})

    # DETERMINISTIC ENFORCEMENT: once band + channels + condition are all resolved
    # (band_ok + channels_ok + condition_ok all True above), never emit a
    # clarification_question regardless of what the LLM's narration/text says.
    # The local clarification_question variable was initialized to None above and
    # not modified on this resolved path; we explicitly re-null it and the
    # returned dict below to guarantee gate_1_review routing.
    clarification_question = None

    # Return state update — preflight_confirmed remains False until human signs off at Gate 1!
    return {
        "clarification_question": None,
        "plan": plan,
        "parameter_manifest": manifest,
        "preflight_confirmed": False,
        "informational_response": None,
        "informational_artifacts": None,
    }
