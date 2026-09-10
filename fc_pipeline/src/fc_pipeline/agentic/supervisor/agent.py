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

                # Capture resolution metadata for manifest compilation
                if t_name == "resolve_frequency_band" and "confidence" in observation:
                    tool_confidences["frequency_band"] = observation["confidence"]
                    resolved_band_info = observation
                elif t_name == "resolve_channel_selection" and "confidence" in observation:
                    tool_confidences["channels"] = observation["confidence"]
                    resolved_channel_info = observation
                elif t_name == "get_dataset_conditions" and "conditions" in observation:
                    # Match requested condition against valid dataset conditions
                    # Collapse whitespace on both sides so single-space user
                    # input matches double-spaced BIDS trigger labels (and vice-versa).
                    _norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
                    _user_norm = _norm(user_request)
                    for cond in observation["conditions"]:
                        if _norm(cond) in _user_norm:
                            resolved_condition = cond
                            tool_confidences["condition"] = 1.0
                            break

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

    # If any axis is missing, halt with a precise clarification question
    if not (band_ok and channels_ok and condition_ok):
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
    }