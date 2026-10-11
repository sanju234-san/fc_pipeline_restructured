"""Supervisor ReAct agent loop, parameter resolution, and manifest assembly."""

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel

from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand
from fc_pipeline.schemas.state import GraphState
from fc_pipeline.schemas.clarification import AXIS_KINDS, build_clarification, legacy_fields
from fc_pipeline.config.metric_canonicalization import METRIC_LOOKUP
from fc_pipeline.config.thresholds import (
    SUPERVISOR_CONFIDENCE_THRESHOLD,
    TAU_PHASE,
    TAU_ZEROLAG,
    BAD_CHANNEL_VARIANCE_THRESHOLD,
    MIN_CYCLES,
    OCULAR_VARIANCE_RATIO,
    MUSCLE_POWER_THRESHOLD,
    ELECTRODE_POP_SIGMA,
)
from fc_pipeline.agentic.supervisor.prompts import SUPERVISOR_SYSTEM_PROMPT
from fc_pipeline.agentic.supervisor.action_policy import DecisionType, evaluate_action
from fc_pipeline.agentic.supervisor.tracer import SupervisorTracer
from fc_pipeline.agentic.supervisor.tool_call_parser import extract_mistral_style_tool_calls
from fc_pipeline.observability.mlflow_tracker import trace_span
from fc_pipeline.toolbox.registry import toolbox_registry
from fc_pipeline.toolbox.dataset import (
    CANONICAL_BANDS,
    REGION_MAP,
    normalize_channel_label,
)


# Available Supervisor inspection tools sourced from centralized toolbox registry (5 total)
SUPERVISOR_TOOLS = [
    toolbox_registry.get_tool("get_dataset_info"),
    toolbox_registry.get_tool("get_dataset_conditions"),
    toolbox_registry.get_tool("resolve_frequency_band"),
    toolbox_registry.get_tool("resolve_channel_selection"),
    toolbox_registry.get_tool("generate_dataset_overview_plot"),
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


UNSUPPORTED_METRICS: List[str] = [
    "mutual information",
    "transfer entropy",
    "granger causality",
    "directed transfer function",
    "dtf",
    "direct directed transfer function",
    "ddtf",
    "partial directed coherence",
    "pdc",
    "envelope correlation",
    "cross-correlation",
]


def canonicalize_metrics_inline(user_request: str) -> Tuple[List[MetricEnum], Optional[str]]:
    """Sub-part 3: Deterministic inline metric canonicalization against METRIC_LOOKUP (Section 6.1 Step 5).
    
    Returns (selected_metrics, error_message). Defaults to all 5 foundational metrics if unspecified.
    """
    cleaned_request = user_request.lower()

    # Check for explicit requests for unsupported metrics
    for unsup in UNSUPPORTED_METRICS:
        pattern = r"\b" + re.escape(unsup) + r"\b"
        if re.search(pattern, cleaned_request):
            return [], f"Unsupported connectivity metric requested: '{unsup}'. The pipeline strictly supports only: PLI, wPLI, ImCoh, PLV, Coherence."

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
# Condition resolution (Token-boundary, collision-free & recency-aware)
# ---------------------------------------------------------------------------

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

    Uses token-boundary matching to prevent substring collisions (e.g. 'A' matching 'AB'
    or 'task' matching 'task_motor').
    If multiple distinct conditions match ambiguously in the active turn window, returns
    None to require human clarification rather than guessing.
    """
    if not conditions:
        return None

    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())

    def _find_boundary_matches(haystack_norm: str) -> List[Tuple[int, str]]:
        matches = []
        for cond in conditions:
            cond_norm = norm(cond)
            # Match cond_norm on token boundaries (not preceded or followed by alphanumeric/underscore)
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(cond_norm) + r"(?![A-Za-z0-9_])"
            for m in re.finditer(pattern, haystack_norm):
                matches.append((m.start(), cond))
        # Sort by match position in text
        matches.sort(key=lambda x: x[0])
        return matches

    recent_window_norm = norm(_condition_search_window(user_request))
    recent_matches = _find_boundary_matches(recent_window_norm)

    if recent_matches:
        # Check if multiple distinct conditions matched in this recent window
        distinct_conds = set(c for _, c in recent_matches)
        if len(distinct_conds) > 1:
            # Ambiguous: multiple conditions mentioned in the same turn -> do not guess!
            return None
        # Unambiguous match in the recent turn
        return recent_matches[-1][1]

    # Fall back to searching full user_request if recent window did not mention any condition
    full_norm = norm(user_request)
    full_matches = _find_boundary_matches(full_norm)
    if full_matches:
        # Check distinct conditions matched
        distinct_conds = set(c for _, c in full_matches)
        if len(distinct_conds) > 1:
            # If earlier turns had multiple conditions without revision tags disambiguating,
            # check if the last mention is at a strictly later position than all other conditions
            last_pos, last_cond = full_matches[-1]
            same_pos_conds = set(c for pos, c in full_matches if pos == last_pos)
            if len(same_pos_conds) > 1:
                return None
            return last_cond
        return full_matches[-1][1]

    return None


def compile_and_confirm_manifest(
    plan: AnalysisPlan,
    tool_confidences: Dict[str, float],
    trial_count: Optional[int] = None,
    discovered_reference: Optional[str] = None,
    reference: Optional[str] = None,
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

    # 4. Reference Scheme (Risk Tier: elevated, Section 3 Constraint 5)
    ref_value = reference if reference else (discovered_reference if discovered_reference else "average")
    manifest.append(
        ParameterManifestEntry(
            name="reference",
            category="engineering_threshold",
            proposed_value=ref_value,
            confidence=None,
            needs_human_input=False,
            risk_tier="elevated",
        )
    )

    # 5. Quality-Gate Thresholds (Risk Tier: elevated, Section 6.2)
    manifest.append(
        ParameterManifestEntry(
            name="bad_channel_variance_threshold",
            category="engineering_threshold",
            proposed_value=str(BAD_CHANNEL_VARIANCE_THRESHOLD),
            confidence=None,
            needs_human_input=False,
            risk_tier="elevated",
        )
    )

    manifest.append(
        ParameterManifestEntry(
            name="min_cycles",
            category="engineering_threshold",
            proposed_value=str(MIN_CYCLES),
            confidence=None,
            needs_human_input=False,
            risk_tier="elevated",
        )
    )

    # 6. Coupling Classification Thresholds (Risk Tier: elevated, Section 4.2 & 6.3)
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

    # 7. Artifact Detection Thresholds (Risk Tier: elevated, Section 6.4)
    manifest.append(
        ParameterManifestEntry(
            name="ocular_variance_ratio",
            category="engineering_threshold",
            proposed_value=str(OCULAR_VARIANCE_RATIO),
            confidence=None,
            needs_human_input=False,
            risk_tier="elevated",
        )
    )

    manifest.append(
        ParameterManifestEntry(
            name="muscle_power_threshold",
            category="engineering_threshold",
            proposed_value=str(MUSCLE_POWER_THRESHOLD),
            confidence=None,
            needs_human_input=False,
            risk_tier="elevated",
        )
    )

    manifest.append(
        ParameterManifestEntry(
            name="electrode_pop_sigma",
            category="engineering_threshold",
            proposed_value=str(ELECTRODE_POP_SIGMA),
            confidence=None,
            needs_human_input=False,
            risk_tier="elevated",
        )
    )

    # 8. Trial Adequacy Check (Scientific Completeness Advisory)
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

    # 9. Cross-Metric Synthesis Advisory (Section 10: Option b with Guardrails)
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

    # 10. Dataset Overview Plot (Informational / Diagnostic — not a scientific axis)
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

    # 11. Bad Channel Screening Status (Informational)
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


# Providers such as Groq validate tool calls server-side: a call to a tool that was
# not bound for this request is rejected with HTTP 400 ("tool_use_failed") instead
# of being returned to us. That is the provider-side twin of the TOOL_OUT_OF_SCOPE
# guard below, so it is recovered the same way (tell the model, let it retry).
_MAX_TOOL_REJECTIONS = 2
_TOOL_REJECTION_PATTERN = re.compile(
    r"tool_use_failed|not in request\.tools|tool call validation failed", re.IGNORECASE
)
_REJECTED_TOOL_NAME = re.compile(r"attempted to call tool '([^']+)'")


def _is_provider_tool_rejection(err: Exception) -> bool:
    return bool(_TOOL_REJECTION_PATTERN.search(str(err)))


def _rejected_tool_name(err: Exception) -> str:
    m = _REJECTED_TOOL_NAME.search(str(err))
    return m.group(1) if m else "unknown"


def _tool_rejection_notice(tool_name: str, available: List[str]) -> str:
    listing = ", ".join(available) if available else "none"
    return (
        f"[system notice] Your last tool call to '{tool_name}' was rejected: that tool is not "
        "available at this step (it is already done, or not unlocked yet). "
        f"Tools available now: {listing}. Do not call '{tool_name}' again. "
        "Call one of the available tools, or reply in plain text."
    )


_METRIC_WORD = re.compile(r"metric", re.IGNORECASE)


def _strip_metric_prompts(text: str) -> str:
    """Drop any line of an LLM-written clarification that talks about metrics.

    Metrics are never a question to the user: they default to all five and are
    shown (and editable) at Gate 1. Only the three scientific axes (condition,
    frequency band, channels) may be asked for up front.
    """
    kept = [ln for ln in str(text or "").splitlines() if not _METRIC_WORD.search(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _unresolved_axes_question(band_ok: bool, channels_ok: bool, condition_ok: bool) -> str:
    """Deterministic clarification text naming only the unresolved axes."""
    missing = []
    if not condition_ok:
        missing.append("**Condition** - which condition label in the recording should be used")
    if not band_ok:
        missing.append("**Frequency band** - for example alpha, or a range such as 8-12 Hz")
    if not channels_ok:
        missing.append("**Channels** - at least two electrodes, or a brain region")
    if not missing:
        return "Please provide the missing analysis parameter(s)."
    return "Please specify the following for the analysis:\n\n" + "\n".join(
        f"{i}. {m}" for i, m in enumerate(missing, 1)
    )


def _single_condition_question(label: Any) -> str:
    return (
        f"This recording contains only one condition label: **{label}**. "
        "Please confirm you want to analyze it, or type a different label. "
        "With a single label, comparing conditions within this file is not possible."
    )


def _condition_candidates_from_request(user_request: str, conditions: List[str]) -> List[str]:
    """Return all dataset conditions mentioned in the active request.

    This is intentionally conservative: if multiple labels are present, the
    UI can present them as explicit human choices instead of asking the user
    to retype exact event labels.
    """
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    window = norm(_condition_search_window(user_request))
    matches: List[Tuple[int, str]] = []
    for cond in conditions:
        c = norm(cond)
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(c) + r"(?![A-Za-z0-9_])"
        m = re.search(pattern, window)
        if m:
            matches.append((m.start(), cond))
    matches.sort(key=lambda x: x[0])
    return [c for _, c in matches]


def _valid_frequency_band_options(sfreq: Optional[float], duration_seconds: Optional[float]) -> List[Dict[str, str]]:
    """Build bounded canonical-band buttons that pass deterministic validation."""
    options: List[Dict[str, str]] = []
    if not sfreq or sfreq <= 0:
        return options
    nyquist = sfreq / 2.0
    for name, bounds in CANONICAL_BANDS.items():
        if bounds["fmax"] >= nyquist:
            continue
        if duration_seconds and duration_seconds > 0:
            # Same minimum-cycle rule as resolve_frequency_band, without invoking
            # the tool solely to construct UI choices.
            if duration_seconds < MIN_CYCLES / bounds["fmin"]:
                continue
        options.append({
            "name": name,
            "value": name,
            "label": f"{name.title()} ({bounds['fmin']:.0f}–{bounds['fmax']:.0f} Hz)",
        })
    return options


def _channel_options(available_channels: Optional[List[str]], max_buttons: int = 12) -> List[Dict[str, str]]:
    """Build deterministic bounded EEG channel HITL choices.

    The UI must *always* have a bounded choice when the dataset exposes a
    usable electrode list.  Small datasets get representative channel-pair
    buttons; larger datasets get anatomical-region buttons backed by the
    actual electrode labels.  Normalisation intentionally mirrors
    ``resolve_channel_selection`` so labels such as ``EEG F3-REF`` and
    ``Fp1`` are recognised consistently.
    """
    channels = [str(ch).strip() for ch in (available_channels or []) if str(ch).strip()]
    if len(channels) < 2:
        return []

    clean_to_raw: Dict[str, str] = {}
    for raw in channels:
        clean_to_raw.setdefault(normalize_channel_label(raw), raw)

    # For small electrode sets, show actual pairs from the dataset.
    if len(channels) <= max_buttons:
        options: List[Dict[str, str]] = []
        for i in range(0, len(channels) - 1, 2):
            pair = channels[i:i + 2]
            if len(pair) == 2:
                options.append({
                    "name": f"channels_{i}",
                    "value": ", ".join(pair),
                    "label": f"{pair[0]} + {pair[1]}",
                })
        return options

    # For larger datasets, use the same anatomical map as the resolver.
    # A region button is only emitted if at least two *real* electrodes from
    # that region exist in this dataset.
    options: List[Dict[str, str]] = []
    seen_regions = set()
    for region, region_channels in REGION_MAP.items():
        matched = [clean_to_raw[ch] for ch in region_channels if ch in clean_to_raw]
        if len(matched) >= 2 and region not in seen_regions:
            seen_regions.add(region)
            preview_names = [normalize_channel_label(ch) for ch in matched[:4]]
            preview = ", ".join(preview_names)
            if len(matched) > 4:
                preview += "…"
            options.append({
                "name": f"region_{region}",
                "value": region,
                "label": f"{region.title()} region ({preview})",
            })

    # If the dataset uses a standard EEG montage but its exact labels do not
    # map to REGION_MAP (e.g. extended 10-10 names), still provide bounded
    # electrode-pair choices from the actual data. Never invent labels.
    if not options:
        usable = list(clean_to_raw.items())
        for i in range(0, min(len(usable), 12), 2):
            pair = usable[i:i + 2]
            if len(pair) == 2:
                options.append({
                    "name": f"electrodes_{i}",
                    "value": f"{pair[0][1]}, {pair[1][1]}",
                    "label": f"{normalize_channel_label(pair[0][1])} + {normalize_channel_label(pair[1][1])}",
                })

    return options


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

    # Sub-part 1: Tool isolation — initialize dynamic binding trackers
    bound_tool_names: set[str] = set()
    llm_with_tools = llm

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Dataset Path: {raw_data_path}\n"
                f"Run ID: {run_id}\n"
                f"<user_request>\n{user_request}\n</user_request>"
            )
        ),
    ]

    tool_confidences: Dict[str, float] = {}
    # Carry successful axis resolutions across native HITL resumes. The prior
    # implementation recreated these locals on every clarification response,
    # which made an already-valid selection (e.g. Theta) appear unresolved.
    resolved_band_info: Optional[Dict[str, Any]] = state.get("resolved_frequency_band_info")
    resolved_channel_info: Optional[Dict[str, Any]] = state.get("resolved_channel_info")
    resolved_condition: Optional[str] = state.get("resolved_condition_value")
    if resolved_band_info and "confidence" in resolved_band_info:
        tool_confidences["frequency_band"] = resolved_band_info["confidence"]
    if resolved_channel_info and "confidence" in resolved_channel_info:
        tool_confidences["channels"] = resolved_channel_info["confidence"]
    if resolved_condition:
        tool_confidences["condition"] = 1.0
    resolved_trial_count: Optional[int] = None
    resolved_condition_counts: Dict[str, int] = {}
    conditions_tool_called = False
    clarification_question: Optional[str] = None
    clarification_kind: Optional[str] = None
    clarification_options: List[Dict[str, str]] = []
    condition_candidates: List[str] = []
    attempted_frequency_band: bool = False
    attempted_channel_selection: bool = False
    last_band_error: Optional[str] = None
    last_unresolved_channels: List[str] = []
    informational_tool_outputs: Dict[str, Any] = {}
    # Keep dataset metadata available across native HITL interrupts so the UI
    # can always build bounded choices from the actual loaded EEG.
    dataset_sfreq = state.get("dataset_sfreq")
    dataset_duration_seconds = state.get("dataset_duration_seconds")
    dataset_available_channels = state.get("dataset_available_channels")
    dataset_reference = state.get("dataset_reference")

    loop_exhausted: bool = True
    response: Optional[AIMessage] = None  # sentinel; guards post-loop `response and ...` checks

    def _direct_tool(t_name: str, t_args: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic (non-LLM) tool call with the same trace events as the ReAct loop."""
        tracer.log_event("tool_call", {"tool": t_name, "args": t_args})
        try:
            with trace_span(name=f"tool:{t_name}", span_type="TOOL", inputs={"tool": t_name, "args": t_args}):
                obs = TOOL_MAP[t_name].invoke(t_args)
        except Exception as exc:  # never let UI-support lookups crash the halt
            obs = {"error": f"{type(exc).__name__}: {exc}"}
        tracer.log_event("tool_observation", {"tool": t_name, "observation": obs})
        return obs

    def _capture_dataset_info(obs: Dict[str, Any]) -> None:
        nonlocal dataset_sfreq, dataset_duration_seconds, dataset_available_channels, dataset_reference
        if not isinstance(obs, dict):
            return
        if obs.get("duration_seconds", 0) and obs["duration_seconds"] > 0:
            dataset_duration_seconds = obs["duration_seconds"]
        if obs.get("sfreq", 0) and obs["sfreq"] > 0:
            dataset_sfreq = obs["sfreq"]
        if obs.get("available_channels"):
            dataset_available_channels = obs["available_channels"]
        if obs.get("reference"):
            dataset_reference = obs["reference"]

    def _capture_conditions(obs: Dict[str, Any], resolve: bool = True) -> None:
        nonlocal conditions_tool_called, resolved_condition_counts, resolved_trial_count
        nonlocal condition_candidates, resolved_condition
        if not (isinstance(obs, dict) and "conditions" in obs):
            return
        conditions_tool_called = True
        resolved_condition_counts = obs["conditions"]
        if resolved_trial_count is None and "total_trials" in obs:
            resolved_trial_count = obs["total_trials"]
        condition_candidates = _condition_candidates_from_request(
            user_request, list(obs.get("conditions", {}).keys())
        )
        if resolve and resolved_condition is None:
            match = resolve_condition_from_request(user_request, obs["conditions"])
            if match is not None:
                resolved_condition = match
                tool_confidences["condition"] = 1.0

    # HITL resume: a Supervisor axis (band / channels / condition) was just
    # answered, validated and persisted by clarification_pause.  Do not ask the
    # LLM to re-derive it; re-read the (cheap, deterministic) dataset metadata,
    # keep the persisted axes and let the halt logic below ask ONLY for the next
    # unresolved axis.
    resuming = (
        state.get("clarification_response") is not None
        and state.get("clarification_resume_kind") in AXIS_KINDS
    )
    if resuming:
        if not (dataset_sfreq and dataset_available_channels):
            _capture_dataset_info(_direct_tool("get_dataset_info", {"data_path": raw_data_path}))
        _capture_conditions(_direct_tool("get_dataset_conditions", {"data_path": raw_data_path}))
        attempted_frequency_band = resolved_band_info is not None
        attempted_channel_selection = resolved_channel_info is not None
        loop_exhausted = False

    # Sub-part 1: Scoping helper driven by registry metadata (deliberate hybrid per Decision C)
    def _compute_scoped_tools() -> List[Any]:
        """Compute the set of tools available to the LLM at this point in the ReAct loop.

        Deliberate hybrid per Decision C: uses per-tool exclusion rules, but reads allowed_agents,
        execution_mode, and pipeline_stage from toolbox_registry metadata, and guarantees
        never to return anything outside toolbox_registry.get_scoped_llm_tools(agent="supervisor").
        """
        all_supervisor_tools = toolbox_registry.get_scoped_llm_tools(agent="supervisor")
        scoped: List[Any] = []

        # Whole-word regex overview plot intent per Decision D
        # Note: will need revisiting when Node 3 adds connectivity plotting.
        has_plot_intent = bool(
            re.search(
                r"\b(plot|plots|psd|spectrum|spectrogram|visualize|visualization|overview)\b",
                user_request or "",
                re.IGNORECASE,
            )
        )
        plot_completed = "generate_dataset_overview_plot" in informational_tool_outputs

        dataset_loaded = bool(dataset_sfreq and dataset_available_channels)
        conditions_done = bool(conditions_tool_called or resolved_condition)

        for tool_fn in all_supervisor_tools:
            t_name = getattr(tool_fn, "name", "")
            meta = toolbox_registry.get_metadata(t_name)

            if "get_dataset_info" in meta.requires_prerequisites and not dataset_loaded:
                continue

            if meta.pipeline_stage == "discovery":
                if t_name == "get_dataset_info":
                    if not dataset_loaded:
                        scoped.append(tool_fn)
                elif t_name == "get_dataset_conditions":
                    if not conditions_done:
                        scoped.append(tool_fn)

            elif meta.pipeline_stage == "resolution":
                if t_name == "resolve_frequency_band":
                    if not (resolved_band_info and not resolved_band_info.get("error")):
                        scoped.append(tool_fn)
                elif t_name == "resolve_channel_selection":
                    if not (resolved_channel_info and not resolved_channel_info.get("error")):
                        scoped.append(tool_fn)

            elif meta.pipeline_stage == "diagnostic":
                if t_name == "generate_dataset_overview_plot":
                    if has_plot_intent and not plot_completed:
                        scoped.append(tool_fn)

        return scoped

    # Sub-part 2: The ReAct Execution Loop
    tool_rejections = 0
    for iteration in range(0 if resuming else max_iterations):
        # Compute scoped tools for this iteration
        current_scoped_tools = _compute_scoped_tools()
        current_scoped_names = {getattr(t, "name", "") for t in current_scoped_tools}

        # Decision B: Empty scope check
        # When all axes are resolved and no requested plot is pending, skip LLM call and break directly
        if not current_scoped_tools:
            loop_exhausted = False
            break

        # Rebind only when the set of tool names changes
        if current_scoped_names != bound_tool_names:
            llm_with_tools = llm.bind_tools(current_scoped_tools)
            bound_tool_names = current_scoped_names

        # Decision A: Strict snapshot guard
        # Validate tool calls in response against scope snapshot before this LLM invocation
        iteration_scoped_tool_names = set(bound_tool_names)
        tracer.log_event("iteration_tool_scope", {
            "iteration": iteration + 1,
            "scoped_tools": sorted(list(iteration_scoped_tool_names)),
        })

        try:
            response: AIMessage = llm_with_tools.invoke(messages)
        except (ValueError, Exception) as _llm_err:
            # Groq (and other providers) can return a completely empty response
            # after rate-limit retries, which LangChain raises as:
            #   ValueError: model output must contain either output text or tool calls
            # Treat this as a graceful loop exit — the outer halt logic will
            # emit a clarification question instead of crashing the pipeline.
            _err_str = str(_llm_err)
            if _is_provider_tool_rejection(_llm_err):
                tool_rejections += 1
                _rejected = _rejected_tool_name(_llm_err)
                tracer.log_event("tool_scope_violation", {
                    "tool": _rejected,
                    "scoped_tools": sorted(list(iteration_scoped_tool_names)),
                    "source": "provider_rejection",
                    "attempt": tool_rejections,
                })
                if tool_rejections > _MAX_TOOL_REJECTIONS:
                    # The model keeps asking for an unavailable tool: leave the loop
                    # and let the halt logic below ask the user for what is unresolved.
                    loop_exhausted = True
                    break
                messages.append(
                    HumanMessage(
                        content=_tool_rejection_notice(_rejected, sorted(list(iteration_scoped_tool_names)))
                    )
                )
                continue
            if "model output" in _err_str or "tool calls" in _err_str or "output text" in _err_str:
                tracer.log_event("llm_empty_response", {
                    "iteration": iteration + 1,
                    "error": _err_str,
                })
                loop_exhausted = True
                break
            raise  # re-raise unexpected errors

        # Normalise content — some providers return None instead of ""
        if response.content is None:
            response = response.model_copy(update={"content": ""})

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

                # Decision A: Strict snapshot scope guard before action policy check
                if t_name not in iteration_scoped_tool_names:
                    observation = {
                        "error": (
                            f"TOOL_OUT_OF_SCOPE: Tool '{t_name}' is not accessible at this stage. "
                            f"Available tools: {sorted(list(iteration_scoped_tool_names))}"
                        )
                    }
                    tracer.log_event("tool_scope_violation", {
                        "tool": t_name,
                        "scoped_tools": sorted(list(iteration_scoped_tool_names)),
                    })
                    call_id = tool_call.get("id", "fallback_id")
                    messages.append(ToolMessage(tool_call_id=call_id, content=str(observation)))
                    continue

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
                # Deterministically pass actual duration and sfreq to resolve_frequency_band (NEVER fabricate duration)
                actual_duration = dataset_duration_seconds or state.get("dataset_duration_seconds")
                actual_sfreq = dataset_sfreq or state.get("dataset_sfreq")
                actual_channels = dataset_available_channels or state.get("dataset_available_channels")

                if t_name == "resolve_frequency_band":
                    if "duration_seconds" not in t_args or t_args.get("duration_seconds") is None:
                        t_args["duration_seconds"] = actual_duration
                    if actual_sfreq and ("sfreq" not in t_args or t_args.get("sfreq") != actual_sfreq):
                        t_args["sfreq"] = actual_sfreq
                elif t_name == "resolve_channel_selection":
                    if actual_channels:
                        t_args["available_channels"] = actual_channels

                # PreToolUse-style policy check (see action_policy.evaluate_action).
                # All current tools are read-only/reversible -> allow. ask_human is
                # not honoured inside this loop (interrupt() would replay the whole
                # node, re-running LLM calls), so it is treated as a deny here.
                policy_block: Optional[str] = None
                if tool_fn:
                    tool_decision = evaluate_action(t_name, t_args, {"run_id": run_id})
                    if (
                        tool_decision.behavior == DecisionType.ALLOW_WITH_MODIFIED_INPUT
                        and tool_decision.updated_input is not None
                    ):
                        t_args = dict(tool_decision.updated_input)
                    elif tool_decision.behavior != DecisionType.ALLOW:
                        policy_block = tool_decision.reason or "blocked by action policy"

                if policy_block is not None:
                    observation = {"error": f"Tool '{t_name}' blocked by action policy: {policy_block}"}
                elif tool_fn:
                    with trace_span(
                        name=f"tool:{t_name}",
                        span_type="TOOL",
                        inputs={"tool": t_name, "args": t_args},
                    ):
                        observation = tool_fn.invoke(t_args)
                else:
                    observation = {"error": f"Tool '{t_name}' not found."}

                # Capture per-condition trial counts for manifest compilation.
                if t_name == "get_dataset_conditions" and "conditions" in observation:
                    conditions_tool_called = True
                    resolved_condition_counts = observation["conditions"]
                    if resolved_trial_count is None and "total_trials" in observation:
                        resolved_trial_count = observation["total_trials"]
                # Capture metadata from get_dataset_info for downstream tools and manifest
                if t_name == "get_dataset_info":
                    if "duration_seconds" in observation and observation["duration_seconds"] > 0:
                        state["dataset_duration_seconds"] = observation["duration_seconds"]
                        dataset_duration_seconds = observation["duration_seconds"]
                    if "sfreq" in observation and observation["sfreq"] > 0:
                        state["dataset_sfreq"] = observation["sfreq"]
                        dataset_sfreq = observation["sfreq"]
                    if "available_channels" in observation and observation["available_channels"]:
                        state["dataset_available_channels"] = observation["available_channels"]
                        dataset_available_channels = observation["available_channels"]
                    if "reference" in observation and observation["reference"]:
                        state["dataset_reference"] = observation["reference"]
                        dataset_reference = observation["reference"]

                tracer.log_event("tool_observation", {"tool": t_name, "observation": observation})

                # Capture raw output of successful informational/diagnostic tool calls
                if t_name in INFORMATIONAL_TOOLS and not (
                    isinstance(observation, dict) and observation.get("error")
                ):
                    informational_tool_outputs[t_name] = observation

                # Capture resolution metadata for manifest compilation
                if t_name == "resolve_frequency_band" and "confidence" in observation:
                    tool_confidences["frequency_band"] = observation["confidence"]
                    if not observation.get("error"):
                        resolved_band_info = observation
                        last_band_error = None
                    else:
                        last_band_error = str(observation["error"])
                elif t_name == "resolve_channel_selection" and "confidence" in observation:
                    tool_confidences["channels"] = observation["confidence"]
                    if not observation.get("error"):
                        resolved_channel_info = observation
                    else:
                        last_unresolved_channels = list(observation.get("unresolved_channels") or [])
                elif t_name == "get_dataset_conditions" and "conditions" in observation:
                    condition_candidates = _condition_candidates_from_request(
                        user_request, list(observation.get("conditions", {}).keys())
                    )
                    match = resolve_condition_from_request(user_request, observation["conditions"])
                    if match is not None:
                        resolved_condition = match
                        tool_confidences["condition"] = 1.0

                # Append tool observation back to messages
                call_id = tool_call.get("id", "fallback_id")
                messages.append(ToolMessage(tool_call_id=call_id, content=str(observation)))
        else:
            # No tool calls: LLM finished reasoning
            loop_exhausted = False
            break

    # Sub-part 3: Structural Guardrail Evaluation across all 3 Scientific Axes
    band_ok = bool(resolved_band_info and not resolved_band_info.get("error"))
    channels_ok = bool(resolved_channel_info and not resolved_channel_info.get("error"))
    condition_ok = bool(resolved_condition)

    axis_resolution_attempted = (
        attempted_frequency_band or attempted_channel_selection or resolved_condition is not None
    )

    if not (band_ok and channels_ok and condition_ok):
        # Genuine informational requests are those where the user asked for info/plots,
        # NOT an analysis request where parameters are missing or the LLM is asking for clarification.
        informational_text = (
            response.content.strip()
            if (response and isinstance(response.content, str))
            else ""
        )
        clarification_indicators = [
            "clarif", "specify", "missing", "please specify", "lacks", "require",
            "not specified", "unspecified", "frequency band", "target condition",
        ]
        has_clarif_text = any(ci in informational_text.lower() for ci in clarification_indicators)
        user_req_lc = (user_request or "").lower()
        is_analysis_query = any(k in user_req_lc for k in [
            "analy", "compute", "calc", "pipeline", "fc", "functional connectivity",
            "metric", "pli", "wpli", "coherence"
        ])
        has_overview_plot = "generate_dataset_overview_plot" in informational_tool_outputs

        is_pure_informational = (
            not resuming
            and not loop_exhausted
            and not axis_resolution_attempted
            and informational_tool_outputs
            and not has_clarif_text
            and (has_overview_plot or not is_analysis_query)
        )

        if is_pure_informational:
            if not informational_text or informational_text.startswith("[TOOL_CALLS]"):
                informational_text = "Request completed."

            tracer.log_event("supervisor_informational_complete", {
                "response": informational_text,
                "tools": list(informational_tool_outputs.keys()),
            })
            return {
                **legacy_fields(None),
                "condition_candidates": condition_candidates,
                "clarification_response": None,
                "resolved_frequency_band_info": resolved_band_info,
                "resolved_channel_info": resolved_channel_info,
                "resolved_condition_value": resolved_condition,
                "plan": None,
                "parameter_manifest": None,
                "preflight_confirmed": False,
                "informational_response": informational_text,
                "informational_artifacts": informational_tool_outputs,
                "dataset_sfreq": dataset_sfreq,
                "dataset_duration_seconds": dataset_duration_seconds,
                "dataset_available_channels": dataset_available_channels,
                "dataset_reference": dataset_reference,
            }

        # Genuine halt: an axis was attempted (or partially resolved) and is still missing.
        # HITL choices must come from the ACTUAL loaded dataset, never from whatever the
        # LLM happened to call: fetch any metadata the ReAct loop skipped, deterministically.
        if not (dataset_sfreq and dataset_available_channels):
            _capture_dataset_info(_direct_tool("get_dataset_info", {"data_path": raw_data_path}))
        axes_engaged = bool(
            attempted_frequency_band or attempted_channel_selection
            or resolved_band_info or resolved_channel_info
        )
        if not condition_ok and not conditions_tool_called and axes_engaged:
            # Options only: resolution stays with the tool loop / persisted HITL answer.
            _capture_conditions(
                _direct_tool("get_dataset_conditions", {"data_path": raw_data_path}),
                resolve=False,
            )

        last_thought = response.content.strip() if (response and isinstance(response.content, str)) else ""
        if loop_exhausted:
            clarification_kind = "max_iterations"
            clarification_question = (
                "Maximum reasoning iterations were reached without resolving all analysis parameters. "
                "Please specify the missing frequency band, channels, or condition."
            )
        elif not band_ok and (attempted_frequency_band or state.get("dataset_sfreq")):
            clarification_kind = "frequency_band"
            band_err = last_band_error
            if band_err:
                clarification_question = f"{band_err} Please choose a valid physiological frequency band or type a custom numeric range."
            else:
                clarification_question = "Please choose a physiological frequency band or type a custom numeric range."
            clarification_options = _valid_frequency_band_options(
                dataset_sfreq, dataset_duration_seconds
            )
        elif not channels_ok and (attempted_channel_selection or state.get("dataset_available_channels")):
            clarification_kind = "channel_selection"
            channel_err = (
                f"Unrecognized channel label(s) not found in dataset: {', '.join(last_unresolved_channels)}."
                if last_unresolved_channels else None
            )
            if channel_err:
                clarification_question = f"{channel_err} Choose from the available channels below, or type a channel list/brain region."
            else:
                clarification_question = "Please choose at least two EEG channels or type a valid brain region."
            clarification_options = _channel_options(dataset_available_channels)
        elif not condition_ok and (conditions_tool_called or condition_candidates):
            clarification_kind = "condition"
            clarification_question = "Could not resolve a valid experimental condition from the dataset. Choose one below or type the exact condition label."
            if not condition_candidates:
                condition_candidates = list(resolved_condition_counts.keys())
            if len(condition_candidates) == 1:
                clarification_question = _single_condition_question(condition_candidates[0])
            clarification_options = [
                {"name": f"condition_{i}", "value": c, "label": c}
                for i, c in enumerate(condition_candidates)
            ]
        elif last_thought and not last_thought.startswith("[TOOL_CALLS]"):
            clarification_kind = "clarification"
            clarification_question = _strip_metric_prompts(last_thought) or _unresolved_axes_question(
                band_ok, channels_ok, condition_ok
            )
        else:
            clarification_kind = "clarification"
            clarification_question = "Please provide the missing analysis parameter(s)."

        tracer.log_event("supervisor_halt_unresolved_axes", {
            "question": clarification_question,
            "band_ok": band_ok,
            "channels_ok": channels_ok,
            "condition_ok": condition_ok,
        })
        clarification = build_clarification(
            clarification_kind, clarification_question, clarification_options
        )
        return {
            **legacy_fields(clarification),
            "condition_candidates": condition_candidates,
            "clarification_response": None,
            "resolved_frequency_band_info": resolved_band_info,
            "resolved_channel_info": resolved_channel_info,
            "resolved_condition_value": resolved_condition,
            "plan": None,
            "parameter_manifest": None,
            "preflight_confirmed": False,
            "informational_response": None,
            "informational_artifacts": None,
            "dataset_sfreq": dataset_sfreq,
            "dataset_duration_seconds": dataset_duration_seconds,
            "dataset_available_channels": dataset_available_channels,
            "dataset_reference": dataset_reference,
        }

    # Resolve trial count for specific matched condition
    if resolved_condition in resolved_condition_counts:
        resolved_trial_count = resolved_condition_counts[resolved_condition]

    # Inline Metric Canonicalization (Sub-part 3)
    metrics, metric_err = canonicalize_metrics_inline(user_request)
    if metric_err:
        return {
            **legacy_fields(build_clarification("metric_selection", metric_err)),
            "condition_candidates": condition_candidates,
            "clarification_response": None,
            "resolved_frequency_band_info": resolved_band_info,
            "resolved_channel_info": resolved_channel_info,
            "resolved_condition_value": resolved_condition,
            "plan": None,
            "parameter_manifest": None,
            "preflight_confirmed": False,
            "informational_response": None,
            "informational_artifacts": None,
            "dataset_sfreq": dataset_sfreq,
            "dataset_duration_seconds": dataset_duration_seconds,
            "dataset_available_channels": dataset_available_channels,
            "dataset_reference": dataset_reference,
        }

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
    discovered_ref = state.get("dataset_reference")
    manifest, validation_error = compile_and_confirm_manifest(
        plan, tool_confidences, resolved_trial_count, discovered_reference=discovered_ref
    )
    if validation_error:
        tracer.log_event("structural_validation_failed", {"error": validation_error})
        return {
            **legacy_fields(build_clarification("validation_error", validation_error)),
            "condition_candidates": condition_candidates,
            "plan": None,
            "parameter_manifest": None,
            "preflight_confirmed": False,
            "informational_response": None,
            "informational_artifacts": None,
            "dataset_sfreq": dataset_sfreq,
            "dataset_duration_seconds": dataset_duration_seconds,
            "dataset_available_channels": dataset_available_channels,
            "dataset_reference": dataset_reference,
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
        **legacy_fields(None),
        "condition_candidates": condition_candidates,
        "clarification_response": None,
        "resolved_frequency_band_info": resolved_band_info,
        "resolved_channel_info": resolved_channel_info,
        "resolved_condition_value": resolved_condition,
        "plan": plan,
        "parameter_manifest": manifest,
        "preflight_confirmed": False,
        "informational_response": None,
        "informational_artifacts": None,
        "dataset_sfreq": dataset_sfreq,
        "dataset_duration_seconds": dataset_duration_seconds,
        "dataset_available_channels": dataset_available_channels,
        "dataset_reference": dataset_reference,
    }
