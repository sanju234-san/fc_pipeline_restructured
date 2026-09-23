"""GraphState TypedDict defining shared LangGraph pipeline state."""

from typing import Any, Dict, List, Optional, TypedDict

from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan


class GraphState(TypedDict):
    # Pipeline Execution & Error Tracking
    pipeline_error: Optional[str]
    run_id: Optional[str]

    # Input & Dataset Reference
    raw_data_path: str
    user_request: str
    latest_user_message: Optional[str]

    # 1. Supervisor Outputs
    clarification_question: Optional[str]
    plan: Optional[AnalysisPlan]
    parameter_manifest: Optional[List[ParameterManifestEntry]]
    preflight_confirmed: bool
    gate_1_approved: bool

    # 1b. Supervisor Informational/Diagnostic Completion (no AnalysisPlan
    # required, nothing further to ask). Set when a request is fully
    # satisfied by informational/diagnostic tools alone (e.g. a bare
    # dataset overview plot, or "what conditions does this dataset have")
    # without ever needing the 3 mandatory scientific axes resolved.
    # `informational_response` is the Supervisor's natural-language answer;
    # `informational_artifacts` holds the raw observation dict(s) from the
    # informational tool call(s) keyed by tool name (e.g.
    # informational_artifacts["generate_dataset_overview_plot"]) so the
    # frontend can render artifacts (like the plot image) directly instead
    # of parsing paths out of free text. Both are reset to None on every
    # other Supervisor outcome (clarification halt, validation error, or a
    # resolved AnalysisPlan) so a stale value never lingers across turns.
    informational_response: Optional[str]
    informational_artifacts: Optional[Dict[str, Any]]

    # 1c. Generalized HITL decision layer (see agentic/supervisor/action_policy.py).
    # When evaluate_action() returns ask_human, `decision_context` describes what
    # the human is being asked to review ({"kind": "input_rail" | "output_rail" |
    # "manifest_review", "reason", "details", ...}) and gate_1_review pauses on it
    # via the existing interrupt(). None means the classic manifest-review gate.
    # After a rail pause is answered, gate_1_review adds "resolution":
    # "approved" | "rejected" for routing. `input_rail_cleared` records that a
    # human already cleared the NeMo input rail so it is not re-run on loop-back.
    decision_context: Optional[Dict[str, Any]]
    input_rail_cleared: bool

    # 2. Data Preparation Outputs (Placeholders for Node 2)
    bad_channels_dropped: Optional[List[str]]
    channel_plot_paths: Optional[Dict[str, str]]
    preprocessed_data_path: Optional[str]
    data_prep_error: Optional[str]

    # 3. Connectivity Analysis Outputs (Placeholders for Node 3)
    metric_csv_paths: Optional[Dict[str, str]]
    heatmap_image_paths: Optional[Dict[str, str]]
    network_image_paths: Optional[Dict[str, str]]
    numerical_summaries: Optional[Dict[str, Dict[str, float]]]
    evidence_summary: Optional[List[Any]]

    # 4. Evaluator Outputs (Placeholders for Node 4)
    evaluation_verdict: Optional[Any]
    evaluation_summary_csv_path: Optional[str]

    # 5. Final Synthesis Outputs (Placeholders for Node 5)
    final_report_path: Optional[str]
