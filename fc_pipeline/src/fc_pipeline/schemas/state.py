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

    # 1. Supervisor Outputs
    clarification_question: Optional[str]
    plan: Optional[AnalysisPlan]
    parameter_manifest: Optional[List[ParameterManifestEntry]]
    preflight_confirmed: bool

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
