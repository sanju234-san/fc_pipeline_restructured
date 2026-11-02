"""Deterministic connectivity analysis module.

Re-exports spectral connectivity estimation, classification, plotting, and execution.
"""

from fc_pipeline.deterministic.connectivity.classification import (
    classify_channel_pairs,
    export_connectivity_matrices_to_csv,
    export_evidence_summary_to_csv,
)
from fc_pipeline.deterministic.connectivity.executor import (
    run_connectivity_analysis,
)
from fc_pipeline.deterministic.connectivity.metrics import (
    ConnectivityComputationError,
    compute_metric_numerical_summaries,
    compute_spectral_connectivity,
)
from fc_pipeline.deterministic.connectivity.models import (
    ConnectivityInput,
    ConnectivityResult,
)
from fc_pipeline.deterministic.connectivity.plotting import (
    render_all_connectivity_plots,
    render_connectivity_heatmap,
    render_network_graph,
)

__all__ = [
    "compute_spectral_connectivity",
    "compute_metric_numerical_summaries",
    "ConnectivityComputationError",
    "classify_channel_pairs",
    "export_connectivity_matrices_to_csv",
    "export_evidence_summary_to_csv",
    "render_connectivity_heatmap",
    "render_network_graph",
    "render_all_connectivity_plots",
    "ConnectivityInput",
    "ConnectivityResult",
    "run_connectivity_analysis",
]
