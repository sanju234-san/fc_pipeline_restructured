"""Connectivity Agent orchestrator (Node 3).

Follows Appendix B.2 of PIPELINE_PROTOTYPE_DESIGN.md.
A pure deterministic orchestrator (ReAct-less): agents plan/orchestrate;
the LLM never computes scientific EEG numbers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fc_pipeline.deterministic.connectivity.executor import run_connectivity_analysis
from fc_pipeline.deterministic.connectivity.models import ConnectivityInput
from fc_pipeline.schemas.state import GraphState

logger = logging.getLogger(__name__)


def run_connectivity_agent(state: GraphState) -> Dict[str, Any]:
    """LangGraph node execution function for the Connectivity Analysis Agent.

    Parameters
    ----------
    state : GraphState
        Incoming graph state containing preprocessed_data_path, plan, manifest, run_id.

    Returns
    -------
    Dict[str, Any]
        State updates for metric_csv_paths, heatmap_image_paths, network_image_paths,
        numerical_summaries, and evidence_summary.
    """
    preprocessed_path = state.get("preprocessed_data_path")
    plan = state.get("plan")
    run_id = state.get("run_id") or "unknown"
    manifest = state.get("parameter_manifest") or []

    if not preprocessed_path:
        err = "MISSING_PREPROCESSED_DATA: GraphState contains no preprocessed_data_path."
        logger.error(err)
        return {
            "pipeline_error": err,
            "metric_csv_paths": {},
            "heatmap_image_paths": {},
            "network_image_paths": {},
            "numerical_summaries": {},
            "evidence_summary": [],
        }

    if not plan:
        err = "MISSING_ANALYSIS_PLAN: GraphState contains no AnalysisPlan."
        logger.error(err)
        return {
            "pipeline_error": err,
            "metric_csv_paths": {},
            "heatmap_image_paths": {},
            "network_image_paths": {},
            "numerical_summaries": {},
            "evidence_summary": [],
        }

    connectivity_input = ConnectivityInput(
        preprocessed_data_path=str(preprocessed_path),
        plan=plan,
        parameter_manifest=manifest,
        run_id=run_id,
    )

    logger.info("Executing Connectivity Agent for run_id=%s", run_id)
    result = run_connectivity_analysis(connectivity_input)

    if not result.success:
        logger.error("Connectivity Agent execution failed: %s", result.error)
        return {
            "pipeline_error": result.error,
            "metric_csv_paths": result.metric_csv_paths,
            "heatmap_image_paths": result.heatmap_image_paths,
            "network_image_paths": result.network_image_paths,
            "numerical_summaries": result.numerical_summaries,
            "evidence_summary": result.evidence_summary,
        }

    logger.info(
        "Connectivity Agent succeeded: %d metrics computed, %d pairs classified",
        len(result.numerical_summaries),
        len(result.evidence_summary),
    )

    return {
        "metric_csv_paths": result.metric_csv_paths,
        "heatmap_image_paths": result.heatmap_image_paths,
        "network_image_paths": result.network_image_paths,
        "numerical_summaries": result.numerical_summaries,
        "evidence_summary": result.evidence_summary,
        "consolidated_report_path": result.consolidated_report_path,
    }
