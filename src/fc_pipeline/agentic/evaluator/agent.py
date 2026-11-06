"""Evaluator coordinator orchestrating sanity checks, heuristics, and VLM.

Follows Section 6.4 of PIPELINE_PROTOTYPE_DESIGN.md.
Executes 4 sequential steps without mid-run interrupts:
  1. Deterministic matrix sanity checks
  2. Deterministic artifact detection heuristics
  3. Single-shot VLM topology reads for Top-5 unresolved zero-lag pairs
  4. CSV serialization to evaluation_summary_<run_id>.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fc_pipeline.agentic.evaluator.csv_writer import write_evaluation_summary_csv
from fc_pipeline.agentic.evaluator.prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    build_evaluator_user_prompt,
)
from fc_pipeline.agentic.evaluator.vlm_client import (
    build_vlm_messages,
    fallback_topology_reads,
    parse_vlm_topology_response,
)
from fc_pipeline.config.thresholds import (
    ELECTRODE_POP_SIGMA,
    MUSCLE_POWER_THRESHOLD,
    OCULAR_VARIANCE_RATIO,
    TAU_ZEROLAG,
)
from fc_pipeline.deterministic.artifact_detection import run_artifact_heuristics
from fc_pipeline.deterministic.sanity_checks import (
    evaluate_all_matrices_sanity,
    load_matrices_from_csv_paths,
)
from fc_pipeline.schemas.enums import EvidenceCategoryEnum, OverallVerdictEnum
from fc_pipeline.schemas.evaluation import EvaluationVerdict, SanityCheckFlags
from fc_pipeline.schemas.evidence import PairEvidenceSynthesis
from fc_pipeline.schemas.state import GraphState

logger = logging.getLogger(__name__)


def _extract_evaluator_thresholds(manifest: list) -> Dict[str, float]:
    """Reads artifact and zero-lag thresholds from Gate 1 manifest with config defaults."""
    thresholds = {
        "ocular_variance_ratio": OCULAR_VARIANCE_RATIO,
        "muscle_power_threshold": MUSCLE_POWER_THRESHOLD,
        "electrode_pop_sigma": ELECTRODE_POP_SIGMA,
        "tau_zerolag": TAU_ZEROLAG,
    }
    for entry in manifest or []:
        name = getattr(entry, "name", None) if not isinstance(entry, dict) else entry.get("name")
        if name not in thresholds:
            continue
        val_str = (
            (getattr(entry, "human_approved_value", None) or getattr(entry, "proposed_value", None))
            if not isinstance(entry, dict)
            else (entry.get("human_approved_value") or entry.get("proposed_value"))
        )
        if val_str is not None:
            try:
                thresholds[name] = float(val_str)
            except (ValueError, TypeError):
                pass
    return thresholds


def _determine_overall_verdict(
    per_metric_sanity: Dict[str, SanityCheckFlags],
    ocular_detected: bool,
    muscle_detected: bool,
    pop_detected: bool,
    n_unresolved_pairs: int,
) -> OverallVerdictEnum:
    """Computes PASS / WARNING / FAIL from sanity checks, artifacts, and unresolved pairs."""
    if not per_metric_sanity:
        return OverallVerdictEnum.WARNING

    for flags in per_metric_sanity.values():
        if not flags.values_in_bounds or not flags.matrix_is_symmetric:
            return OverallVerdictEnum.FAIL

    if all(flags.is_sparse_or_flat for flags in per_metric_sanity.values()):
        return OverallVerdictEnum.FAIL

    if (
        any(flags.is_sparse_or_flat for flags in per_metric_sanity.values())
        or ocular_detected
        or muscle_detected
        or pop_detected
        or n_unresolved_pairs > 0
    ):
        return OverallVerdictEnum.WARNING

    return OverallVerdictEnum.PASS


def run_evaluator_agent(
    state: GraphState,
    *,
    llm: Optional[Any] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executes Node 4 (Evaluator Agent) across all 4 sequential steps."""
    run_id = str(state.get("run_id") or "unknown")
    manifest = state.get("parameter_manifest") or []
    metric_csv_paths = state.get("metric_csv_paths") or {}
    heatmap_paths = state.get("heatmap_image_paths") or {}
    network_paths = state.get("network_image_paths") or {}
    evidence_raw = state.get("evidence_summary") or []

    evidence_list: List[PairEvidenceSynthesis] = [
        item if isinstance(item, PairEvidenceSynthesis) else PairEvidenceSynthesis.model_validate(item)
        for item in evidence_raw
    ]

    # Step 1: Deterministic matrix sanity checks
    matrices, channel_names = load_matrices_from_csv_paths(metric_csv_paths)
    per_metric_sanity = evaluate_all_matrices_sanity(matrices)

    # Step 2: Deterministic artifact detection heuristics
    plan = state.get("plan")
    if not channel_names and plan is not None:
        channel_names = list(getattr(plan, "channels", []) or [])

    thresholds = _extract_evaluator_thresholds(manifest)
    artifacts = run_artifact_heuristics(
        preprocessed_data_path=state.get("preprocessed_data_path"),
        raw_data_path=state.get("raw_data_path"),
        plv_matrix=matrices.get("plv"),
        channel_names=channel_names,
        ocular_variance_ratio=thresholds["ocular_variance_ratio"],
        muscle_power_threshold=thresholds["muscle_power_threshold"],
        electrode_pop_sigma=thresholds["electrode_pop_sigma"],
        tau_zerolag=thresholds["tau_zerolag"],
    )

    # Step 3: Top-5 unresolved zero-lag pairs VLM topology read
    unresolved = [
        p for p in evidence_list
        if p.classification == EvidenceCategoryEnum.ZERO_LAG_UNRESOLVED
    ]
    unresolved_sorted = sorted(unresolved, key=lambda p: p.zero_lag_mean, reverse=True)
    top_5_unresolved = unresolved_sorted[:5]

    user_prompt = build_evaluator_user_prompt(top_5_unresolved, artifacts, per_metric_sanity)
    image_paths = list(heatmap_paths.values()) + list(network_paths.values())
    _ = build_vlm_messages(EVALUATOR_SYSTEM_PROMPT, user_prompt, image_paths)

    if top_5_unresolved and llm is not None:
        try:
            resp = llm.invoke(
                [
                    {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            raw_text = getattr(resp, "content", str(resp))
            top_reads, visual_obs, recs = parse_vlm_topology_response(
                str(raw_text), top_5_unresolved, artifacts
            )
        except Exception as exc:
            logger.warning("Evaluator VLM call failed (%s); using deterministic topology read.", exc)
            top_reads, visual_obs, recs = fallback_topology_reads(top_5_unresolved, artifacts)
    else:
        top_reads, visual_obs, recs = fallback_topology_reads(top_5_unresolved, artifacts)

    artifacts.top_unresolved_visual_reads = top_reads

    # Merge Top-5 reads back into full evidence_summary without altering classifications
    read_lookup = {
        (p.channel_pair[0], p.channel_pair[1]): p for p in top_reads
    }
    updated_evidence: List[PairEvidenceSynthesis] = []
    for item in evidence_list:
        key = (item.channel_pair[0], item.channel_pair[1])
        updated_evidence.append(read_lookup.get(key, item))

    overall_verdict = _determine_overall_verdict(
        per_metric_sanity=per_metric_sanity,
        ocular_detected=artifacts.ocular_contamination.detected,
        muscle_detected=artifacts.muscle_contamination.detected,
        pop_detected=artifacts.electrode_pop.detected,
        n_unresolved_pairs=len(unresolved),
    )

    verdict = EvaluationVerdict(
        overall_verdict=overall_verdict,
        per_metric_sanity=per_metric_sanity,
        artifacts=artifacts,
        visual_observations=visual_obs,
        recommendations_for_user=recs,
    )

    # Step 4: CSV serialization
    csv_path = write_evaluation_summary_csv(verdict, run_id=run_id, output_dir=output_dir)

    return {
        "evaluation_verdict": verdict,
        "evaluation_summary_csv_path": csv_path,
        "evidence_summary": updated_evidence,
    }
