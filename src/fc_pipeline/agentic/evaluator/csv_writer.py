"""Flattens EvaluationVerdict into evaluation_summary_<run_id>.csv per Section 6.4 Step 4; pure serialization, no new computation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from fc_pipeline.config.constants import EVALUATION_CSV_FILENAME_TEMPLATE, OUTPUT_DIR
from fc_pipeline.schemas.evaluation import EvaluationVerdict

EVALUATION_CSV_COLUMNS = [
    "run_id",
    "check_category",
    "check_name",
    "result",
    "confidence",
    "affected_channels",
    "detail",
]


def write_evaluation_summary_csv(
    verdict: EvaluationVerdict,
    run_id: str,
    output_dir: Optional[Path] = None,
) -> str:
    """Serializes an EvaluationVerdict into evaluation_summary_<run_id>.csv."""
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / EVALUATION_CSV_FILENAME_TEMPLATE.format(run_id=run_id)

    rows = []

    # 1. Sanity checks per metric
    for metric, flags in (verdict.per_metric_sanity or {}).items():
        rows.append({
            "run_id": run_id,
            "check_category": "sanity",
            "check_name": f"{metric}_values_in_bounds",
            "result": "pass" if flags.values_in_bounds else "fail",
            "confidence": f"{flags.confidence:.4f}",
            "affected_channels": "",
            "detail": f"All {metric} matrix values within unit interval [0, 1]: {flags.values_in_bounds}",
        })
        rows.append({
            "run_id": run_id,
            "check_category": "sanity",
            "check_name": f"{metric}_matrix_is_symmetric",
            "result": "pass" if flags.matrix_is_symmetric else "fail",
            "confidence": f"{flags.confidence:.4f}",
            "affected_channels": "",
            "detail": f"Matrix symmetry (M = M^T) verified for {metric}: {flags.matrix_is_symmetric}",
        })
        rows.append({
            "run_id": run_id,
            "check_category": "sanity",
            "check_name": f"{metric}_is_sparse_or_flat",
            "result": "fail" if flags.is_sparse_or_flat else "pass",
            "confidence": f"{flags.confidence:.4f}",
            "affected_channels": "",
            "detail": f"SNR flatness / collapse flag for {metric}: {flags.is_sparse_or_flat}",
        })

    # 2. Artifact heuristic checks
    artifact_entries = [
        ("ocular_contamination", verdict.artifacts.ocular_contamination),
        ("muscle_contamination", verdict.artifacts.muscle_contamination),
        ("electrode_pop", verdict.artifacts.electrode_pop),
    ]
    for art_name, art_res in artifact_entries:
        rows.append({
            "run_id": run_id,
            "check_category": "artifact",
            "check_name": art_name,
            "result": "detected" if art_res.detected else "clear",
            "confidence": f"{art_res.confidence:.4f}",
            "affected_channels": ",".join(art_res.affected_channels or []),
            "detail": art_res.detail,
        })

    # 3. Top-5 VLM visual topology reads
    for pair_ev in verdict.artifacts.top_unresolved_visual_reads or []:
        pair_label = "-".join(pair_ev.channel_pair)
        read_val = (
            pair_ev.evaluator_visual_read.value
            if pair_ev.evaluator_visual_read is not None
            else "indeterminate"
        )
        rows.append({
            "run_id": run_id,
            "check_category": "topology_read",
            "check_name": f"{pair_label}_topology",
            "result": read_val,
            "confidence": "0.8000",
            "affected_channels": ",".join(pair_ev.channel_pair),
            "detail": pair_ev.evaluator_visual_note or "",
        })

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVALUATION_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return str(file_path.resolve())
