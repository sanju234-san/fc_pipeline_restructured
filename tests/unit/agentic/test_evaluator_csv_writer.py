"""Tests verifying EvaluationVerdict serialization to CSV matching schema."""

import csv
from pathlib import Path

from fc_pipeline.agentic.evaluator.csv_writer import (
    EVALUATION_CSV_COLUMNS,
    write_evaluation_summary_csv,
)
from fc_pipeline.schemas.enums import (
    EvidenceCategoryEnum,
    OverallVerdictEnum,
    VisualTopologyReadEnum,
)
from fc_pipeline.schemas.evaluation import (
    ArtifactDetectionResult,
    ArtifactSignatures,
    EvaluationVerdict,
    SanityCheckFlags,
)
from fc_pipeline.schemas.evidence import PairEvidenceSynthesis


def test_write_evaluation_summary_csv_columns_and_rows(tmp_path: Path):
    verdict = EvaluationVerdict(
        overall_verdict=OverallVerdictEnum.WARNING,
        per_metric_sanity={
            "pli": SanityCheckFlags(
                values_in_bounds=True,
                matrix_is_symmetric=True,
                is_sparse_or_flat=False,
                confidence=1.0,
            )
        },
        artifacts=ArtifactSignatures(
            ocular_contamination=ArtifactDetectionResult(
                detected=True,
                affected_channels=["Fp1", "Fp2"],
                confidence=0.85,
                detail="Ratio 2.1 vs 1.5",
            ),
            muscle_contamination=ArtifactDetectionResult(
                detected=False,
                affected_channels=[],
                confidence=0.90,
                detail="Clear",
            ),
            electrode_pop=ArtifactDetectionResult(
                detected=False,
                affected_channels=[],
                confidence=0.95,
                detail="Clear",
            ),
            top_unresolved_visual_reads=[
                PairEvidenceSynthesis(
                    channel_pair=["Fp1", "Fp2"],
                    phase_robust_mean=0.05,
                    zero_lag_mean=0.78,
                    classification=EvidenceCategoryEnum.ZERO_LAG_UNRESOLVED,
                    evaluator_visual_read=VisualTopologyReadEnum.CONSISTENT_WITH_ARTIFACT,
                    evaluator_visual_note="Fronto-polar ocular proximity.",
                )
            ],
        ),
        visual_observations="Fronto-polar zero-lag elevation observed.",
        recommendations_for_user=["Inspect Fp1-Fp2 ocular leakage."],
    )

    out_csv = write_evaluation_summary_csv(verdict, run_id="test_run_01", output_dir=tmp_path)
    assert Path(out_csv).is_file()

    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == EVALUATION_CSV_COLUMNS
        rows = list(reader)

    # 3 sanity rows + 3 artifact rows + 1 topology_read row = 7 rows
    assert len(rows) == 7
    categories = {r["check_category"] for r in rows}
    assert categories == {"sanity", "artifact", "topology_read"}
    topo_row = [r for r in rows if r["check_category"] == "topology_read"][0]
    assert topo_row["check_name"] == "Fp1-Fp2_topology"
    assert topo_row["result"] == "consistent_with_artifact"
    assert topo_row["affected_channels"] == "Fp1,Fp2"
