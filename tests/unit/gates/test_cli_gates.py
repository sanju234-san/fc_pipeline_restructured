"""Tests for Gate 1 and Gate 2 terminal prompt parsing and input validation."""

import pytest

from fc_pipeline.gates.cli.gate2_prepublication import (
    format_gate2_cli_review,
    run_gate2_cli_review,
    verify_report_prepublication,
)
from fc_pipeline.schemas.enums import OverallVerdictEnum
from fc_pipeline.schemas.evaluation import (
    ArtifactDetectionResult,
    ArtifactSignatures,
    EvaluationVerdict,
    SanityCheckFlags,
)


def _dummy_verdict() -> EvaluationVerdict:
    return EvaluationVerdict(
        overall_verdict=OverallVerdictEnum.PASS,
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
                detected=False, affected_channels=[], confidence=0.9, detail="Clear"
            ),
            muscle_contamination=ArtifactDetectionResult(
                detected=False, affected_channels=[], confidence=0.9, detail="Clear"
            ),
            electrode_pop=ArtifactDetectionResult(
                detected=False, affected_channels=[], confidence=0.9, detail="Clear"
            ),
            top_unresolved_visual_reads=[],
        ),
        visual_observations="All checks passed.",
        recommendations_for_user=["None"],
    )


def test_gate2_cli_formatting_and_input_validation():
    verdict = _dummy_verdict()
    text = format_gate2_cli_review(verdict, run_id="r1")
    assert "Subsection A" in text
    assert "Subsection B" in text

    assert run_gate2_cli_review(verdict, run_id="r1", input_fn=lambda _: "approve") is True
    assert run_gate2_cli_review(verdict, run_id="r1", input_fn=lambda _: "reject") is False

    with pytest.raises(ValueError, match="Available: approve, reject"):
        run_gate2_cli_review(verdict, run_id="r1", input_fn=lambda _: "maybe")


def test_verify_report_prepublication_detects_phi():
    res = verify_report_prepublication(
        "Patient Name: John Doe, tau_phase=0.20, tau_zerolag=0.35, ZERO-LAG COUPLING, CAUSE UNRESOLVED, Multiple Comparisons, Evaluator Quality Audit",
        {"numerical_summaries": {}},
    )
    assert res["passed"] is False
    assert res["checks"]["phi"] is False
