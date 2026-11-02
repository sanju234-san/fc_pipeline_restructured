"""Tests for 4-category threshold comparison logic and CSV outputs."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from fc_pipeline.deterministic.connectivity.classification import (
    classify_channel_pairs,
    export_connectivity_matrices_to_csv,
    export_evidence_summary_to_csv,
)
from fc_pipeline.schemas.enums import EvidenceCategoryEnum


def test_4_category_classification_logic():
    """Verifies all four classification outcomes using controlled artificial matrices."""
    # 4 channels: A, B, C, D
    # Pairs:
    # (A, B): Phase = 0.40 (>= 0.20), Zerolag = 0.50 (>= 0.35) -> FULLY CONVERGENT
    # (A, C): Phase = 0.30 (>= 0.20), Zerolag = 0.10 (< 0.35)  -> CONFIRMED COUPLING
    # (A, D): Phase = 0.10 (< 0.20),  Zerolag = 0.60 (>= 0.35) -> ZERO-LAG COUPLING, CAUSE UNRESOLVED
    # (B, C): Phase = 0.05 (< 0.20),  Zerolag = 0.15 (< 0.35)  -> NO COUPLING
    ch_names = ["A", "B", "C", "D"]
    n = 4

    # Setup Phase metrics: pli and wpli
    pli_mat = np.zeros((n, n))
    pli_mat[0, 1] = pli_mat[1, 0] = 0.40
    pli_mat[0, 2] = pli_mat[2, 0] = 0.30
    pli_mat[0, 3] = pli_mat[3, 0] = 0.10
    pli_mat[1, 2] = pli_mat[2, 1] = 0.05

    wpli_mat = np.copy(pli_mat)  # Identical so phase_robust_mean equals pli_mat

    # Setup Zerolag metrics: plv and coh
    plv_mat = np.zeros((n, n))
    plv_mat[0, 1] = plv_mat[1, 0] = 0.50
    plv_mat[0, 2] = plv_mat[2, 0] = 0.10
    plv_mat[0, 3] = plv_mat[3, 0] = 0.60
    plv_mat[1, 2] = plv_mat[2, 1] = 0.15

    coh_mat = np.copy(plv_mat)

    matrices = {
        "pli": pli_mat,
        "wpli": wpli_mat,
        "plv": plv_mat,
        "coh": coh_mat,
    }

    evidence = classify_channel_pairs(
        matrices=matrices,
        channel_names=ch_names,
        tau_phase=0.20,
        tau_zerolag=0.35,
    )

    # 4 channels -> 6 unique pairs
    assert len(evidence) == 6

    # Convert to lookup by tuple
    lookup = {tuple(item.channel_pair): item for item in evidence}

    # (A, B) -> FULLY CONVERGENT
    ab = lookup[("A", "B")]
    assert ab.phase_robust_mean == 0.40
    assert ab.zero_lag_mean == 0.50
    assert ab.classification == EvidenceCategoryEnum.FULLY_CONVERGENT

    # (A, C) -> CONFIRMED COUPLING
    ac = lookup[("A", "C")]
    assert ac.phase_robust_mean == 0.30
    assert ac.zero_lag_mean == 0.10
    assert ac.classification == EvidenceCategoryEnum.CONFIRMED_COUPLING

    # (A, D) -> ZERO-LAG COUPLING, CAUSE UNRESOLVED
    ad = lookup[("A", "D")]
    assert ad.phase_robust_mean == 0.10
    assert ad.zero_lag_mean == 0.60
    assert ad.classification == EvidenceCategoryEnum.ZERO_LAG_UNRESOLVED

    # (B, C) -> NO COUPLING
    bc = lookup[("B", "C")]
    assert bc.phase_robust_mean == 0.05
    assert bc.zero_lag_mean == 0.15
    assert bc.classification == EvidenceCategoryEnum.NO_COUPLING


def test_csv_serialization():
    """Verifies CSV generation for per-metric matrices and evidence summary."""
    ch_names = ["F3", "F4"]
    mat = np.array([[0.0, 0.45], [0.45, 0.0]])
    matrices = {"pli": mat, "plv": mat}

    evidence = classify_channel_pairs(matrices, ch_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        run_id = "test_run_123"

        # Export matrices
        csv_paths = export_connectivity_matrices_to_csv(matrices, ch_names, tmp_path, run_id)
        assert "pli" in csv_paths
        assert "plv" in csv_paths

        df_pli = pd.read_csv(csv_paths["pli"], index_col=0)
        assert df_pli.shape == (2, 2)
        assert df_pli.loc["F3", "F4"] == 0.45

        # Export evidence summary
        summary_path = export_evidence_summary_to_csv(evidence, matrices, ch_names, tmp_path, run_id)
        assert Path(summary_path).exists()

        df_summary = pd.read_csv(summary_path)
        assert len(df_summary) == 1
        assert "channel_1" in df_summary.columns
        assert "channel_2" in df_summary.columns
        assert "phase_robust_mean" in df_summary.columns
        assert "zero_lag_mean" in df_summary.columns
        assert "classification" in df_summary.columns
        assert "pli" in df_summary.columns
        assert "plv" in df_summary.columns


def test_consolidated_report_csv_schema():
    """Verifies consolidated CSV has all 5 metric columns and correct row count."""
    from fc_pipeline.deterministic.connectivity.classification import export_consolidated_connectivity_report_csv

    ch_names = ["F3", "F4", "C3"]
    n = 3
    mat = np.array([
        [0.0, 0.5, 0.3],
        [0.5, 0.0, 0.2],
        [0.3, 0.2, 0.0],
    ])
    matrices = {"pli": mat, "wpli": mat, "imcoh": mat, "plv": mat, "coh": mat}
    evidence = classify_channel_pairs(matrices, ch_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        run_id = "test_consolidated_001"
        report_path = export_consolidated_connectivity_report_csv(
            evidence_list=evidence,
            matrices=matrices,
            channel_names=ch_names,
            output_dir=tmp_path,
            run_id=run_id,
            condition="resting_state",
            freq_band="alpha",
        )
        assert Path(report_path).exists(), "Consolidated report path must exist"

        df = pd.read_csv(report_path)
        # 3 channels → 3 unique pairs (C(3,2) = 3)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

        required_cols = ["channel_1", "channel_2", "PLI", "wPLI", "ImCoh", "PLV", "Coherence",
                         "phase_robust_mean", "zero_lag_mean", "classification", "run_id", "condition", "freq_band"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Verify metadata columns
        assert (df["run_id"] == run_id).all(), "run_id must be populated in all rows"
        assert (df["condition"] == "resting_state").all()
        assert (df["freq_band"] == "alpha").all()

        # Verify metric values are floats in [0, 1]
        for metric_col in ["PLI", "wPLI", "ImCoh", "PLV", "Coherence"]:
            assert df[metric_col].between(0.0, 1.0).all(), f"{metric_col} values must be in [0, 1]"


def test_consolidated_report_csv_values_match_matrices():
    """Verifies consolidated CSV values match the source matrices exactly."""
    from fc_pipeline.deterministic.connectivity.classification import export_consolidated_connectivity_report_csv

    ch_names = ["A", "B"]
    pli_val = 0.421
    wpli_val = 0.388
    imcoh_val = 0.155
    plv_val = 0.610
    coh_val = 0.723

    matrices = {
        "pli": np.array([[0.0, pli_val], [pli_val, 0.0]]),
        "wpli": np.array([[0.0, wpli_val], [wpli_val, 0.0]]),
        "imcoh": np.array([[0.0, imcoh_val], [imcoh_val, 0.0]]),
        "plv": np.array([[0.0, plv_val], [plv_val, 0.0]]),
        "coh": np.array([[0.0, coh_val], [coh_val, 0.0]]),
    }
    evidence = classify_channel_pairs(matrices, ch_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        report_path = export_consolidated_connectivity_report_csv(
            evidence_list=evidence,
            matrices=matrices,
            channel_names=ch_names,
            output_dir=tmp_path,
            run_id="test_values_002",
        )
        df = pd.read_csv(report_path)
        assert len(df) == 1  # Only (A, B) pair
        row = df.iloc[0]

        assert abs(row["PLI"] - pli_val) < 1e-4, f"PLI mismatch: {row['PLI']} != {pli_val}"
        assert abs(row["wPLI"] - wpli_val) < 1e-4
        assert abs(row["ImCoh"] - imcoh_val) < 1e-4
        assert abs(row["PLV"] - plv_val) < 1e-4
        assert abs(row["Coherence"] - coh_val) < 1e-4


def test_consolidated_report_generic_copy_created():
    """Verifies that a generic 'consolidated_connectivity_report.csv' alias is also written."""
    from fc_pipeline.deterministic.connectivity.classification import export_consolidated_connectivity_report_csv

    ch_names = ["X", "Y"]
    mat = np.array([[0.0, 0.25], [0.25, 0.0]])
    matrices = {"pli": mat, "wpli": mat, "imcoh": mat, "plv": mat, "coh": mat}
    evidence = classify_channel_pairs(matrices, ch_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        export_consolidated_connectivity_report_csv(
            evidence_list=evidence,
            matrices=matrices,
            channel_names=ch_names,
            output_dir=tmp_path,
            run_id="test_copy_003",
        )
        # The generic alias must also exist
        generic = tmp_path / "consolidated_connectivity_report.csv"
        assert generic.exists(), "Generic alias 'consolidated_connectivity_report.csv' must be created"
