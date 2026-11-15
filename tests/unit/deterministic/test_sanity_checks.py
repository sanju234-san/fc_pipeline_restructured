"""Tests verifying matrix [0, 1] range, symmetry, and flatline SNR assertions."""

import numpy as np
import pandas as pd

from fc_pipeline.deterministic.sanity_checks import (
    evaluate_all_matrices_sanity,
    evaluate_matrix_sanity,
    load_matrices_from_csv_paths,
)


def test_healthy_symmetric_matrix_passes():
    mat = np.array([
        [0.0, 0.25, 0.60],
        [0.25, 0.0, 0.42],
        [0.60, 0.42, 0.0],
    ])
    flags = evaluate_matrix_sanity(mat, metric_name="pli")
    assert flags.values_in_bounds is True
    assert flags.matrix_is_symmetric is True
    assert flags.is_sparse_or_flat is False
    assert flags.confidence == 1.0


def test_out_of_bounds_or_asymmetric_fails():
    asym = np.array([
        [0.0, 0.8, 0.2],
        [0.1, 0.0, 0.5],
        [0.2, 0.5, 0.0],
    ])
    flags = evaluate_matrix_sanity(asym)
    assert flags.matrix_is_symmetric is False
    assert flags.confidence == 0.0

    oob = np.array([
        [0.0, 1.2],
        [1.2, 0.0],
    ])
    flags_oob = evaluate_matrix_sanity(oob)
    assert flags_oob.values_in_bounds is False
    assert flags_oob.confidence == 0.0


def test_flatline_matrix_flagged():
    flat = np.zeros((3, 3))
    flags = evaluate_matrix_sanity(flat)
    assert flags.is_sparse_or_flat is True
    assert flags.confidence == 0.0


def test_load_matrices_from_csv_paths(tmp_path):
    csv_path = tmp_path / "pli_run1.csv"
    df = pd.DataFrame(
        [[0.0, 0.3, 0.5], [0.3, 0.0, 0.1], [0.5, 0.1, 0.0]],
        index=["C3", "C4", "Cz"],
        columns=["C3", "C4", "Cz"],
    )
    df.to_csv(csv_path)
    mats, chs = load_matrices_from_csv_paths({"pli": str(csv_path), "evidence_summary": "ignored.csv"})
    assert "pli" in mats
    assert chs == ["C3", "C4", "Cz"]
    all_flags = evaluate_all_matrices_sanity(mats)
    assert all_flags["pli"].confidence == 1.0
