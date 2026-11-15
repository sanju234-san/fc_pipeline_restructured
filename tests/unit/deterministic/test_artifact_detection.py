"""Tests for deterministic ocular, muscle, and pop threshold heuristics."""

import numpy as np

from fc_pipeline.deterministic.artifact_detection import (
    detect_electrode_pop,
    detect_muscle_contamination,
    detect_ocular_contamination,
)


def test_detect_ocular_contamination_fires_and_clears():
    res_detected = detect_ocular_contamination(
        {"Fp1": 4.0e-10, "Fp2": 3.8e-10, "O1": 1.0e-10, "O2": 1.0e-10},
        ocular_variance_ratio=1.5,
    )
    assert res_detected.detected is True
    assert set(res_detected.affected_channels) == {"Fp1", "Fp2"}
    assert 0.5 <= res_detected.confidence <= 1.0

    res_clear = detect_ocular_contamination(
        {"Fp1": 1.1e-10, "Fp2": 1.0e-10, "O1": 1.0e-10, "O2": 1.0e-10},
        ocular_variance_ratio=1.5,
    )
    assert res_clear.detected is False
    assert res_clear.affected_channels == []


def test_detect_muscle_contamination_fires_and_clears():
    res = detect_muscle_contamination(
        high_freq_power={"T7": 5.0, "T8": 1.0, "C3": 0.5},
        full_band_power={"T7": 1.0, "T8": 1.0, "C3": 1.0},
        muscle_power_threshold=2.0,
    )
    assert res.detected is True
    assert res.affected_channels == ["T7"]


def test_detect_electrode_pop_fires_on_outlier():
    variances = {f"Ch{i}": 1.0e-11 for i in range(15)}
    variances["Ch0"] = 1.0e-9  # huge pop outlier
    res = detect_electrode_pop(variances, electrode_pop_sigma=3.0)
    assert res.detected is True
    assert res.affected_channels == ["Ch0"]
    assert 0.5 < res.confidence <= 1.0
