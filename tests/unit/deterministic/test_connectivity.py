"""Tests verifying 5-metric computation, matrix symmetry, bounds, and plotting."""

from pathlib import Path
import tempfile
import mne
import numpy as np
import pytest

from fc_pipeline.deterministic.connectivity.metrics import (
    ConnectivityComputationError,
    compute_metric_numerical_summaries,
    compute_spectral_connectivity,
)
from fc_pipeline.deterministic.connectivity.plotting import (
    render_connectivity_heatmap,
    render_network_graph,
)
from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.toolbox.connectivity import (
    compute_coherence,
    compute_imcoh,
    compute_pli,
    compute_plv,
    compute_wpli,
)


@pytest.fixture
def synthetic_epochs():
    """Generates synthetic 4-channel, 5-epoch EEG data at 250 Hz."""
    np.random.seed(42)
    sfreq = 250.0
    n_epochs = 5
    n_channels = 4
    n_times = 250  # 1 second epochs
    ch_names = ["F3", "F4", "C3", "C4"]

    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-5
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.EpochsArray(data, info, verbose=False)


def test_single_pass_5_metrics(synthetic_epochs):
    """Verifies that all five foundational metrics compute in a single pass."""
    methods = [
        MetricEnum.PLI,
        MetricEnum.WPLI,
        MetricEnum.IMAGINARY_COHERENCE,
        MetricEnum.PLV,
        MetricEnum.COHERENCE,
    ]
    matrices = compute_spectral_connectivity(
        epochs=synthetic_epochs,
        methods=methods,
        fmin=8.0,
        fmax=12.0,
    )

    assert len(matrices) == 5
    assert set(matrices.keys()) == {"pli", "wpli", "imcoh", "plv", "coh"}

    for metric_name, mat in matrices.items():
        assert mat.shape == (4, 4)
        # Symmetry check
        assert np.allclose(mat, mat.T, atol=1e-7), f"{metric_name} is not symmetric"
        # Zero diagonal check
        assert np.allclose(np.diag(mat), 0.0), f"{metric_name} diagonal is not 0"
        # Bound check [0, 1]
        assert np.all(mat >= 0.0) and np.all(mat <= 1.0), f"{metric_name} out of bounds [0, 1]"
        # Non-negative ImCoh check
        if metric_name == "imcoh":
            assert np.min(mat) >= 0.0


def test_individual_toolbox_tools(synthetic_epochs):
    """Verifies each leaf tool in toolbox/connectivity."""
    band = (8.0, 12.0)
    sfreq = synthetic_epochs.info["sfreq"]

    mat_pli = compute_pli(synthetic_epochs, band, sfreq)
    assert mat_pli.shape == (4, 4)
    assert np.allclose(mat_pli, mat_pli.T)

    mat_wpli = compute_wpli(synthetic_epochs, band, sfreq)
    assert mat_wpli.shape == (4, 4)
    assert np.allclose(mat_wpli, mat_wpli.T)

    mat_imcoh = compute_imcoh(synthetic_epochs, band, sfreq)
    assert mat_imcoh.shape == (4, 4)
    assert np.all(mat_imcoh >= 0.0)
    assert np.allclose(mat_imcoh, mat_imcoh.T)

    mat_plv = compute_plv(synthetic_epochs, band, sfreq)
    assert mat_plv.shape == (4, 4)
    assert np.allclose(mat_plv, mat_plv.T)

    mat_coh = compute_coherence(synthetic_epochs, band, sfreq)
    assert mat_coh.shape == (4, 4)
    assert np.allclose(mat_coh, mat_coh.T)


def test_numerical_summaries(synthetic_epochs):
    """Verifies that numerical summary statistics are computed over off-diagonal elements."""
    matrices = compute_spectral_connectivity(
        epochs=synthetic_epochs,
        methods=["pli", "plv"],
        fmin=8.0,
        fmax=12.0,
    )
    summaries = compute_metric_numerical_summaries(matrices)

    assert "pli" in summaries
    assert "plv" in summaries

    for metric_name, s in summaries.items():
        assert "mean" in s
        assert "std" in s
        assert "median" in s
        assert "min" in s
        assert "max" in s
        assert "sparsity" in s
        assert "top_5_pairs_mean" in s
        assert "bottom_5_pairs_mean" in s
        assert 0.0 <= s["mean"] <= 1.0
        assert 0.0 <= s["sparsity"] <= 1.0


def test_connectivity_error_insufficient_channels():
    """Verifies error raised when fewer than 2 channels are provided."""
    info = mne.create_info(["Cz"], 250.0, "eeg")
    data = np.random.randn(2, 1, 250)
    epochs = mne.EpochsArray(data, info, verbose=False)

    with pytest.raises(ConnectivityComputationError, match="at least 2 channels"):
        compute_spectral_connectivity(epochs)


def test_plotting(synthetic_epochs):
    """Verifies heatmap and network graph PNG generation."""
    matrices = compute_spectral_connectivity(
        epochs=synthetic_epochs,
        methods=["pli"],
        fmin=8.0,
        fmax=12.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        heatmap_path = tmp_path / "heatmap_pli.png"
        network_path = tmp_path / "network_pli.png"

        h_out = render_connectivity_heatmap(
            matrix=matrices["pli"],
            channel_names=synthetic_epochs.ch_names,
            metric_name="pli",
            output_path=heatmap_path,
        )
        assert Path(h_out).exists()
        assert Path(h_out).stat().st_size > 0

        n_out = render_network_graph(
            matrix=matrices["pli"],
            channel_names=synthetic_epochs.ch_names,
            metric_name="pli",
            output_path=network_path,
        )
        assert Path(n_out).exists()
        assert Path(n_out).stat().st_size > 0
