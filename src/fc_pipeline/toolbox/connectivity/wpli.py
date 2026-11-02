"""wPLI (Weighted Phase Lag Index) computation tool."""

from __future__ import annotations

from typing import Tuple
import mne
import numpy as np

from fc_pipeline.deterministic.connectivity.metrics import compute_spectral_connectivity


def compute_wpli(
    epochs: mne.Epochs,
    band: Tuple[float, float],
    sfreq: float,
) -> np.ndarray:
    """Computes symmetric, [0, 1] bounded Weighted Phase Lag Index (wPLI) matrix."""
    fmin, fmax = band
    matrices = compute_spectral_connectivity(
        epochs=epochs,
        methods=["wpli"],
        fmin=fmin,
        fmax=fmax,
        sfreq=sfreq,
    )
    return matrices["wpli"]
