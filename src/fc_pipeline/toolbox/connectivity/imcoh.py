"""Absolute Imaginary Coherence (|ImCoh|) computation tool."""

from __future__ import annotations

from typing import Tuple
import mne
import numpy as np

from fc_pipeline.deterministic.connectivity.metrics import compute_spectral_connectivity


def compute_imcoh(
    epochs: mne.Epochs,
    band: Tuple[float, float],
    sfreq: float,
) -> np.ndarray:
    """Computes symmetric, [0, 1] bounded absolute Imaginary Coherence (|ImCoh|) matrix.

    Guarantees absolute value magnitude transformation across all electrode pairs.
    """
    fmin, fmax = band
    matrices = compute_spectral_connectivity(
        epochs=epochs,
        methods=["imcoh"],
        fmin=fmin,
        fmax=fmax,
        sfreq=sfreq,
    )
    return matrices["imcoh"]
