"""Single-pass 5-metric spectral estimation (PLI, wPLI, ImCoh, PLV, Coherence).

Follows Section 4.2, 4.3, and 6.3 of PIPELINE_PROTOTYPE_DESIGN.md.
Strictly deterministic, zero LLM / VLM / MLflow imports.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mne
import numpy as np

try:
    import mne_connectivity
except ImportError:
    mne_connectivity = None

from fc_pipeline.schemas.enums import MetricEnum

logger = logging.getLogger(__name__)

# Canonical metric string list
ALL_5_METRICS = [
    MetricEnum.PLI.value,
    MetricEnum.WPLI.value,
    MetricEnum.IMAGINARY_COHERENCE.value,
    MetricEnum.PLV.value,
    MetricEnum.COHERENCE.value,
]


class ConnectivityComputationError(Exception):
    """Raised when connectivity matrix computation fails or produces invalid results."""


def _symmetrize_and_clean_matrix(
    raw_dense_matrix: np.ndarray,
    is_imcoh: bool = False,
) -> np.ndarray:
    """Takes a dense (N, N) lower/upper triangular matrix from MNE-connectivity,

    applies absolute value if ImCoh, symmetrizes, and zeros out the diagonal.
    """
    mat = np.copy(raw_dense_matrix)
    if is_imcoh:
        mat = np.abs(mat)

    # Symmetrize: MNE spectral_connectivity_epochs with faverage=True populates
    # the lower-triangular portion when indices=None.
    # To guarantee strict symmetry: M = triu(M + M^T) or M + M.T (since diagonal is 0)
    # If both upper and lower had values, we take (mat + mat.T) / (1 + (mat!=0 & mat.T!=0))
    # In MNE indices=None, only one triangle is computed.
    # Check if upper triangle is all zero:
    if np.allclose(np.triu(mat, k=1), 0.0):
        sym = mat + mat.T
    elif np.allclose(np.tril(mat, k=-1), 0.0):
        sym = mat + mat.T
    else:
        # Both triangles have values, average them
        sym = 0.5 * (mat + mat.T)

    # Zero diagonal
    np.fill_diagonal(sym, 0.0)

    # Clean any tiny numerical noise outside [0, 1]
    sym = np.clip(sym, 0.0, 1.0)
    return sym


def compute_spectral_connectivity(
    epochs: mne.Epochs,
    methods: Optional[Sequence[Union[str, MetricEnum]]] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    sfreq: Optional[float] = None,
    mode: str = "multitaper",
) -> Dict[str, np.ndarray]:
    """Computes spectral functional connectivity matrices in a single pass.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data containing at least 2 channels and 1 epoch.
    methods : Sequence[str | MetricEnum], optional
        List of connectivity metrics to compute. Defaults to all 5 foundational metrics.
    fmin : float, optional
        Lower frequency cutoff in Hz. Defaults to 1.0 Hz if not provided.
    fmax : float, optional
        Upper frequency cutoff in Hz. Defaults to Nyquist if not provided.
    sfreq : float, optional
        Sampling frequency in Hz. Extracted from epochs.info['sfreq'] if None.
    mode : str
        Spectrum estimation mode ('multitaper' or 'fourier'). Default 'multitaper'.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping canonical metric string ('pli', 'wpli', 'imcoh', 'plv', 'coh')
        to symmetric (N, N) NumPy ndarrays bounded in [0, 1] with zero diagonal.
    """
    if mne_connectivity is None:
        raise ConnectivityComputationError(
            "mne-connectivity is not installed. Please run `pip install mne-connectivity`."
        )

    if len(epochs.ch_names) < 2:
        raise ConnectivityComputationError(
            f"Connectivity analysis requires at least 2 channels, but got {len(epochs.ch_names)}."
        )

    if len(epochs) < 1:
        raise ConnectivityComputationError("Epochs object contains zero epochs.")

    effective_sfreq = float(sfreq or epochs.info["sfreq"])
    effective_fmin = float(fmin if fmin is not None else 1.0)
    effective_fmax = float(fmax if fmax is not None else (effective_sfreq / 2.0 - 1.0))

    if effective_fmin >= effective_fmax:
        raise ConnectivityComputationError(
            f"Invalid frequency bounds: fmin ({effective_fmin}) must be strictly less than fmax ({effective_fmax})."
        )

    # Normalize method strings
    if methods is None or len(methods) == 0:
        requested_methods = list(ALL_5_METRICS)
    else:
        requested_methods = []
        for m in methods:
            val = m.value if isinstance(m, MetricEnum) else str(m).lower()
            if val == "coherence":
                val = "coh"
            elif val in ("imaginary_coherence", "icoh"):
                val = "imcoh"
            if val not in requested_methods:
                requested_methods.append(val)

    logger.info(
        "Computing spectral connectivity: methods=%s, fmin=%.1f, fmax=%.1f, n_channels=%d, n_epochs=%d",
        requested_methods,
        effective_fmin,
        effective_fmax,
        len(epochs.ch_names),
        len(epochs),
    )

    try:
        con_results = mne_connectivity.spectral_connectivity_epochs(
            epochs,
            method=requested_methods,
            indices=None,
            mode=mode,
            sfreq=effective_sfreq,
            fmin=effective_fmin,
            fmax=effective_fmax,
            faverage=True,
            verbose=False,
        )
    except Exception as exc:
        raise ConnectivityComputationError(
            f"mne_connectivity.spectral_connectivity_epochs failed: {type(exc).__name__}: {exc}"
        ) from exc

    # If single method was passed, mne_connectivity may return a single object instead of a list
    if not isinstance(con_results, list):
        con_results = [con_results]

    matrices: Dict[str, np.ndarray] = {}
    for con in con_results:
        canonical_name = con.method.lower()
        if canonical_name == "imaginary coherence":
            canonical_name = "imcoh"
        elif canonical_name == "coherence":
            canonical_name = "coh"

        # get dense matrix shape: (N, N, 1) or (N, N)
        dense_data = con.get_data(output="dense")
        if dense_data.ndim == 3:
            matrix_2d = dense_data[:, :, 0]
        else:
            matrix_2d = dense_data

        is_imcoh = canonical_name == "imcoh"
        sym_matrix = _symmetrize_and_clean_matrix(matrix_2d, is_imcoh=is_imcoh)

        # Verification of mathematical invariants
        if not np.all(np.isfinite(sym_matrix)):
            raise ConnectivityComputationError(
                f"Non-finite values (NaN/Inf) encountered in computed {canonical_name} matrix."
            )

        if not np.allclose(sym_matrix, sym_matrix.T):
            raise ConnectivityComputationError(
                f"Symmetry check failed for computed {canonical_name} matrix."
            )

        matrices[canonical_name] = sym_matrix

    return matrices


def compute_metric_numerical_summaries(
    matrices: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Computes comprehensive numerical summary statistics for each connectivity metric.

    Returns dict mapping metric name -> summary dict with keys:
    mean, std, median, min, max, sparsity, top_5_pairs_mean, bottom_5_pairs_mean.
    All calculations are performed strictly over off-diagonal pairwise connections.
    """
    summaries: Dict[str, Dict[str, float]] = {}

    for metric_name, matrix in matrices.items():
        n = matrix.shape[0]
        # Extract upper triangle off-diagonal elements
        triu_indices = np.triu_indices(n, k=1)
        pairs_values = matrix[triu_indices]

        if len(pairs_values) == 0:
            summaries[metric_name] = {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sparsity": 1.0,
                "top_5_pairs_mean": 0.0,
                "bottom_5_pairs_mean": 0.0,
            }
            continue

        sorted_vals = np.sort(pairs_values)
        top_k = min(5, len(sorted_vals))
        bottom_k = min(5, len(sorted_vals))

        top_5_mean = float(np.mean(sorted_vals[-top_k:]))
        bottom_5_mean = float(np.mean(sorted_vals[:bottom_k]))

        # Sparsity: fraction of off-diagonal pairs with near-zero coupling (< 0.05)
        sparsity_val = float(np.mean(pairs_values < 0.05))

        summaries[metric_name] = {
            "mean": float(np.mean(pairs_values)),
            "std": float(np.std(pairs_values)),
            "median": float(np.median(pairs_values)),
            "min": float(np.min(pairs_values)),
            "max": float(np.max(pairs_values)),
            "sparsity": sparsity_val,
            "top_5_pairs_mean": top_5_mean,
            "bottom_5_pairs_mean": bottom_5_mean,
        }

    return summaries
