"""Unit interval [0,1] bounds, symmetry (M = M^T), and flatline SNR assertions.

Follows Section 6.4 Step 1 of PIPELINE_PROTOTYPE_DESIGN.md.
Strictly deterministic: zero LLM / VLM / MLflow imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fc_pipeline.schemas.evaluation import SanityCheckFlags

_SKIP_CSV_KEYS = {"evidence_summary", "consolidated_report"}


def evaluate_matrix_sanity(
    matrix: np.ndarray,
    metric_name: str = "",
    tol: float = 1e-6,
    flat_std_threshold: float = 1e-4,
) -> SanityCheckFlags:
    """Evaluates mathematical bounds [0, 1], symmetry (M = M^T), and SNR flatness for a single matrix.

    Per Section 6.4 Step 1, deterministic math checks produce confidence 1.0 when
    all assertions pass and 0.0 when any assertion fails.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        return SanityCheckFlags(
            values_in_bounds=False,
            matrix_is_symmetric=False,
            is_sparse_or_flat=True,
            confidence=0.0,
        )

    finite = bool(np.all(np.isfinite(arr)))
    in_bounds = bool(finite and np.all(arr >= -tol) and np.all(arr <= 1.0 + tol))
    is_symmetric = bool(finite and np.allclose(arr, arr.T, atol=tol, equal_nan=False))

    n = arr.shape[0]
    if n > 1:
        iu = np.triu_indices(n, k=1)
        off_diag = arr[iu]
    else:
        off_diag = arr.ravel()

    if not finite or off_diag.size == 0:
        is_sparse_or_flat = True
    else:
        all_near_zero = bool(np.all(np.abs(off_diag) < 1e-3))
        all_near_one = bool(np.all(off_diag > 1.0 - 1e-3))
        low_spread = bool(off_diag.size > 1 and float(np.std(off_diag)) < flat_std_threshold)
        is_sparse_or_flat = bool(all_near_zero or all_near_one or low_spread)

    all_passed = in_bounds and is_symmetric and (not is_sparse_or_flat)
    return SanityCheckFlags(
        values_in_bounds=in_bounds,
        matrix_is_symmetric=is_symmetric,
        is_sparse_or_flat=is_sparse_or_flat,
        confidence=1.0 if all_passed else 0.0,
    )


def evaluate_all_matrices_sanity(
    matrices: Dict[str, np.ndarray],
) -> Dict[str, SanityCheckFlags]:
    """Runs deterministic sanity checks across all computed connectivity matrices."""
    return {
        metric: evaluate_matrix_sanity(mat, metric_name=metric)
        for metric, mat in matrices.items()
    }


def load_matrices_from_csv_paths(
    metric_csv_paths: Dict[str, str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Loads N x N connectivity matrices and channel names from exported CSV paths."""
    matrices: Dict[str, np.ndarray] = {}
    channel_names: List[str] = []
    for metric, raw_path in (metric_csv_paths or {}).items():
        if metric in _SKIP_CSV_KEYS or not raw_path:
            continue
        p = Path(str(raw_path))
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p, index_col=0)
            if df.shape[0] == df.shape[1] and df.shape[0] > 0:
                matrices[metric] = df.to_numpy(dtype=float)
                if not channel_names:
                    channel_names = [str(c) for c in df.columns]
        except Exception:
            continue
    return matrices, channel_names
