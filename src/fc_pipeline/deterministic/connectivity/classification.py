"""Pairwise 4-category evidence classification and CSV serialization.

Follows Section 4.2 and Section 6.3 of PIPELINE_PROTOTYPE_DESIGN.md.
Strictly deterministic, zero LLM / VLM / MLflow imports.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fc_pipeline.config.thresholds import TAU_PHASE, TAU_ZEROLAG
from fc_pipeline.schemas.enums import EvidenceCategoryEnum
from fc_pipeline.schemas.evidence import PairEvidenceSynthesis

logger = logging.getLogger(__name__)

PHASE_ROBUST_METRICS = {"pli", "wpli", "imcoh"}
ZERO_LAG_METRICS = {"plv", "coh"}


def classify_channel_pairs(
    matrices: Dict[str, np.ndarray],
    channel_names: Sequence[str],
    tau_phase: float = TAU_PHASE,
    tau_zerolag: float = TAU_ZEROLAG,
) -> List[PairEvidenceSynthesis]:
    """Performs deterministic complementary evidence synthesis classification across all unique channel pairs.

    Parameters
    ----------
    matrices : Dict[str, np.ndarray]
        Dict of symmetric (N, N) connectivity matrices for computed metrics.
    channel_names : Sequence[str]
        List of channel label strings corresponding to the rows/columns of matrices.
    tau_phase : float
        Fixed heuristic threshold for the phase-robust group. Defaults to 0.20.
    tau_zerolag : float
        Fixed heuristic threshold for the zero-lag-inclusive group. Defaults to 0.35.

    Returns
    -------
    List[PairEvidenceSynthesis]
        Strongly typed list of pairwise evidence syntheses for all unique pairs (i < j).
    """
    n_channels = len(channel_names)
    available_metrics = set(matrices.keys())
    phase_metrics = [m for m in available_metrics if m in PHASE_ROBUST_METRICS]
    zerolag_metrics = [m for m in available_metrics if m in ZERO_LAG_METRICS]

    has_phase = len(phase_metrics) > 0
    has_zerolag = len(zerolag_metrics) > 0

    if not has_phase or not has_zerolag:
        logger.warning(
            "Evidence synthesis lacks representation from both groups: "
            "phase_metrics=%s, zerolag_metrics=%s. Cross-metric synthesis may be incomplete.",
            phase_metrics,
            zerolag_metrics,
        )

    evidence_list: List[PairEvidenceSynthesis] = []

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            ch_i = str(channel_names[i])
            ch_j = str(channel_names[j])

            # Phase-robust group mean
            if has_phase:
                phase_vals = [float(matrices[m][i, j]) for m in phase_metrics]
                phase_mean = float(np.mean(phase_vals))
            else:
                phase_mean = 0.0

            # Zero-lag-inclusive group mean
            if has_zerolag:
                zerolag_vals = [float(matrices[m][i, j]) for m in zerolag_metrics]
                zerolag_mean = float(np.mean(zerolag_vals))
            else:
                zerolag_mean = 0.0

            # 4-Category Decision Rule (Section 4.2 & 6.3)
            phase_strong = phase_mean >= tau_phase
            zerolag_strong = zerolag_mean >= tau_zerolag

            if phase_strong and zerolag_strong:
                category = EvidenceCategoryEnum.FULLY_CONVERGENT
            elif phase_strong and not zerolag_strong:
                category = EvidenceCategoryEnum.CONFIRMED_COUPLING
            elif not phase_strong and zerolag_strong:
                category = EvidenceCategoryEnum.ZERO_LAG_UNRESOLVED
            else:
                category = EvidenceCategoryEnum.NO_COUPLING

            synthesis = PairEvidenceSynthesis(
                channel_pair=[ch_i, ch_j],
                phase_robust_mean=round(phase_mean, 6),
                zero_lag_mean=round(zerolag_mean, 6),
                classification=category,
            )
            evidence_list.append(synthesis)

    return evidence_list


def export_connectivity_matrices_to_csv(
    matrices: Dict[str, np.ndarray],
    channel_names: Sequence[str],
    output_dir: Path,
    run_id: str,
) -> Dict[str, str]:
    """Saves each computed N x N symmetric connectivity matrix as a CSV file.

    Parameters
    ----------
    matrices : Dict[str, np.ndarray]
        Computed metric matrices.
    channel_names : Sequence[str]
        Channel names for headers and index.
    output_dir : Path
        Directory where CSV files will be saved.
    run_id : str
        Run identifier for filename templating.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping metric name -> absolute or resolved path string to the CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: Dict[str, str] = {}

    for metric_name, matrix in matrices.items():
        filename = f"{metric_name}_{run_id}.csv"
        file_path = output_dir / filename

        df = pd.DataFrame(matrix, index=list(channel_names), columns=list(channel_names))
        df.to_csv(file_path, index=True)
        csv_paths[metric_name] = str(file_path.resolve())

    return csv_paths


def export_evidence_summary_to_csv(
    evidence_list: List[PairEvidenceSynthesis],
    matrices: Dict[str, np.ndarray],
    channel_names: Sequence[str],
    output_dir: Path,
    run_id: str,
) -> str:
    """Exports consolidated pairwise evidence synthesis and metric values to CSV.

    Parameters
    ----------
    evidence_list : List[PairEvidenceSynthesis]
        Pairwise classifications and group means.
    matrices : Dict[str, np.ndarray]
        Computed metric matrices to look up individual pair values.
    channel_names : Sequence[str]
        Ordered channel names.
    output_dir : Path
        Target directory for evidence_summary_<run_id>.csv.
    run_id : str
        Pipeline run ID.

    Returns
    -------
    str
        Resolved path to the evidence summary CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_filename = f"evidence_summary_{run_id}.csv"
    summary_path = output_dir / summary_filename

    ch_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    computed_metrics = sorted(list(matrices.keys()))

    fieldnames = [
        "channel_1",
        "channel_2",
        "phase_robust_mean",
        "zero_lag_mean",
        "classification",
    ] + computed_metrics

    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in evidence_list:
            ch1, ch2 = item.channel_pair[0], item.channel_pair[1]
            i = ch_to_idx.get(ch1)
            j = ch_to_idx.get(ch2)

            row: Dict[str, object] = {
                "channel_1": ch1,
                "channel_2": ch2,
                "phase_robust_mean": item.phase_robust_mean,
                "zero_lag_mean": item.zero_lag_mean,
                "classification": item.classification.value,
            }

            for m in computed_metrics:
                if i is not None and j is not None:
                    row[m] = round(float(matrices[m][i, j]), 6)
                else:
                    row[m] = ""

            writer.writerow(row)

    return str(summary_path.resolve())


def export_consolidated_connectivity_report_csv(
    evidence_list: List[PairEvidenceSynthesis],
    matrices: Dict[str, np.ndarray],
    channel_names: Sequence[str],
    output_dir: Path,
    run_id: str,
    condition: Optional[str] = None,
    freq_band: Optional[str] = None,
    subject: Optional[str] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> str:
    """Exports ONE consolidated CSV with all five connectivity metrics.

    One row per unique (unordered) channel pair for the analysis context
    (subject / condition / run / frequency band). Metric values are joined to
    each pair by an explicit canonical channel-pair key, never by row position.

    Parameters
    ----------
    evidence_list : List[PairEvidenceSynthesis]
        Pairwise classifications and group means (joined by pair key).
    matrices : Dict[str, np.ndarray]
        Computed metric matrices (pli, wpli, imcoh, plv, coh), indexed by
        ``channel_names``.
    channel_names : Sequence[str]
        Ordered channel names of the matrices.
    output_dir : Path
        Target directory for the consolidated report.
    run_id : str
        Pipeline run ID.
    condition, freq_band, subject, fmin, fmax : optional
        Analysis-context metadata. Written as empty fields when unavailable
        (nothing is invented).

    Returns
    -------
    str
        Resolved path to the run-specific consolidated CSV report file.
    """
    import math
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"consolidated_connectivity_report_{run_id}.csv"
    report_path = output_dir / report_filename

    channel_order: Dict[str, int] = {}
    for name in channel_names:
        channel_order.setdefault(str(name), len(channel_order))
    # Channels that only appear in the evidence list are appended, never dropped.
    for item in evidence_list:
        for ch in item.channel_pair[:2]:
            channel_order.setdefault(str(ch), len(channel_order))

    def _pair_key(a: object, b: object) -> Tuple[str, str]:
        """Canonical, order-independent key: sorted by channel order."""
        a, b = str(a), str(b)
        return (a, b) if channel_order[a] <= channel_order[b] else (b, a)

    metric_map = [
        ("PLI", ["pli", "PLI"]),
        ("wPLI", ["wpli", "wPLI"]),
        ("ImCoh", ["imcoh", "ImCoh", "im_coh"]),
        ("PLV", ["plv", "PLV"]),
        ("Coherence", ["coh", "coherence", "COH"]),
    ]

    # --- metric values keyed by canonical channel pair ---
    n_ch = len(channel_names)
    metric_values: Dict[str, Dict[Tuple[str, str], float]] = {}
    for header, candidate_keys in metric_map:
        values: Dict[Tuple[str, str], float] = {}
        mat = next((matrices[k] for k in candidate_keys if k in matrices), None)
        if mat is not None and getattr(mat, "shape", None) == (n_ch, n_ch):
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    v = float(mat[i, j])
                    if math.isfinite(v):  # non-finite -> leave unavailable (empty)
                        values[_pair_key(channel_names[i], channel_names[j])] = v
        elif mat is not None:
            logger.warning(
                "Metric %s matrix shape %s does not match %d channels; left empty.",
                header, getattr(mat, "shape", None), n_ch,
            )
        metric_values[header] = values

    # --- evidence keyed by canonical channel pair (first occurrence wins) ---
    evidence_by_key: Dict[Tuple[str, str], PairEvidenceSynthesis] = {}
    for item in evidence_list:
        key = _pair_key(item.channel_pair[0], item.channel_pair[1])
        if key in evidence_by_key:
            logger.warning("Duplicate evidence entry for pair %s ignored.", key)
            continue
        evidence_by_key[key] = item

    # --- unique pair universe: every matrix pair + any evidence-only pair ---
    all_keys: List[Tuple[str, str]] = []
    seen = set()
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            key = _pair_key(channel_names[i], channel_names[j])
            if key not in seen:
                seen.add(key)
                all_keys.append(key)
    for key in evidence_by_key:
        if key not in seen and key[0] != key[1]:
            seen.add(key)
            all_keys.append(key)

    fieldnames = [
        "subject",
        "condition",
        "run_id",
        "freq_band",
        "fmin_hz",
        "fmax_hz",
        "channel_1",
        "channel_2",
        "PLI",
        "wPLI",
        "ImCoh",
        "PLV",
        "Coherence",
        "phase_robust_mean",
        "zero_lag_mean",
        "classification",
    ]

    def _cell(v: object) -> object:
        if v is None:
            return ""
        if isinstance(v, (int, float, np.floating)):
            f = float(v)
            return repr(f) if math.isfinite(f) else ""
        return v

    with open(report_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in all_keys:
            ev = evidence_by_key.get(key)
            if ev is not None:
                cat = ev.classification
                cat_val = cat.value if hasattr(cat, "value") else str(cat)
                phase_mean, zero_mean = ev.phase_robust_mean, ev.zero_lag_mean
            else:
                cat_val, phase_mean, zero_mean = "", None, None

            row: Dict[str, object] = {
                "subject": subject or "",
                "condition": condition or "",
                "run_id": run_id,
                "freq_band": freq_band or "",
                "fmin_hz": _cell(fmin),
                "fmax_hz": _cell(fmax),
                "channel_1": key[0],
                "channel_2": key[1],
                "phase_robust_mean": _cell(phase_mean),
                "zero_lag_mean": _cell(zero_mean),
                "classification": cat_val,
            }
            for header, _ in metric_map:
                row[header] = _cell(metric_values[header].get(key))
            writer.writerow(row)

    generic_path = output_dir / "consolidated_connectivity_report.csv"
    try:
        shutil.copyfile(report_path, generic_path)
    except Exception as e:
        logger.warning("Could not copy consolidated report to generic path: %s", e)

    return str(report_path.resolve())
