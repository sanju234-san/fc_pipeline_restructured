"""Ocular variance ratio, muscle band power, and electrode pop sigma checks.

Follows Section 6.4 Step 2 of PIPELINE_PROTOTYPE_DESIGN.md.
Strictly deterministic: zero LLM / VLM / MLflow imports.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.signal import welch as scipy_welch

from fc_pipeline.config.thresholds import (
    ELECTRODE_POP_SIGMA,
    MUSCLE_POWER_THRESHOLD,
    OCULAR_VARIANCE_RATIO,
    TAU_ZEROLAG,
)
from fc_pipeline.schemas.evaluation import ArtifactDetectionResult, ArtifactSignatures

_FRONTO_POLAR_SET = {"FP1", "FP2", "AF3", "AF4", "FPZ", "AF7", "AF8"}
_POSTERIOR_SET = {"O1", "O2", "OZ", "P7", "P8", "P3", "P4", "PO3", "PO4"}
_TEMPORAL_OCCIPITAL_SET = {
    "T7", "T8", "T3", "T4", "T5", "T6", "TP7", "TP8", "TP9", "TP10",
    "FT7", "FT8", "O1", "O2", "OZ",
}


def _boundary_confidence(measured: float, threshold: float) -> float:
    """Computes confidence in [0.5, 1.0] based on normalized distance from threshold."""
    scale = max(abs(threshold), 1e-9)
    dist = abs(measured - threshold) / scale
    return round(float(np.clip(0.5 + 0.5 * np.tanh(dist), 0.5, 1.0)), 4)


def detect_ocular_contamination(
    channel_variances: Dict[str, float],
    plv_matrix: Optional[np.ndarray] = None,
    channel_names: Optional[Sequence[str]] = None,
    ocular_variance_ratio: float = OCULAR_VARIANCE_RATIO,
    tau_zerolag: float = TAU_ZEROLAG,
) -> ArtifactDetectionResult:
    """Detects fronto-polar vs posterior variance elevation and fronto-polar zero-lag PLV."""
    fp_channels = [ch for ch in channel_variances if ch.upper() in _FRONTO_POLAR_SET]
    post_channels = [ch for ch in channel_variances if ch.upper() in _POSTERIOR_SET]

    if not fp_channels or not post_channels:
        return ArtifactDetectionResult(
            detected=False,
            affected_channels=[],
            confidence=1.0,
            detail=(
                "Fronto-polar or posterior reference channels not both present in active montage; "
                f"ocular variance ratio check skipped (threshold={ocular_variance_ratio:.2f})."
            ),
        )

    fp_mean = float(np.mean([channel_variances[c] for c in fp_channels]))
    post_mean = float(np.mean([channel_variances[c] for c in post_channels]))
    ratio = fp_mean / max(post_mean, 1e-18)

    plv_note = ""
    if plv_matrix is not None and channel_names is not None and len(fp_channels) >= 2:
        name_to_idx = {str(n): idx for idx, n in enumerate(channel_names)}
        fp_plvs: List[float] = []
        for i, c1 in enumerate(fp_channels):
            for c2 in fp_channels[i + 1 :]:
                if c1 in name_to_idx and c2 in name_to_idx:
                    fp_plvs.append(float(plv_matrix[name_to_idx[c1], name_to_idx[c2]]))
        if fp_plvs:
            max_fp_plv = max(fp_plvs)
            plv_note = f"; max fronto-polar PLV={max_fp_plv:.3f} (tau_zerolag={tau_zerolag:.2f})"

    detected = bool(ratio > ocular_variance_ratio)
    conf = _boundary_confidence(ratio, ocular_variance_ratio)
    return ArtifactDetectionResult(
        detected=detected,
        affected_channels=fp_channels if detected else [],
        confidence=conf,
        detail=(
            f"Fronto-polar/posterior variance ratio={ratio:.2f} vs threshold={ocular_variance_ratio:.2f}"
            f"{plv_note}."
        ),
    )


def detect_muscle_contamination(
    high_freq_power: Dict[str, float],
    full_band_power: Dict[str, float],
    muscle_power_threshold: float = MUSCLE_POWER_THRESHOLD,
) -> ArtifactDetectionResult:
    """Detects broadband high-frequency (>40 Hz) power elevation at temporal/occipital channels."""
    if not high_freq_power or not full_band_power:
        return ArtifactDetectionResult(
            detected=False,
            affected_channels=[],
            confidence=1.0,
            detail=(
                f"High-frequency (>40 Hz) power unavailable or Nyquist <= 40 Hz "
                f"(threshold={muscle_power_threshold:.2f})."
            ),
        )

    candidates = [ch for ch in high_freq_power if ch.upper() in _TEMPORAL_OCCIPITAL_SET]
    if not candidates:
        candidates = list(high_freq_power.keys())

    mean_full = float(np.mean(list(full_band_power.values())))
    denom = max(mean_full, 1e-18)

    flagged: List[str] = []
    max_ratio = 0.0
    for ch in candidates:
        r = float(high_freq_power[ch]) / denom
        if r > max_ratio:
            max_ratio = r
        if r > muscle_power_threshold:
            flagged.append(ch)

    detected = len(flagged) > 0
    conf = _boundary_confidence(max_ratio, muscle_power_threshold)
    return ArtifactDetectionResult(
        detected=detected,
        affected_channels=flagged,
        confidence=conf,
        detail=(
            f"Max >40 Hz to full-band power ratio={max_ratio:.2f} vs threshold={muscle_power_threshold:.2f}."
        ),
    )


def detect_electrode_pop(
    channel_variances: Dict[str, float],
    electrode_pop_sigma: float = ELECTRODE_POP_SIGMA,
) -> ArtifactDetectionResult:
    """Flags channels whose variance exceeds mean + electrode_pop_sigma * std across channels."""
    if len(channel_variances) < 2:
        return ArtifactDetectionResult(
            detected=False,
            affected_channels=[],
            confidence=1.0,
            detail="Fewer than 2 channels available for cross-channel variance pop check.",
        )

    names = list(channel_variances.keys())
    vals = np.asarray([float(channel_variances[c]) for c in names], dtype=float)
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals))

    if std_v <= 1e-24:
        return ArtifactDetectionResult(
            detected=False,
            affected_channels=[],
            confidence=1.0,
            detail=f"Uniform channel variance (z_max=0.00 vs threshold={electrode_pop_sigma:.2f} sigma).",
        )

    z_scores = (vals - mean_v) / std_v
    max_z = float(np.max(z_scores))
    flagged = [names[i] for i, z in enumerate(z_scores) if z > electrode_pop_sigma]
    detected = len(flagged) > 0
    conf = _boundary_confidence(max_z, electrode_pop_sigma)
    return ArtifactDetectionResult(
        detected=detected,
        affected_channels=flagged,
        confidence=conf,
        detail=f"Max channel variance z-score={max_z:.2f} sigma vs threshold={electrode_pop_sigma:.2f} sigma.",
    )


def run_artifact_heuristics(
    preprocessed_data_path: Optional[str] = None,
    raw_data_path: Optional[str] = None,
    plv_matrix: Optional[np.ndarray] = None,
    channel_names: Optional[Sequence[str]] = None,
    ocular_variance_ratio: float = OCULAR_VARIANCE_RATIO,
    muscle_power_threshold: float = MUSCLE_POWER_THRESHOLD,
    electrode_pop_sigma: float = ELECTRODE_POP_SIGMA,
    tau_zerolag: float = TAU_ZEROLAG,
) -> ArtifactSignatures:
    """Computes channel statistics from preprocessed/raw EEG files and runs all 3 artifact checks."""
    import mne

    channel_variances: Dict[str, float] = {}
    high_freq_power: Dict[str, float] = {}
    full_band_power: Dict[str, float] = {}

    if preprocessed_data_path and Path(str(preprocessed_data_path)).is_file():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                epochs = mne.read_epochs(str(preprocessed_data_path), preload=True, verbose=False)
            data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            ch_list = list(epochs.ch_names)
            for idx, ch in enumerate(ch_list):
                channel_variances[ch] = float(np.var(data[:, idx, :]))
            if not channel_names:
                channel_names = ch_list
        except Exception:
            pass

    # Compute broadband (>40 Hz) vs full-band PSD from raw recording if available, else epochs
    psd_source_path = raw_data_path if (raw_data_path and Path(str(raw_data_path)).is_file()) else preprocessed_data_path
    if psd_source_path and Path(str(psd_source_path)).is_file():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low_p = str(psd_source_path).lower()
                if low_p.endswith(".edf"):
                    raw_obj = mne.io.read_raw_edf(str(psd_source_path), preload=True, verbose=False)
                    sfreq = float(raw_obj.info["sfreq"])
                    arr_2d = raw_obj.get_data()
                    psd_chs = list(raw_obj.ch_names)
                else:
                    try:
                        raw_obj = mne.io.read_raw_fif(str(psd_source_path), preload=True, verbose=False)
                        sfreq = float(raw_obj.info["sfreq"])
                        arr_2d = raw_obj.get_data()
                        psd_chs = list(raw_obj.ch_names)
                    except Exception:
                        ep_obj = mne.read_epochs(str(psd_source_path), preload=True, verbose=False)
                        sfreq = float(ep_obj.info["sfreq"])
                        ep_data = ep_obj.get_data()
                        arr_2d = np.concatenate(list(ep_data), axis=1)
                        psd_chs = list(ep_obj.ch_names)

            active_set = set(channel_names) if channel_names else set(psd_chs)
            if sfreq > 82.0 and arr_2d.shape[1] >= 16:
                nperseg = min(int(sfreq), arr_2d.shape[1])
                for idx, ch in enumerate(psd_chs):
                    if ch not in active_set:
                        continue
                    freqs, psd = scipy_welch(arr_2d[idx], fs=sfreq, nperseg=nperseg)
                    full_mask = (freqs >= 1.0) & (freqs <= min(sfreq / 2.0, 80.0))
                    hf_mask = (freqs > 40.0) & (freqs <= min(sfreq / 2.0, 80.0))
                    if np.any(full_mask) and np.any(hf_mask):
                        full_band_power[ch] = float(np.mean(psd[full_mask]))
                        high_freq_power[ch] = float(np.mean(psd[hf_mask]))
                    if ch not in channel_variances:
                        channel_variances[ch] = float(np.var(arr_2d[idx]))
        except Exception:
            pass

    return ArtifactSignatures(
        ocular_contamination=detect_ocular_contamination(
            channel_variances=channel_variances,
            plv_matrix=plv_matrix,
            channel_names=channel_names,
            ocular_variance_ratio=ocular_variance_ratio,
            tau_zerolag=tau_zerolag,
        ),
        muscle_contamination=detect_muscle_contamination(
            high_freq_power=high_freq_power,
            full_band_power=full_band_power,
            muscle_power_threshold=muscle_power_threshold,
        ),
        electrode_pop=detect_electrode_pop(
            channel_variances=channel_variances,
            electrode_pop_sigma=electrode_pop_sigma,
        ),
        top_unresolved_visual_reads=[],
    )
