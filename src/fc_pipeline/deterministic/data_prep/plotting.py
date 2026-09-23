"""Pre/post cleaning channel time-series diagnostic figure rendering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import mne

from fc_pipeline.deterministic.data_prep.models import ValidatedDataPrepParams
from fc_pipeline.deterministic.data_prep.validation import (
    DataPrepValidationError,
    build_safe_output_path,
)

logger = logging.getLogger(__name__)


def generate_diagnostics(
    epochs: mne.Epochs,
    params: ValidatedDataPrepParams,
    output_dir: Path,
) -> Dict[str, str]:
    """Generate diagnostic plots for the preprocessed data.

    Produces:
      1. **channel_variance** — Bar chart of per-channel variance across epochs,
         useful for spotting residual noisy or dead channels.
      2. **psd_overview** — Power spectral density across channels with the
         analysis band highlighted, confirming the bandpass filter worked.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed, epoched EEG data.
    params : ValidatedDataPrepParams
        Validated parameters (run_id, fmin, fmax, condition, …).
    output_dir : pathlib.Path
        Directory where plots are saved.  Must already exist.

    Returns
    -------
    plot_paths : dict[str, str]
        Mapping ``{plot_name: absolute_path_string}``.

    Raises
    ------
    DataPrepValidationError
        If the output directory is invalid or a plot cannot be written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: Dict[str, str] = {}

    # --- 1. Channel variance overview ---
    try:
        variance_path = build_safe_output_path(
            output_dir, params.run_id, "channel_variance", ".png"
        )
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        # Mean variance across epochs and time for each channel
        channel_variances = np.var(data, axis=2).mean(axis=0)
        ch_names = epochs.ch_names

        fig, ax = plt.subplots(figsize=(max(6, len(ch_names) * 0.8), 4))
        bars = ax.bar(range(len(ch_names)), channel_variances, color="#4a90d9", edgecolor="#2c5f8a")
        ax.set_xticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Variance (V²)")
        ax.set_title(
            f"Channel Variance — {params.condition} "
            f"({len(epochs)} epochs, {params.fmin}–{params.fmax} Hz)"
        )
        ax.set_xlabel("Channel")
        fig.tight_layout()
        fig.savefig(str(variance_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        plot_paths["channel_variance"] = str(variance_path)
        logger.info("Channel variance plot saved: %s", variance_path.name)

    except DataPrepValidationError:
        raise
    except Exception as exc:
        logger.warning("Channel variance plot failed: %s: %s", type(exc).__name__, exc)

    # --- 2. PSD overview ---
    try:
        psd_path = build_safe_output_path(
            output_dir, params.run_id, "psd_overview", ".png"
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        # Compute PSD using the epochs data
        spectrum = epochs.compute_psd(
            method="welch",
            fmin=max(0.1, params.fmin * 0.5),
            fmax=min(params.fmax * 2.0, epochs.info["sfreq"] / 2.0 - 0.1),
            verbose=False,
        )
        psds = spectrum.get_data()  # (n_epochs, n_channels, n_freqs)
        freqs = spectrum.freqs

        # Average across epochs, plot each channel
        mean_psds = psds.mean(axis=0)  # (n_channels, n_freqs)
        for i, ch_name in enumerate(epochs.ch_names):
            ax.semilogy(freqs, mean_psds[i], linewidth=0.8, alpha=0.7, label=ch_name)

        # Highlight the analysis frequency band
        ax.axvspan(params.fmin, params.fmax, alpha=0.15, color="orange", label="Analysis band")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.set_title(
            f"Power Spectral Density — {params.condition} "
            f"({len(epochs)} epochs)"
        )
        # Place legend outside if many channels
        if len(epochs.ch_names) <= 10:
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.legend(fontsize=6, loc="upper right", ncol=2)
        fig.tight_layout()
        fig.savefig(str(psd_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        plot_paths["psd_overview"] = str(psd_path)
        logger.info("PSD overview plot saved: %s", psd_path.name)

    except DataPrepValidationError:
        raise
    except Exception as exc:
        logger.warning("PSD overview plot failed: %s: %s", type(exc).__name__, exc)

    return plot_paths
