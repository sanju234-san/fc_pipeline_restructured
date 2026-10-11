"""Pre/post cleaning channel time-series diagnostic figure rendering.

The channel time-series plots (``channels_before`` / ``channels_after``) are
drawn with MNE's own browser (``Raw.plot`` / ``Epochs.plot``, matplotlib
backend) so they look and behave like standard MNE diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

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


_BROWSER_WINDOW_SECONDS = 10.0
_MAX_EPOCHS_SHOWN = 10
_MAX_CHANNELS_SHOWN = 20


def _eeg_scaling(channel_data: np.ndarray) -> float:
    """Explicit trace scaling (volts) for MNE's browser, from the data itself.

    MNE's ``scalings="auto"`` can be thrown off by flat or outlier channels
    and then draws invisible traces.  Three times the largest per-channel
    standard deviation keeps every trace inside its lane.

    ``channel_data`` has shape ``(n_channels, n_samples)``.
    """
    if channel_data.size == 0:
        return 20e-6
    peak = float(np.max(np.std(channel_data, axis=1)))
    return 3.0 * peak if peak > 0 else 20e-6


def plot_channels_before(
    raw: mne.io.BaseRaw,
    params: ValidatedDataPrepParams,
    flagged_channels: Sequence[str],
    output_dir: Path,
) -> Optional[str]:
    """Render the selected channels as they were recorded (MNE ``Raw.plot``).

    Shows the unfiltered, unreferenced signal of the analysis channels in a
    window that starts at the first event of the requested condition.
    Channels flagged as flatline are drawn in the MNE "bad" colour.

    Only a short snippet is copied, so the full recording is never duplicated.
    Plot failures never abort Data Prep: returns ``None`` after logging.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = build_safe_output_path(output_dir, params.run_id, "channels_before", ".png")

        sfreq = float(raw.info["sfreq"])
        picks = [raw.ch_names.index(ch) for ch in params.channels if ch in raw.ch_names]
        if not picks:
            return None

        # Window start: first event of the condition (fall back to recording start)
        start_sample = 0
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if params.condition in event_id:
                hits = events[events[:, 2] == event_id[params.condition], 0]
                if len(hits):
                    start_sample = max(0, int(hits.min()) - raw.first_samp)
        except Exception:
            pass
        n_window = int(min(_BROWSER_WINDOW_SECONDS * sfreq, raw.n_times - start_sample))
        if n_window < 2:
            start_sample, n_window = 0, int(min(_BROWSER_WINDOW_SECONDS * sfreq, raw.n_times))

        snippet = raw.get_data(picks=picks, start=start_sample, stop=start_sample + n_window)
        info = mne.pick_info(raw.info, picks, copy=True)
        info["bads"] = [ch for ch in info["ch_names"] if ch in set(flagged_channels)]
        view = mne.io.RawArray(snippet, info, verbose=False)
        good_rows = [i for i, ch in enumerate(info["ch_names"]) if ch not in info["bads"]]
        scaling = _eeg_scaling(snippet[good_rows])

        with mne.viz.use_browser_backend("matplotlib"):
            fig = view.plot(
                duration=n_window / sfreq,
                n_channels=min(len(picks), _MAX_CHANNELS_SHOWN),
                scalings={"eeg": scaling},
                show=False,
                block=False,
                title=f"Before Data Prep - {params.condition} (first {n_window / sfreq:.1f} s, "
                f"unfiltered, unreferenced)",
            )
            fig.mne.ax_main.set_title(
                f"Before Data Prep - {params.condition}: first {n_window / sfreq:.1f} s, "
                "unfiltered, unreferenced (flagged channels greyed)",
                fontsize=9,
            )
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(fig)

        logger.info("Channels-before plot saved: %s", path.name)
        return str(path)
    except Exception as exc:
        logger.warning("Channels-before plot failed: %s: %s", type(exc).__name__, exc)
        return None


def plot_channels_after(
    epochs: mne.Epochs,
    params: ValidatedDataPrepParams,
    output_dir: Path,
) -> Optional[str]:
    """Render the final epochs with MNE's ``Epochs.plot`` (matplotlib backend).

    Shows the first few epochs of the filtered, referenced analysis channels
    exactly as they are handed to the connectivity stage.
    Plot failures never abort Data Prep: returns ``None`` after logging.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = build_safe_output_path(output_dir, params.run_id, "channels_after", ".png")

        epoch_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        scaling = _eeg_scaling(
            epoch_data.transpose(1, 0, 2).reshape(epoch_data.shape[1], -1)
        )

        with mne.viz.use_browser_backend("matplotlib"):
            fig = epochs.plot(
                n_epochs=min(len(epochs), _MAX_EPOCHS_SHOWN),
                n_channels=min(len(epochs.ch_names), _MAX_CHANNELS_SHOWN),
                scalings={"eeg": scaling},
                show=False,
                block=False,
                title=f"After Data Prep - {params.condition} "
                f"({len(epochs)} epochs, {params.fmin:g}-{params.fmax:g} Hz, referenced)",
            )
            fig.mne.ax_main.set_title(
                f"After Data Prep - {params.condition}: {len(epochs)} epochs, "
                f"{params.fmin:g}-{params.fmax:g} Hz, average-referenced",
                fontsize=9,
            )
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(fig)

        logger.info("Channels-after plot saved: %s", path.name)
        return str(path)
    except Exception as exc:
        logger.warning("Channels-after plot failed: %s: %s", type(exc).__name__, exc)
        return None


def generate_diagnostics(
    epochs: mne.Epochs,
    params: ValidatedDataPrepParams,
    output_dir: Path,
) -> Dict[str, str]:
    """Generate diagnostic plots for the preprocessed data.

    Produces:
      0. **channels_after** - MNE time-series view of the final epochs.
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

    # --- 0. Final epochs, MNE browser view ---
    after_path = plot_channels_after(epochs, params, output_dir)
    if after_path:
        plot_paths["channels_after"] = after_path

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

        # Native MNE plotting call
        spectrum.plot(average=False, spatial_colors=False, axes=ax, show=False)

        # Highlight the analysis frequency band
        ax.axvspan(params.fmin, params.fmax, alpha=0.15, color="orange", label="Analysis band")
        ax.set_title(
            f"Power Spectral Density — {params.condition} "
            f"({len(epochs)} epochs)"
        )

        # Distinguish channels by color and label for the legend
        n_freqs = len(spectrum.freqs)
        ch_lines = [line for line in ax.lines if len(line.get_xdata()) == n_freqs]
        if len(ch_lines) == len(epochs.ch_names):
            cmap = matplotlib.colormaps["tab10" if len(epochs.ch_names) <= 10 else "tab20"]
            for i, (line, ch_name) in enumerate(zip(ch_lines, epochs.ch_names)):
                line.set_color(cmap(i % cmap.N))
                line.set_label(ch_name)
        else:
            logger.warning(
                "Expected %d channel curves from Spectrum.plot, but found %d",
                len(epochs.ch_names),
                len(ch_lines),
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
