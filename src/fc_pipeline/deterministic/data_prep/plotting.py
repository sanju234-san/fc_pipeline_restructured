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

# --- Light theme palette for the channel time-series figures -----------------
# Traces keep their original colour identity (teal = normal, orange = plan,
# red = flagged/boundary) but in shades that stay legible on a white background.
_BG = "#ffffff"        # figure / axes background
_FG = "#1a1a1a"        # titles
_TICK = "#555555"      # axis labels & ticks
_AXIS = "#999999"      # last-row spine
_ZERO = "#cccccc"      # zero-reference line
_TRACE = "#0e7c86"     # normal trace (dark teal)
_TRACE_PLAN = "#b9770e"  # plan-selected channel (dark amber)
_TRACE_BAD = "#c0392b"   # flagged / flatline channel (dark red)
_LABEL = "#3a3a3a"     # channel label text
_BOUNDARY = "#c0392b"  # epoch boundary dashes


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
    """Render ALL EEG channels before preprocessing as a stacked matplotlib figure.

    Every channel gets its own row so the PNG is self-contained and shows
    the full montage without any scrollbar.  The snippet is centred on the
    first annotation of the requested condition (or the recording start).
    Flagged flatline channels are drawn in red; plan-selected channels in gold.
    Plot failures never abort Data Prep: returns ``None`` after logging.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = build_safe_output_path(output_dir, params.run_id, "channels_before", ".png")

        sfreq = float(raw.info["sfreq"])
        all_eeg = mne.pick_types(raw.info, eeg=True)
        if len(all_eeg) > 0:
            picks = list(all_eeg)
        else:
            picks = (
                [raw.ch_names.index(ch) for ch in params.channels if ch in raw.ch_names]
                or list(range(len(raw.ch_names)))
            )
        if not picks:
            return None

        # Window: start at first condition event (or recording start)
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

        data = raw.get_data(picks=picks, start=start_sample, stop=start_sample + n_window)
        ch_names = [raw.ch_names[i] for i in picks]
        t_axis = np.linspace(0.0, n_window / sfreq, n_window)

        flagged_set = set(flagged_channels)
        plan_set = set(params.channels)

        n_ch = len(ch_names)
        # Cap total figure height so it fits in a browser viewport without scrolling.
        # 9 inches at 130 DPI ≈ 1170 pixels (fits any 1080p+ screen).
        _MAX_FIG_H = 9.0
        row_h = (_MAX_FIG_H - 1.4) / max(n_ch, 1)
        fig_h = _MAX_FIG_H
        lbl_fs = max(4.5, min(8.0, row_h * 52))  # font size scales with row height

        # --- light-themed figure ---
        fig, axes = plt.subplots(
            n_ch, 1,
            figsize=(16, fig_h),
            sharex=True,
            facecolor=_BG,
        )
        if n_ch == 1:
            axes = [axes]

        # global amplitude scale: 3× peak std across all good channels
        good_mask = np.array([ch not in flagged_set for ch in ch_names])
        good_data = data[good_mask] if good_mask.any() else data
        peak_std = float(np.max(np.std(good_data, axis=1))) if good_data.size else 20e-6
        amp = 3.0 * peak_std if peak_std > 0 else 20e-6

        for i, (ax, ch, trace) in enumerate(zip(axes, ch_names, data)):
            ax.set_facecolor(_BG)
            ax.set_xlim(t_axis[0], t_axis[-1])
            ax.set_ylim(-amp, amp)
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            ax.spines[:].set_visible(False)

            is_bad = ch in flagged_set
            is_plan = ch in plan_set

            if is_bad:
                color, lw, lbl_color = _TRACE_BAD, 0.7, _TRACE_BAD
                label = f"{ch} ✕"
            elif is_plan:
                color, lw, lbl_color = _TRACE_PLAN, 0.9, _TRACE_PLAN
                label = f"★ {ch}"
            else:
                color, lw, lbl_color = _TRACE, 0.7, _LABEL
                label = ch

            ax.plot(t_axis, trace, color=color, linewidth=lw, rasterized=True)
            ax.axhline(0, color=_ZERO, linewidth=0.4, zorder=0)
            ax.text(
                -0.01, 0.5, label,
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=6.5, color=lbl_color,
                fontfamily="monospace",
            )

        # x-axis on the last row only
        axes[-1].tick_params(bottom=True, labelbottom=True, colors=_TICK)
        axes[-1].set_xlabel("Time (s)", color=_TICK, fontsize=8)
        axes[-1].spines["bottom"].set_visible(True)
        axes[-1].spines["bottom"].set_color(_AXIS)

        fig.suptitle(
            f"Channels Before — {params.condition}: first {n_window / sfreq:.1f} s, "
            f"unfiltered, unreferenced  ★=plan  ✕=flagline",
            color=_FG, fontsize=9, y=1.0,
        )
        fig.tight_layout(h_pad=0.0)
        fig.savefig(str(path), dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Channels-before plot saved: %s (%d channels)", path.name, n_ch)
        return str(path)
    except Exception as exc:
        logger.warning("Channels-before plot failed: %s: %s", type(exc).__name__, exc)
        return None



def plot_channels_after(
    epochs: mne.Epochs,
    params: ValidatedDataPrepParams,
    output_dir: Path,
) -> Optional[str]:
    """Render ALL preprocessed channels as a stacked matplotlib figure.

    Every channel gets its own row so the PNG is self-contained and shows
    the complete filtered/referenced montage without any scrollbar.
    Plan-selected channels are highlighted in gold.  Concatenates the first
    few epochs along the time axis for a compact but meaningful view.
    Plot failures never abort Data Prep: returns ``None`` after logging.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = build_safe_output_path(output_dir, params.run_id, "channels_after", ".png")

        n_show_epochs = min(len(epochs), _MAX_EPOCHS_SHOWN)
        ep_data = epochs[:n_show_epochs].get_data()  # (n_ep, n_ch, n_t)
        n_ep, n_ch, n_t = ep_data.shape
        sfreq = float(epochs.info["sfreq"])
        ep_dur = n_t / sfreq

        # Concatenate epochs along time with a small NaN gap so epoch
        # boundaries are visible as breaks in the line.
        gap_samples = max(1, int(0.02 * sfreq))
        gap = np.full((n_ch, gap_samples), np.nan)
        parts = []
        for e in range(n_ep):
            parts.append(ep_data[e])  # (n_ch, n_t)
            if e < n_ep - 1:
                parts.append(gap)
        data = np.concatenate(parts, axis=1)  # (n_ch, total_samples)
        total_t = data.shape[1] / sfreq
        t_axis = np.linspace(0.0, total_t, data.shape[1])

        ch_names = list(epochs.ch_names)
        plan_set = set(params.channels)

        # Global amplitude scale (ignoring NaN gaps)
        finite = data[np.isfinite(data).all(axis=1)]
        peak_std = float(np.nanmax(np.nanstd(finite, axis=1))) if finite.size else 20e-6
        amp = 3.0 * peak_std if peak_std > 0 else 20e-6

        n_ch = len(ch_names)
        # Cap total figure height so it always fits in a browser viewport.
        _MAX_FIG_H = 9.0
        row_h = (_MAX_FIG_H - 1.4) / max(n_ch, 1)
        fig_h = _MAX_FIG_H
        lbl_fs = max(4.5, min(8.0, row_h * 52))

        fig, axes = plt.subplots(
            n_ch, 1,
            figsize=(16, fig_h),
            sharex=True,
            facecolor=_BG,
        )
        if n_ch == 1:
            axes = [axes]

        for i, (ax, ch, trace) in enumerate(zip(axes, ch_names, data)):
            ax.set_facecolor(_BG)
            ax.set_xlim(t_axis[0], t_axis[-1])
            ax.set_ylim(-amp, amp)
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            ax.spines[:].set_visible(False)

            is_plan = ch in plan_set
            color = _TRACE_PLAN if is_plan else _TRACE
            lw = 0.9 if is_plan else 0.7
            lbl_color = _TRACE_PLAN if is_plan else _LABEL
            label = f"\u2605 {ch}" if is_plan else ch

            ax.plot(t_axis, trace, color=color, linewidth=lw, rasterized=True)
            ax.axhline(0, color=_ZERO, linewidth=0.4, zorder=0)
            ax.text(
                -0.01, 0.5, label,
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=6.5, color=lbl_color,
                fontfamily="monospace",
            )

            # epoch boundary markers
            for e in range(1, n_ep):
                x_boundary = e * (ep_dur + gap_samples / sfreq)
                ax.axvline(x_boundary, color=_BOUNDARY, linewidth=0.5,
                           linestyle="--", alpha=0.5, zorder=1)

        axes[-1].tick_params(bottom=True, labelbottom=True, colors=_TICK)
        axes[-1].set_xlabel("Time (s)", color=_TICK, fontsize=8)
        axes[-1].spines["bottom"].set_visible(True)
        axes[-1].spines["bottom"].set_color(_AXIS)

        fig.suptitle(
            f"Channels After — {params.condition}: {n_show_epochs} epochs, "
            f"{params.fmin:g}\u2013{params.fmax:g} Hz, average-ref  "
            f"\u2605=plan  red dashes=epoch boundaries",
            color=_FG, fontsize=9, y=1.0,
        )
        fig.tight_layout(h_pad=0.0)
        fig.savefig(str(path), dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Channels-after plot saved: %s (%d channels)", path.name, n_ch)
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
