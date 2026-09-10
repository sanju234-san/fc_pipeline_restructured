"""Tool: generate_dataset_overview_plot (PSD + raw snippet quick-look)."""

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import mne
from langchain_core.tools import tool


@tool
def generate_dataset_overview_plot(data_path: str, run_id: str = "default") -> Dict[str, Any]:
    """Generates a two-panel quick-look overview plot of the EEG dataset.

    Left panel: Raw signal snippet (first 4 seconds) across all EEG channels.
    Right panel: Power Spectral Density (PSD) via Welch's method for all channels.

    The plot is saved to outputs/plots/{run_id}_overview.png.
    All subject metadata, file paths, and patient identifiers are strictly excluded
    from titles, labels, and annotations to enforce privacy masking.

    Args:
        data_path: Path to the raw EEG file (.fif, .edf, .bdf, .set, etc.).
        run_id: Unique identifier for the current pipeline run (used in filename).

    Returns:
        Dictionary with 'plot_path' (str), 'n_channels' (int), 'duration_seconds' (float),
        'sfreq' (float), and 'error' (str or None).
    """
    try:
        raw = mne.io.read_raw(data_path, preload=True, verbose=False)
        sfreq = float(raw.info["sfreq"])
        duration = float(raw.n_times / sfreq) if sfreq > 0 else 0.0

        # Filter to EEG channels only
        try:
            ch_types = raw.get_channel_types()
            eeg_picks = [i for i, t in enumerate(ch_types) if t == "eeg"]
            if eeg_picks:
                raw = raw.pick(eeg_picks)
        except Exception:
            pass  # Keep all channels if type detection fails

        ch_names = raw.ch_names
        n_channels = len(ch_names)
        data = raw.get_data()  # (n_channels, n_times)

        # Determine snippet length: min(4 seconds, full duration)
        snippet_samples = min(int(4.0 * sfreq), data.shape[1])
        snippet_time = np.arange(snippet_samples) / sfreq
        snippet_data = data[:, :snippet_samples]

        # Create two-panel figure
        fig, (ax_raw, ax_psd) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Dataset Overview (Pre-Preprocessing)", fontsize=13, fontweight="bold")

        # --- Left panel: Raw signal snippet ---
        # Normalize each channel for visual stacking
        offsets = np.arange(n_channels) * 1.0
        for i in range(n_channels):
            ch_data = snippet_data[i]
            if np.std(ch_data) > 0:
                ch_norm = (ch_data - np.mean(ch_data)) / np.std(ch_data)
            else:
                ch_norm = ch_data - np.mean(ch_data)
            ax_raw.plot(snippet_time, ch_norm * 0.3 + offsets[i], linewidth=0.5, alpha=0.85)

        ax_raw.set_yticks(offsets)
        ax_raw.set_yticklabels(ch_names, fontsize=7)
        ax_raw.set_xlabel("Time (s)", fontsize=9)
        ax_raw.set_title(f"Raw Signal Snippet ({snippet_time[-1]:.1f}s, {n_channels} ch)", fontsize=10)
        ax_raw.set_xlim(snippet_time[0], snippet_time[-1])
        ax_raw.grid(axis="x", alpha=0.3)

        # --- Right panel: PSD via Welch ---
        from scipy.signal import welch as scipy_welch

        nperseg = min(int(sfreq * 2), data.shape[1])
        for i in range(n_channels):
            freqs, psd = scipy_welch(data[i], fs=sfreq, nperseg=nperseg)
            # Limit display to 0–60 Hz (or Nyquist if lower)
            freq_mask = freqs <= min(60.0, sfreq / 2.0)
            ax_psd.semilogy(freqs[freq_mask], psd[freq_mask], linewidth=0.7, alpha=0.7, label=ch_names[i])

        ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_psd.set_ylabel("PSD (V²/Hz)", fontsize=9)
        ax_psd.set_title("Power Spectral Density (Welch)", fontsize=10)
        ax_psd.grid(alpha=0.3)
        # Show legend only if ≤ 16 channels to avoid clutter
        if n_channels <= 16:
            ax_psd.legend(fontsize=6, ncol=2, loc="upper right")

        plt.tight_layout()

        # Save plot — no file path, subject info, or metadata in the figure
        out_dir = Path("outputs") / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / f"{run_id}_overview.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return {
            "plot_path": str(plot_path),
            "n_channels": n_channels,
            "duration_seconds": round(duration, 2),
            "sfreq": sfreq,
            "error": None,
        }
    except Exception as e:
        return {
            "plot_path": None,
            "n_channels": 0,
            "duration_seconds": 0.0,
            "sfreq": 0.0,
            "error": f"Failed to generate overview plot: {str(e)}",
        }
