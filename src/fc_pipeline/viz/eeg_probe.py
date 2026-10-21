"""Qt-free EEG data resolution/probing helpers for the interactive viewer.

These helpers deliberately contain **no PyQt imports** so they can be called
from the Chainlit server process (via ``viewer_launcher``) without dragging the
Qt widget stack — or ``eeg_viewer``'s hard ``sys.exit`` on missing PyQt5 — into
that process.

``eeg_viewer`` re-uses both functions, so plot-path resolution and probe
semantics stay identical between the preflight and the actual window.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# EEG container extensions the viewer can load with MNE
_EEG_SUFFIXES = (".edf", ".fif", ".bdf", ".set", ".vhdr")


def resolve_eeg_dataset(
    target_path: "str | Path",
    explicit_data: "Optional[str | Path]" = None,
) -> Optional[Path]:
    """Resolve the underlying raw or preprocessed EEG file from plot or data paths.

    Resolution order:
      1. ``explicit_data`` when it exists (what Chainlit passes from the run).
      2. ``target_path`` itself when it is an EEG container.
      3. ``run_context_{run_id}.json`` next to (or one level above) a
         ``channels_before_<run_id>.png`` / ``channels_after_<run_id>.png`` plot.
    """
    if explicit_data:
        p = Path(explicit_data)
        if p.exists():
            return p

    target = Path(target_path)
    if not target.exists():
        return None

    # Direct EEG data file
    if target.suffix.lower() in _EEG_SUFFIXES:
        return target

    # Plot image -> look for corresponding run_context_{run_id}.json
    m = re.search(r"channels_(?:before|after)_(.+)\.png", target.name)
    if m:
        run_id = m.group(1)
        search_dirs = [target.parent, target.parent.parent]
        for sdir in search_dirs:
            cand = sdir / f"run_context_{run_id}.json"
            if cand.exists():
                try:
                    with open(cand, "r", encoding="utf-8") as fh:
                        ctx = json.load(fh)
                    if "after" in target.name.lower() and ctx.get("preprocessed_data_path"):
                        p = Path(ctx["preprocessed_data_path"])
                        if p.exists():
                            return p
                    if ctx.get("raw_data_path"):
                        p = Path(ctx["raw_data_path"])
                        if p.exists():
                            return p
                except Exception as exc:
                    logger.warning("Error reading %s: %s", cand, exc)

    return None


def probe_eeg_dataset(data_path: "str | Path | None") -> Optional[Dict]:
    """Cheaply inspect an EEG file WITHOUT preloading sample data.

    Returns ``{"data_type": "raw"|"epochs", "n_channels": int,
    "sfreq": float, "ch_names": [...]}`` or ``None`` when the file cannot be
    read.  Used as a preflight by ``viewer_launcher`` so the UI can honestly
    report whether the *interactive* channel viewer (or only a static image)
    will be available, with the real channel count.
    """
    if not data_path:
        return None
    data_path = Path(data_path)
    if not data_path.exists():
        return None
    try:
        import mne
    except ImportError:
        logger.warning("probe_eeg_dataset: MNE is not available.")
        return None

    suffix = data_path.suffix.lower()
    if suffix == ".fif":
        # Mirror load_eeg_signals order: Epochs first, then Raw.
        try:
            epochs = mne.read_epochs(data_path, preload=False, verbose=False)
            return {
                "data_type": "epochs",
                "n_channels": len(epochs.ch_names),
                "sfreq": float(epochs.info["sfreq"]),
                "ch_names": list(epochs.ch_names),
            }
        except Exception:
            pass
        try:
            raw = mne.io.read_raw_fif(data_path, preload=False, verbose=False)
            return {
                "data_type": "raw",
                "n_channels": len(raw.ch_names),
                "sfreq": float(raw.info["sfreq"]),
                "ch_names": list(raw.ch_names),
            }
        except Exception as exc:
            logger.warning("probe_eeg_dataset: could not read %s: %s", data_path, exc)
            return None

    try:
        raw = mne.io.read_raw(data_path, preload=False, verbose=False)
        return {
            "data_type": "raw",
            "n_channels": len(raw.ch_names),
            "sfreq": float(raw.info["sfreq"]),
            "ch_names": list(raw.ch_names),
        }
    except Exception as exc:
        logger.warning("probe_eeg_dataset: could not read %s: %s", data_path, exc)
        return None
