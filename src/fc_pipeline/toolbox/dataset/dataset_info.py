"""Toolbox tool: get_dataset_info (header, channels, sample rate, duration)."""

from typing import Any, Dict, List
import mne
from langchain_core.tools import tool


@tool
def get_dataset_info(data_path: str) -> Dict[str, Any]:
    """Reads the dataset header via MNE and returns recording metadata without loading raw signals into memory.
    
    Returns sfreq, Nyquist limit (sfreq / 2), duration, sample count, and available EEG channel labels.
    Strictly scrubs all patient/subject identifiers, experimenter details, and demographics at the source.
    """
    try:
        raw = mne.io.read_raw(data_path, preload=False, verbose=False)
        sfreq = float(raw.info["sfreq"])
        nyquist = sfreq / 2.0
        n_times = int(raw.n_times)
        duration = float(n_times / sfreq) if sfreq > 0 else 0.0
        
        # Filter strictly to EEG channels if channel types are defined
        try:
            ch_types = raw.get_channel_types()
            eeg_channels = [ch for ch, t in zip(raw.ch_names, ch_types) if t == "eeg"]
            # If no channels are marked as 'eeg', fall back to all channels
            channels: List[str] = eeg_channels if len(eeg_channels) > 0 else list(raw.ch_names)
        except Exception:
            channels = list(raw.ch_names)
        
        # Inspect reference state from header (Section 3 Constraint 5)
        custom_ref = raw.info.get("custom_ref_applied")
        if custom_ref == 2:
            reference_status = "average"
        elif custom_ref == 1 or custom_ref is True:
            reference_status = "custom"
        else:
            reference_status = "unreferenced (proposed: average)"
        
        return {
            "sfreq": sfreq,
            "nyquist": nyquist,
            "duration_seconds": duration,
            "n_times": n_times,
            "available_channels": channels,
            "n_channels": len(channels),
            "reference": reference_status,
            "reference_provenance": "MNE header custom_ref_applied",
            "error": None,
        }
    except Exception as e:
        return {
            "sfreq": 0.0,
            "nyquist": 0.0,
            "duration_seconds": 0.0,
            "n_times": 0,
            "available_channels": [],
            "n_channels": 0,
            "reference": "unknown (proposed: average)",
            "reference_provenance": "unreadable header fallback",
            "error": f"Failed to read dataset header at '{data_path}': {str(e)}",
        }
