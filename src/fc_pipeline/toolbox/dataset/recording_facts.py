"""Toolbox helper: get_recording_facts (deterministic, read-only, NOT an LLM tool).

Reads only the header and annotations of the stored raw recording (no signal
data is loaded) and returns plain facts about the *file*: format, channel
count, sampling rate, duration, filter settings recorded in the header, and
the annotation labels with how many annotations each has and how long they
last in total.

Used by the post-run follow-up layer to answer "what is this dataset / this
recording?" with facts that come from the file itself. It is a plain function
rather than a ``@tool`` so it can never be bound to an LLM; it is called by
application code only.

Privacy: subject information, measurement date and experimenter fields are
never read or returned.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _error_facts(data_path: Any, message: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": message,
        "file_format": None,
        "n_channels": 0,
        "channel_names": [],
        "sfreq": 0.0,
        "duration_seconds": 0.0,
        "reference": "unknown",
        "highpass_hz": None,
        "lowpass_hz": None,
        "annotation_labels": {},
        "n_annotation_labels": 0,
        "single_condition": False,
    }


def get_recording_facts(data_path: Optional[str]) -> Dict[str, Any]:
    """Return header-level facts about the recording at ``data_path``.

    Never raises: failures come back as ``{"ok": False, "error": ...}``.
    """
    if not data_path:
        return _error_facts(data_path, "No recording path is stored for this run.")
    path = Path(str(data_path))
    if not path.exists():
        return _error_facts(data_path, "The stored recording file is no longer available.")

    try:
        import mne

        raw = mne.io.read_raw(str(path), preload=False, verbose=False)
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        duration = float(n_times / sfreq) if sfreq > 0 else 0.0

        try:
            types = raw.get_channel_types()
            eeg = [ch for ch, t in zip(raw.ch_names, types) if t == "eeg"]
            channels: List[str] = eeg if eeg else list(raw.ch_names)
        except Exception:
            channels = list(raw.ch_names)

        custom_ref = raw.info.get("custom_ref_applied")
        if custom_ref == 2:
            reference = "average"
        elif custom_ref == 1 or custom_ref is True:
            reference = "custom"
        else:
            reference = "unreferenced"

        labels: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
        ann = raw.annotations
        if ann is not None and len(ann) > 0:
            for desc, dur in zip(ann.description, ann.duration):
                entry = labels.setdefault(str(desc), {"count": 0, "total_seconds": 0.0})
                entry["count"] += 1
                entry["total_seconds"] += float(dur)
        annotation_labels = {
            k: {"count": int(v["count"]), "total_seconds": round(float(v["total_seconds"]), 3)}
            for k, v in labels.items()
        }

        hp = raw.info.get("highpass")
        lp = raw.info.get("lowpass")
        return {
            "ok": True,
            "error": None,
            "file_format": path.suffix.lower().lstrip(".") or None,
            "n_channels": len(channels),
            "channel_names": channels,
            "sfreq": sfreq,
            "duration_seconds": round(duration, 3),
            "reference": reference,
            "highpass_hz": float(hp) if hp is not None else None,
            "lowpass_hz": float(lp) if lp is not None else None,
            "annotation_labels": annotation_labels,
            "n_annotation_labels": len(annotation_labels),
            "single_condition": len(annotation_labels) == 1,
        }
    except Exception as exc:  # unreadable / unsupported file
        return _error_facts(data_path, f"Could not read the recording header: {type(exc).__name__}: {exc}")


def format_recording_facts(facts: Dict[str, Any]) -> str:
    """One compact, plain-language block of the file facts (no interpretation)."""
    if not facts.get("ok"):
        return f"The recording file could not be inspected ({facts.get('error')})."
    parts = [
        f"{facts['n_channels']} EEG channels",
        f"{facts['sfreq']:g} Hz sampling rate",
        f"{facts['duration_seconds']:g} s duration",
    ]
    if facts.get("file_format"):
        parts.append(f"file format: {facts['file_format']}")
    lines = ["- " + "; ".join(parts) + "."]
    hp, lp = facts.get("highpass_hz"), facts.get("lowpass_hz")
    if hp is not None or lp is not None:
        lines.append(
            "- Header filter settings: "
            f"high-pass {hp:g} Hz, low-pass {lp:g} Hz." if hp is not None and lp is not None
            else "- Header filter settings are partially recorded."
        )
    labels = facts.get("annotation_labels") or {}
    if labels:
        shown = ", ".join(
            f"{name} ({v['count']} annotation{'s' if v['count'] != 1 else ''}, {v['total_seconds']:g} s)"
            for name, v in labels.items()
        )
        lines.append(f"- Annotation labels in the file: {shown}.")
        if facts.get("single_condition"):
            lines.append(
                "- Only one annotation label is present, so comparing conditions "
                "within this single recording is not possible."
            )
    else:
        lines.append("- The file contains no annotation labels.")
    return "\n".join(lines)
