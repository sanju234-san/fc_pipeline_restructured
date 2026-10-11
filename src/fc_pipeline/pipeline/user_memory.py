"""User preference memory (M5) — the Deep Agents "Auto memory" layer.

Auto-memory heuristic (mirrors Claude Code / Deep Agents auto memory docs):
  * Each time a HITL axis is resolved (frequency band / channels / condition /
    reference / report_style / metric_set), increment a counter for the
    (user_id, axis, normalised_value) tuple.
  * When the same (axis, value) repeats **3 times** AND it is not yet in the
    confirmed-defaults snapshot, emit a `PreferenceProposal` so the UI can ask:
    "You chose VALUE for AXIS 3 times. Remember as default? [Yes / No / Ask later]"
  * If the user says "Yes" (or writes "remember this"), persist it into
    ``{user_id}_preferences.json`` as a CONFIRMED default.
  * Confirmed defaults are loaded on session start and used to **PRE-POPULATE**
    GraphState's ``resolved_{axis}_info`` fields — so the Supervisor skips
    that HITL axis entirely for that user.

Storage layout (under ``repo_root / user_memory /``, added to .gitignore if
someone runs the pipeline for the first time):

  user_memory/
    {user_id}_history.jsonl      one JSON object per HITL resolution event
    {user_id}_preferences.json   current {confirmed_defaults: {...}, ...}
    .gitignore                   created lazily so personal pref files are
                                 never checked into a shared team repo.

Offline-testable: no LLM, no network, pure file IO + deterministic heuristics.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fc_pipeline.pipeline.memory import _current_user_id, find_repo_root

logger = logging.getLogger(__name__)


# How many repeated identical choices for an axis before we propose remembering.
DEFAULT_REPEAT_THRESHOLD = 3

# The axis keys we track with auto-memory.  New Supervisor axes must be added
# here for "remember as default" to work.  Each entry maps:
#   axis key -> (normalise_fn | None, human_label).  Normalise collapses
#   equivalent values so "CAR" and "average" (and "average (CAR)") all count
#   as the same pick.
_AXIS_HUMAN_LABEL: Dict[str, str] = {
    "frequency_band": "frequency band",
    "channels": "channel selection",
    "condition": "condition / event label",
    "reference": "reference choice",
    "report_style": "report style",
    "metric_set": "metric selection",
}


def _normalize_value(axis: str, value: Any) -> str:
    """Deterministic normalisation so equivalent picks share a counter."""
    if isinstance(value, (list, tuple, set)):
        try:
            ordered = sorted({str(v).strip().lower() for v in value if str(v).strip()})
        except Exception:
            ordered = [str(v).strip().lower() for v in value]
        return "|".join(ordered)
    s = str(value or "").strip()
    low = s.lower()
    if axis == "reference":
        if "car" in low or "average" in low or "avg" in low:
            return "average (car)"
        if "cz" in low:
            return "cz"
        if "linked" in low or "mastoid" in low:
            return "linked mastoids"
    if axis == "frequency_band":
        # e.g. "Theta (4–8 Hz)" / "theta" / "4-8hz" -> canonical name if known.
        import re as _re
        known = {
            "delta": "delta (1-4 Hz)",
            "theta": "theta (4-8 Hz)",
            "alpha": "alpha (8-12 Hz)",
            "beta": "beta (12-30 Hz)",
            "gamma": "gamma (30-45 Hz)",
            "wideband": "wideband (1-45 Hz)",
        }
        for name, canonical in known.items():
            if name in low:
                return canonical
        # Hz range form: strip whitespace + use dashes.
        nums = _re.findall(r"\d+(?:\.\d+)?", s)
        if len(nums) >= 2:
            return f"custom {nums[0]}-{nums[1]} Hz"
    if axis == "channels":
        # Channel lists come as normalised strings, so split + sort + rejoin.
        bits = sorted({b.strip() for b in _re.split(r"[^A-Za-z0-9_\-]+", low) if b.strip()})
        if bits:
            return "|".join(bits)
    return s.lower()


# ---------------------------------------------------------------------------
# Persistence primitives
# ---------------------------------------------------------------------------

def _user_memory_dir() -> Path:
    override = os.getenv("FC_PIPELINE_USER_MEMORY_DIR")
    if override:
        # Test / user override: treat <override>/user_memory/ as the storage
        # directory.  Matches the fixture setup in tests/unit/test_user_memory_m5.py
        target_dir = "user_memory"
        out = Path(override) / target_dir
    else:
        root = find_repo_root()
        if root is None:
            fallback = Path(tempfile.gettempdir())
            out = Path(fallback) / "fc_pipeline_user_memory"
        else:
            out = root / "user_memory"
    out.mkdir(parents=True, exist_ok=True)
    _ensure_gitignore(out)
    return out


def _ensure_gitignore(d: Path) -> None:
    gi = d / ".gitignore"
    if gi.exists():
        return
    try:
        gi.write_text(
            "# Auto-generated by fc_pipeline.pipeline.user_memory.\n"
            "# Personal preference files should never be committed to a shared\n"
            "# team repo (they contain per-user defaults / identifiers).\n"
            "*_history.jsonl\n"
            "*_preferences.json\n"
            "*_instructions.md\n"
            "!.gitignore\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _history_path(user_id: str) -> Path:
    return _user_memory_dir() / f"{user_id}_history.jsonl"


def _pref_path(user_id: str) -> Path:
    return _user_memory_dir() / f"{user_id}_preferences.json"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass
class PreferenceEvent:
    """One HITL resolution row, appended to history JSONL."""

    axis: str
    raw_value: object
    normalised_value: str
    run_id: str
    timestamp_iso: str
    source: str = "hitl_resolution"   # hitl_resolution | explicit | inferred


@dataclass
class PreferenceProposal:
    """Returned by ``record_pick`` when a (axis, value) hits the threshold.

    The UI layer is responsible for asking the user; on "yes", call
    :func:`confirm_default` / :func:`reject_proposal` accordingly.
    """

    user_id: str
    axis: str
    normalised_value: str
    count: int
    threshold: int
    raw_value_repr: str
    axis_label: str

    def question_text(self) -> str:
        return (
            f"You chose **{self.raw_value_repr}** for "
            f"**{self.axis_label}** {self.count} times.  Remember as your "
            f"default for future runs?"
        )


@dataclass
class PreferenceStore:
    """Runtime snapshot loaded from disk.  Passed into pipeline startup."""

    user_id: str
    confirmed_defaults: Dict[str, Any] = field(default_factory=dict)
    rejected_proposals: Dict[str, List[str]] = field(default_factory=dict)
    """axis -> list of normalised values the user said "No / ask later" to."""


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def load_preference_store(user_id: Optional[str] = None) -> PreferenceStore:
    """Load the current confirmed-defaults snapshot (or empty defaults)."""
    user_id = user_id or _current_user_id()
    path = _pref_path(user_id)
    if not path.exists():
        return PreferenceStore(user_id=user_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("user_memory: could not parse %s: %s", path, exc)
        return PreferenceStore(user_id=user_id)
    if not isinstance(raw, dict):
        return PreferenceStore(user_id=user_id)
    confirmed = dict(raw.get("confirmed_defaults") or {})
    rejected_raw = raw.get("rejected_proposals") or {}
    rejected: Dict[str, List[str]] = {}
    if isinstance(rejected_raw, dict):
        for k, v in rejected_raw.items():
            if isinstance(v, list):
                rejected[str(k)] = [str(x) for x in v]
    return PreferenceStore(
        user_id=user_id,
        confirmed_defaults=confirmed,
        rejected_proposals=rejected,
    )


def _count_axis_value(user_id: str, axis: str, normalised_value: str) -> int:
    hp = _history_path(user_id)
    if not hp.exists():
        return 0
    c = 0
    try:
        with hp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    obj.get("axis") == axis
                    and obj.get("normalised_value") == normalised_value
                ):
                    c += 1
    except OSError as exc:
        logger.warning("user_memory: count failed for %s: %s", hp, exc)
        return 0
    return c


def _append_history(event: PreferenceEvent, user_id: str) -> None:
    hp = _history_path(user_id)
    try:
        with hp.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("user_memory: append to %s failed: %s", hp, exc)


def _is_rejected(store: PreferenceStore, axis: str, normalised_value: str) -> bool:
    return normalised_value in (store.rejected_proposals.get(axis) or [])


def record_pick(
    user_id: Optional[str],
    axis: str,
    raw_value: Any,
    run_id: str,
    *,
    source: str = "hitl_resolution",
    threshold: int = DEFAULT_REPEAT_THRESHOLD,
) -> Optional[PreferenceProposal]:
    """Append a HITL-resolution event and MAYBE return a proposal to confirm.

    Returns ``None`` when:
      * axis is not in the tracked set
      * value is empty
      * the value already has a confirmed default in the store
      * the user already rejected this proposal before
      * the repeat count has not yet hit ``threshold`` (default 3)
    """
    if axis not in _AXIS_HUMAN_LABEL:
        return None
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        return None
    if isinstance(raw_value, (list, tuple, set)) and not any(raw_value):
        return None

    user_id = user_id or _current_user_id()
    store = load_preference_store(user_id)

    # Already confirmed for this axis? Then no proposal, just history append.
    if axis in store.confirmed_defaults:
        return None

    norm = _normalize_value(axis, raw_value)
    if not norm:
        return None

    # User explicitly said "don't ask again" for this (axis, value)? Skip.
    if _is_rejected(store, axis, norm):
        return None

    event = PreferenceEvent(
        axis=axis,
        raw_value=(
            list(raw_value) if isinstance(raw_value, (list, tuple, set)) else raw_value
        ),
        normalised_value=norm,
        run_id=run_id,
        timestamp_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=source,
    )
    _append_history(event, user_id)
    count = _count_axis_value(user_id, axis, norm)
    if count >= threshold:
        label = _AXIS_HUMAN_LABEL[axis]
        if isinstance(raw_value, (list, tuple, set)):
            pretty = ", ".join(f"`{v}`" for v in list(raw_value)[:8])
            if len(list(raw_value)) > 8:
                pretty += "…"
        else:
            pretty = f"`{raw_value}`"
        return PreferenceProposal(
            user_id=user_id,
            axis=axis,
            normalised_value=norm,
            count=count,
            threshold=threshold,
            raw_value_repr=pretty,
            axis_label=label,
        )
    return None


def _write_store(user_id: str, store: PreferenceStore) -> None:
    path = _pref_path(user_id)
    payload = {
        "user_id": user_id,
        "confirmed_defaults": store.confirmed_defaults,
        "rejected_proposals": store.rejected_proposals,
        "updated_at_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("user_memory: write to %s failed: %s", path, exc)


def confirm_default(user_id: Optional[str], axis: str, value: Any) -> Dict[str, Any]:
    """Persist (axis, value) as a CONFIRMED default for this user.

    ``value`` should be the already-validated pipeline representation (for
    frequency band: a dict with name/fmin/fmax; for channels: a list of
    canonical channel labels from the dataset; for condition: the exact
    string event label; etc.).  Returns the new confirmed-defaults snapshot.
    """
    if axis not in _AXIS_HUMAN_LABEL:
        raise ValueError(f"Unknown preference axis: {axis}")
    user_id = user_id or _current_user_id()
    store = load_preference_store(user_id)
    store.confirmed_defaults[axis] = (
        list(value) if isinstance(value, (tuple, set)) else value
    )
    _write_store(user_id, store)
    return dict(store.confirmed_defaults)


def reject_proposal(
    user_id: Optional[str],
    axis: str,
    normalised_value: str,
    *,
    forever: bool = False,
) -> None:
    """Call when the user answers "No / Ask later" to a proposal.

    ``forever=True`` means "don't ask me again for this exact (axis, value)"
    — otherwise the counter resets to threshold-1 via a history-only skip.
    """
    user_id = user_id or _current_user_id()
    store = load_preference_store(user_id)
    if forever:
        bucket = store.rejected_proposals.setdefault(axis, [])
        if normalised_value not in bucket:
            bucket.append(normalised_value)
        _write_store(user_id, store)


def clear_default(user_id: Optional[str], axis: str) -> None:
    """Forget a confirmed default (e.g. user says "stop remembering X")."""
    user_id = user_id or _current_user_id()
    store = load_preference_store(user_id)
    store.confirmed_defaults.pop(axis, None)
    _write_store(user_id, store)


def remember_explicit(user_id: Optional[str], text: str, run_id: str) -> Optional[str]:
    """Parse an explicit user command like "remember that I prefer alpha band".

    Best-effort regex-based parser; returns a human-readable confirmation
    string when something was stored, or ``None`` when nothing was matched.
    Real confirmations always go through the HITL proposal path — this is a
    convenience for the chat input box.
    """
    if not text:
        return None
    user_id = user_id or _current_user_id()
    low = text.lower().strip()

    band_kw = r"(?:band|freq|frequency)"
    band_names = r"(delta|theta|alpha|beta|gamma|wideband)"
    # Name-then-band OR band-then-name.
    match = re.search(
        rf"(?:remember|save)\b.*?(?:(?:{band_names}\b.*?{band_kw}\b)|(?:{band_kw}\b.*?{band_names}\b))",
        low,
    )
    if match:
        name = next((g for g in match.groups() if g), None)
        if name is None:
            name = match.group(1)
        ranges = {
            "delta": {"name": "delta", "fmin": 1.0, "fmax": 4.0},
            "theta": {"name": "theta", "fmin": 4.0, "fmax": 8.0},
            "alpha": {"name": "alpha", "fmin": 8.0, "fmax": 12.0},
            "beta": {"name": "beta", "fmin": 12.0, "fmax": 30.0},
            "gamma": {"name": "gamma", "fmin": 30.0, "fmax": 45.0},
            "wideband": {"name": "wideband", "fmin": 1.0, "fmax": 45.0},
        }
        confirm_default(user_id, "frequency_band", ranges[name])
        return (
            f"✅ Remembered **{name} band** as your default frequency band."
        )
    if re.search(r"(?:remember|save)\b.*?(?:car|average ?ref)", low):
        confirm_default(user_id, "reference", "average (CAR)")
        return "✅ Remembered **average (CAR)** as your default reference."
    if re.search(r"(?:forget|clear|stop remember)", low):
        for a in ("frequency_band", "reference", "report_style", "metric_set", "condition", "channels"):
            if a.replace("_", " ") in low or a in low:
                clear_default(user_id, a)
                return f"✅ Cleared remembered default for **{_AXIS_HUMAN_LABEL.get(a, a)}**."
    return None


# ---------------------------------------------------------------------------
# Startup integration: GraphState default pre-population
# ---------------------------------------------------------------------------

def apply_confirmed_defaults_to_state(
    state: Dict[str, Any],
    store: Optional[PreferenceStore] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a GraphState update dict pre-populating confirmed defaults.

    For each confirmed axis that maps to a GraphState resolution field, set
    that field so the Supervisor skips the corresponding HITL clarification.
    """
    if store is None:
        store = load_preference_store(user_id)
    updates: Dict[str, Any] = {}

    freq = store.confirmed_defaults.get("frequency_band")
    if freq and state.get("resolved_frequency_band_info") is None:
        if isinstance(freq, dict) and "name" in freq and "fmin" in freq and "fmax" in freq:
            updates["resolved_frequency_band_info"] = {
                "name": freq["name"],
                "fmin": float(freq["fmin"]),
                "fmax": float(freq["fmax"]),
                "source": "user_default",
            }

    ch = store.confirmed_defaults.get("channels")
    if ch and state.get("resolved_channel_info") is None and isinstance(ch, (list, tuple)) and len(ch) >= 2:
        updates["resolved_channel_info"] = {
            "resolved_channels": list(ch),
            "region_label": None,
            "source": "user_default",
        }

    cond = store.confirmed_defaults.get("condition")
    if cond and state.get("resolved_condition_value") is None and isinstance(cond, str) and cond.strip():
        updates["resolved_condition_value"] = cond

    ref = store.confirmed_defaults.get("reference")
    if ref and state.get("dataset_reference") is None and isinstance(ref, str):
        updates["dataset_reference"] = ref

    return updates
