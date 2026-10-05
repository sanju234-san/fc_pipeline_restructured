"""Deterministic validation + persistence of a HITL clarification reply.

A reply to a Supervisor axis clarification (frequency band / channels /
condition) is validated against the *actual dataset* with the same
deterministic tools the Supervisor uses, and the result is written into the
resolved-axis GraphState fields.  This means an already-answered axis can never
depend on the LLM re-deriving it after a checkpoint resume.

    validate -> normalize -> persist (resolved_* fields)

Failures return an error string; nothing is persisted in that case so the axis
stays unresolved and is asked again with the error explained.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from fc_pipeline.agentic.supervisor.tools.channel_selection import (
    normalize_channel_label,
    resolve_channel_selection,
)
from fc_pipeline.agentic.supervisor.tools.dataset_conditions import get_dataset_conditions
from fc_pipeline.agentic.supervisor.tools.frequency_band import resolve_frequency_band
from fc_pipeline.schemas.clarification import get_clarification


def _norm_condition(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def dataset_conditions_for_state(state: Mapping[str, Any]) -> Dict[str, int]:
    """Actual event/condition labels in the loaded dataset ({} if unreadable)."""
    path = state.get("raw_data_path")
    if not path:
        return {}
    try:
        obs = get_dataset_conditions.invoke({"data_path": str(path)})
    except Exception:
        return {}
    if isinstance(obs, dict) and not obs.get("error"):
        return dict(obs.get("conditions") or {})
    return {}


def apply_clarification_reply(
    state: Mapping[str, Any], kind: Optional[str], reply: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Return ``(state_update, error)`` for a reply to a clarification of ``kind``.

    ``state_update`` is empty (and ``error`` None) for kinds that are not
    Supervisor scientific axes, or when the dataset metadata needed to validate
    is unavailable (the Supervisor then resolves the axis itself).
    """
    reply = (reply or "").strip()
    if not reply:
        return {}, None

    if kind == "frequency_band":
        sfreq = state.get("dataset_sfreq")
        if not sfreq:
            return {}, None
        obs = resolve_frequency_band.invoke({
            "band_name_or_range": reply,
            "sfreq": float(sfreq),
            "duration_seconds": state.get("dataset_duration_seconds"),
        })
        if obs.get("error"):
            return {}, str(obs["error"])
        return {"resolved_frequency_band_info": obs}, None

    if kind == "channel_selection":
        available: List[str] = list(state.get("dataset_available_channels") or [])
        if not available:
            return {}, None
        obs = resolve_channel_selection.invoke({
            "requested_channels_or_region": reply,
            "available_channels": available,
        })
        if obs.get("error"):
            return {}, str(obs["error"])
        resolved = obs.get("resolved_channels") or []
        if len(resolved) < 2:
            return {}, (
                "Functional connectivity needs at least two channels; "
                f"'{reply}' resolved to {len(resolved)}."
            )
        # Every resolved label must be a real dataset label (raw label kept for MNE).
        raw_by_norm = {normalize_channel_label(c): c for c in available}
        for ch in resolved:
            if normalize_channel_label(ch) not in raw_by_norm:
                return {}, f"Channel '{ch}' is not present in the dataset."
        return {"resolved_channel_info": obs}, None

    if kind == "condition":
        valid = dataset_conditions_for_state(state)
        if not valid:
            clar = get_clarification(state) or {}
            valid = {o["value"]: 0 for o in clar.get("options", [])}
            valid.update({c: 0 for c in (state.get("condition_candidates") or [])})
        if not valid:
            return {}, None
        by_norm = {_norm_condition(c): c for c in valid}
        match = by_norm.get(_norm_condition(reply))
        if match is None:
            shown = ", ".join(sorted(valid)[:12])
            return {}, f"'{reply}' is not a condition in this dataset. Available: {shown}."
        return {"resolved_condition_value": match}, None

    return {}, None
