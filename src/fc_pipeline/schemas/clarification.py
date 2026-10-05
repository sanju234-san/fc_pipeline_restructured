"""Single authoritative HITL clarification payload.

Every clarification (Supervisor axis halt, Query-Transformer contradiction,
metric error, ...) is described by ONE dict::

    {
        "axis": "channels" | "frequency_band" | "condition" | "query" | ...,
        "kind": "<legacy clarification_kind string>",
        "question": "<text shown to the user>",
        "options": [{"name": ..., "value": ..., "label": ...}, ...],
        "allow_manual": True,
    }

It is stored in ``GraphState["clarification"]``, emitted in the LangGraph
``interrupt()`` payload, and read by the Chainlit renderer.  The legacy flat
fields (``clarification_question`` / ``clarification_kind`` /
``clarification_options``) are kept for compatibility and are always *derived*
from this payload, never authored independently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

KIND_TO_AXIS: Dict[str, str] = {
    "frequency_band": "frequency_band",
    "channel_selection": "channels",
    "condition": "condition",
    "metric_selection": "metrics",
    "query_contradiction": "query",
    "max_iterations": "unspecified",
    "validation_error": "validation",
    "clarification": "unspecified",
}
AXIS_TO_KIND: Dict[str, str] = {
    "frequency_band": "frequency_band",
    "channels": "channel_selection",
    "condition": "condition",
}
# Kinds whose reply is deterministically validated/persisted as a Supervisor axis.
AXIS_KINDS = frozenset(AXIS_TO_KIND.values())


def _clean_options(options: Optional[List[Mapping[str, Any]]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    seen = set()
    for i, opt in enumerate(options or []):
        if not isinstance(opt, Mapping):
            continue
        value = str(opt.get("value", opt.get("label", "")) or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append({
            "name": str(opt.get("name") or f"option_{i}"),
            "value": value,
            "label": str(opt.get("label") or value),
        })
    return cleaned


def build_clarification(
    kind: str,
    question: str,
    options: Optional[List[Mapping[str, Any]]] = None,
    axis: Optional[str] = None,
) -> Dict[str, Any]:
    """Create the authoritative clarification payload (manual entry always allowed)."""
    return {
        "axis": axis or KIND_TO_AXIS.get(kind, "unspecified"),
        "kind": kind,
        "question": question,
        "options": _clean_options(options),
        "allow_manual": True,
    }


def legacy_fields(clarification: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Flat legacy GraphState fields derived from the authoritative payload."""
    if not clarification:
        return {
            "clarification": None,
            "clarification_question": None,
            "clarification_kind": None,
            "clarification_options": [],
        }
    return {
        "clarification": dict(clarification),
        "clarification_question": clarification.get("question"),
        "clarification_kind": clarification.get("kind"),
        "clarification_options": list(clarification.get("options") or []),
    }


def get_clarification(state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the authoritative clarification for ``state``.

    Prefers ``state["clarification"]``; otherwise synthesises one from the legacy
    flat fields (older checkpoints, Query-Transformer nodes).  Returns ``None``
    only when there is genuinely nothing to ask.  ``allow_manual`` is always True.
    """
    payload = state.get("clarification")
    if isinstance(payload, Mapping) and payload.get("question"):
        merged = dict(payload)
    elif state.get("clarification_question"):
        kind = str(state.get("clarification_kind") or "clarification")
        merged = build_clarification(
            kind,
            str(state["clarification_question"]),
            state.get("clarification_options"),
        )
    else:
        return None
    merged["options"] = _clean_options(merged.get("options"))
    merged["allow_manual"] = True
    merged.setdefault("kind", "clarification")
    merged.setdefault("axis", KIND_TO_AXIS.get(merged["kind"], "unspecified"))
    return merged
