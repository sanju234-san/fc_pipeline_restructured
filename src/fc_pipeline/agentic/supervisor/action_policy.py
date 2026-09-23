"""Pre-action decision layer (Claude Agent SDK ``canUseTool`` / ``PreToolUse`` pattern).

A single function, :func:`evaluate_action`, runs *before* a sensitive action and
returns one of four decisions:

  - ``allow``                      proceed unchanged
  - ``deny``                       refuse, with a reason (no human involved)
  - ``ask_human``                  pause for a human via the existing Gate 1
                                   ``interrupt()`` -> Chainlit -> ``Command(resume=...)``
                                   flow; the returned ``decision_context`` tells the
                                   gate what it is being asked to review
  - ``allow_with_modified_input``  proceed, but with ``updated_input`` instead

The policy only pauses a human when it says to: read-only / reversible actions
auto-resolve to ``allow``; irreversible actions (Gate 1 manifest approval) and
NeMo Guardrails rail failures resolve to ``ask_human``.

Human response vocabulary is the one Gate 1 already speaks: ``approve`` /
``reject`` resume the graph; ``request_changes`` (the "respond" case) and
per-row ``edit`` are handled by the Chainlit handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class DecisionType(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK_HUMAN = "ask_human"
    ALLOW_WITH_MODIFIED_INPUT = "allow_with_modified_input"


@dataclass(frozen=True)
class Decision:
    """Result of :func:`evaluate_action` (mirrors the SDK's permission result)."""

    behavior: DecisionType
    reason: str = ""
    updated_input: Optional[Dict[str, Any]] = None
    # Populated only for ASK_HUMAN; carried in GraphState and in the interrupt payload.
    decision_context: Optional[Dict[str, Any]] = None


# --- Decision-context kinds (what the gate is being asked to review) ---------
KIND_MANIFEST_REVIEW = "manifest_review"
KIND_INPUT_RAIL = "input_rail"
KIND_OUTPUT_RAIL = "output_rail"
RAIL_CONTEXT_KINDS = frozenset({KIND_INPUT_RAIL, KIND_OUTPUT_RAIL})

# --- Risk registry -----------------------------------------------------------
# Keep in sync with SUPERVISOR_TOOLS in agent.py. Unknown actions are NOT
# auto-allowed: they resolve to ask_human (fail-safe).
ACTION_RISK: Dict[str, str] = {
    # Supervisor inspection tools (read-only)
    "get_dataset_info": "read_only",
    "get_dataset_conditions": "read_only",
    "resolve_frequency_band": "read_only",
    "resolve_channel_selection": "read_only",
    # Writes one PNG under outputs/plots — reversible
    "generate_dataset_overview_plot": "reversible_write",
    # Gate 1: approval unlocks downstream execution — irreversible
    "gate_1_manifest_review": "irreversible",
    # NeMo Guardrails rail points
    "nemo_input_rail": "rail",
    "nemo_output_rail": "rail",
}

# What to do when a NeMo rail blocks (or cannot run). ASK_HUMAN per spec; flip a
# rail to DENY here for a hard, non-overridable block.
RAIL_FAILURE_POLICY: Dict[str, DecisionType] = {
    "input": DecisionType.ASK_HUMAN,
    "output": DecisionType.ASK_HUMAN,
}

_RAIL_ALLOWED_RESPONSES = {
    "input": ("approve", "request_changes", "reject"),
    "output": ("approve", "reject"),
}


def _context(
    kind: str,
    action_name: str,
    title: str,
    reason: str,
    details: Optional[Dict[str, Any]] = None,
    allowed_responses: tuple = ("approve", "reject"),
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "action_name": action_name,
        "title": title,
        "reason": reason,
        "details": details or {},
        "allowed_responses": list(allowed_responses),
    }


def _evaluate_rail(
    action_name: str,
    action_input: Mapping[str, Any],
    risk_context: Mapping[str, Any],
) -> Decision:
    rail_type = "input" if action_name == "nemo_input_rail" else "output"
    status = risk_context.get("rail_status", "error")

    if status == "passed":
        return Decision(DecisionType.ALLOW, reason="Guardrail check passed.")

    if status == "modified":
        modified = risk_context.get("modified_text")
        return Decision(
            DecisionType.ALLOW_WITH_MODIFIED_INPUT,
            reason="Guardrail modified the text.",
            updated_input={**dict(action_input), "text": modified},
        )

    # blocked, or the rail itself failed to run (fail closed)
    if status == "blocked":
        if rail_type == "input":
            reason = (
                "The input guardrail flagged this message as a possible "
                "jailbreak / prompt-injection attempt, or as unrelated to EEG analysis."
            )
            title = "Guardrail Review — Input Screening"
        else:
            reason = (
                "The output guardrail could not verify this response against the "
                "dataset tool results (possible unsupported or fabricated claim)."
            )
            title = "Guardrail Review — Output Screening"
    else:
        reason = (
            f"The {rail_type} guardrail check could not run "
            f"({risk_context.get('error') or 'unknown error'}); failing closed."
        )
        title = f"Guardrail Review — {rail_type.capitalize()} Screening (check unavailable)"

    if RAIL_FAILURE_POLICY.get(rail_type, DecisionType.ASK_HUMAN) == DecisionType.DENY:
        return Decision(DecisionType.DENY, reason=reason)

    ctx = _context(
        KIND_INPUT_RAIL if rail_type == "input" else KIND_OUTPUT_RAIL,
        action_name,
        title,
        reason,
        details={
            "rail": risk_context.get("rail"),
            "rail_status": status,
            "flagged_text": risk_context.get("flagged_text"),
            "run_id": risk_context.get("run_id"),
        },
        allowed_responses=_RAIL_ALLOWED_RESPONSES[rail_type],
    )
    return Decision(DecisionType.ASK_HUMAN, reason=reason, decision_context=ctx)


def evaluate_action(
    action_name: str,
    action_input: Optional[Mapping[str, Any]] = None,
    risk_context: Optional[Mapping[str, Any]] = None,
) -> Decision:
    """Decide whether ``action_name`` may run, before it runs.

    Args:
        action_name: Tool name or pseudo-action (``gate_1_manifest_review``,
            ``nemo_input_rail``, ``nemo_output_rail``).
        action_input: The action's arguments (tool args, or ``{"text": ...}`` for rails).
        risk_context: Extra facts for the policy (``run_id``; for rails:
            ``rail_status`` in {passed, modified, blocked, error}, ``rail``,
            ``flagged_text``, ``modified_text``, ``error``).
    """
    action_input = action_input or {}
    risk_context = risk_context or {}
    tier = ACTION_RISK.get(action_name)

    if tier in ("read_only", "reversible_write"):
        return Decision(DecisionType.ALLOW, reason=f"{tier} action.")

    if tier == "irreversible":
        ctx = _context(
            KIND_MANIFEST_REVIEW,
            action_name,
            "Gate 1 — Plan & Parameter Preflight Review",
            "Approving this plan unlocks downstream execution and cannot be undone.",
            details={"run_id": risk_context.get("run_id")},
            allowed_responses=("approve", "edit", "request_changes", "reject"),
        )
        return Decision(
            DecisionType.ASK_HUMAN,
            reason="Irreversible action requires human approval.",
            decision_context=ctx,
        )

    if tier == "rail":
        return _evaluate_rail(action_name, action_input, risk_context)

    return Decision(
        DecisionType.ASK_HUMAN,
        reason=f"Action '{action_name}' is not in the risk registry.",
        decision_context=_context(
            "unknown_action",
            action_name,
            "Unrecognised Action",
            f"Action '{action_name}' is not in the risk registry.",
            details={"run_id": risk_context.get("run_id")},
        ),
    )
