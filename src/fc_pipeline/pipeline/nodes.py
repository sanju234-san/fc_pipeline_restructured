"""LangGraph node adapter functions matching Callable[[GraphState], Dict[str, Any]]."""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fc_pipeline.agentic.supervisor.action_policy import (
    Decision,
    DecisionType,
    evaluate_action,
)
from fc_pipeline.agentic.supervisor.agent import supervisor_node
from fc_pipeline.agentic.supervisor.llm_provider import get_supervisor_llm
from fc_pipeline.agentic.supervisor.query_transformer import transform_query
from fc_pipeline.schemas.clarification import build_clarification, legacy_fields
from fc_pipeline.schemas.state import GraphState
from fc_pipeline.pipeline.rail_prefilter import is_obviously_in_scope

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVIDIA NeMo Guardrails — input/output boundary rails
# ---------------------------------------------------------------------------
# Config lives in <project root>/guardrails (RailsConfig.from_path layout).
# Enabled with NEMO_GUARDRAILS_ENABLED=true (see .env.example); off otherwise so
# the pipeline behaves exactly as before when the flag is unset. Rails reuse the
# Supervisor LLM, so no separate endpoint configuration exists.
_GUARDRAILS_DIR = Path(__file__).resolve().parents[3] / "guardrails"
_rails = None
_rails_lock = threading.Lock()
_EVIDENCE_CHAR_LIMIT = 8000


def _rails_enabled() -> bool:
    return os.getenv("NEMO_GUARDRAILS_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _get_rails():
    """Lazily builds (once) the LLMRails instance; nemoguardrails is imported on demand."""
    global _rails
    with _rails_lock:
        if _rails is None:
            from nemoguardrails import LLMRails, RailsConfig

            llm = get_supervisor_llm()
            try:
                from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter

                llm = LangChainLLMAdapter(llm)
            except ImportError:  # older nemoguardrails wraps LangChain models itself
                pass
            _rails = LLMRails(RailsConfig.from_path(str(_GUARDRAILS_DIR)), llm=llm)
        return _rails


def _check_rail(rail_type: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs one NeMo rail point. Never raises: infra failures become status='error'."""
    try:
        from nemoguardrails.rails.llm.options import RailType

        rt = RailType.INPUT if rail_type == "input" else RailType.OUTPUT
        result = _get_rails().check(messages, rail_types=[rt])
        logger.info(
            "NeMo %s rail result: status=%s rail=%s", rail_type, result.status.value, result.rail
        )
        return {
            "status": result.status.value,  # passed | modified | blocked
            "content": result.content,
            "rail": result.rail,
            "error": None,
        }
    except Exception as exc:  # fail closed — the policy turns this into ask_human
        logger.warning("NeMo %s rail could not run: %s: %s", rail_type, type(exc).__name__, exc)
        return {"status": "error", "content": "", "rail": None, "error": f"{type(exc).__name__}: {exc}"}


def _screen_with_rails(
    rail_type: str,
    text: str,
    *,
    user_text: str = "",
    evidence: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> Optional[Decision]:
    """Runs a rail and routes the outcome through evaluate_action().

    Returns None when rails are disabled or there is nothing to screen.
    """
    if not _rails_enabled() or not text:
        return None

    if rail_type == "input":
        # Plainly in-scope, marker-free EEG requests skip the LLM classifier,
        # which small models answer unreliably (see rail_prefilter.py).
        if is_obviously_in_scope(text):
            logger.info("NeMo input rail skipped by deterministic pre-filter (in-scope EEG request).")
            return None
        messages: List[Dict[str, Any]] = [{"role": "user", "content": text}]
    else:
        evidence_text = evidence if isinstance(evidence, str) else json.dumps(evidence, default=str)
        messages = [
            # Supplies the context variables NeMo's built-in `self check facts` flow reads.
            {
                "role": "context",
                "content": {"check_facts": True, "relevant_chunks": evidence_text[:_EVIDENCE_CHAR_LIMIT]},
            },
            {"role": "user", "content": user_text or ""},
            {"role": "assistant", "content": text},
        ]

    if rail_type == "input":
        from fc_pipeline.research.dspy_rail import check_input_with_dspy, dspy_rail_enabled

        result = check_input_with_dspy(text) if dspy_rail_enabled() else _check_rail(rail_type, messages)
    else:
        result = _check_rail(rail_type, messages)
    return evaluate_action(
        f"nemo_{rail_type}_rail",
        {"text": text},
        {
            "rail_status": result["status"],
            "rail": result["rail"],
            "error": result["error"],
            "modified_text": result["content"],
            "flagged_text": text,
            "run_id": run_id,
        },
    )


def apply_input_rail(state: GraphState, text: str) -> Optional[Dict[str, Any]]:
    """Input rail, called at the top of the Query Transformer node.

    Returns None to continue into the Query Transformer, or a state update that
    ends this node early (ask_human -> Gate; deny -> refusal).
    """
    if state.get("input_rail_cleared"):
        return None  # a human already reviewed and cleared this exact input

    decision = _screen_with_rails("input", text, run_id=state.get("run_id"))
    if decision is None or decision.behavior in (
        DecisionType.ALLOW,
        DecisionType.ALLOW_WITH_MODIFIED_INPUT,  # input self-check never rewrites; treat as pass
    ):
        return None

    halted = {
        "plan": None,
        "clarification_question": None,
        "preflight_confirmed": False,
        "gate_1_approved": False,
    }
    if decision.behavior == DecisionType.ASK_HUMAN:
        return {
            **halted,
            "informational_response": None,
            "informational_artifacts": None,
            "decision_context": decision.decision_context,
        }
    return {  # DENY
        **halted,
        "informational_response": "🛡️ Guardrail: Cannot process this query.",
        "informational_artifacts": None,
        "decision_context": None,
    }


def apply_output_rail(state: GraphState) -> Dict[str, Any]:
    """Output rail, called from the informational_complete node.

    Screens the LLM-authored `informational_response` against the tool
    observations in `informational_artifacts` (its only grounding). Canned
    messages (e.g. out-of-scope replies) have no artifacts and are skipped.
    """
    text = state.get("informational_response")
    artifacts = state.get("informational_artifacts")
    if not text or not artifacts:
        return {}

    # Do not run factual grounding check on clarification or parameter prompt text
    clarif_words = ["clarif", "specify", "missing", "please specify", "lacks", "require", "not specified"]
    if any(w in text.lower() for w in clarif_words):
        return {}

    decision = _screen_with_rails(
        "output",
        text,
        user_text=state.get("latest_user_message") or state.get("user_request") or "",
        evidence=artifacts,
        run_id=state.get("run_id"),
    )
    if decision is None or decision.behavior == DecisionType.ALLOW:
        return {}
    if decision.behavior == DecisionType.ALLOW_WITH_MODIFIED_INPUT:
        return {"informational_response": (decision.updated_input or {}).get("text", text)}
    if decision.behavior == DecisionType.ASK_HUMAN:
        return {"decision_context": decision.decision_context}
    return {  # DENY
        "informational_response": "🛡️ Guardrail: Cannot process this query.",
        "decision_context": None,
    }


def supervisor_node_adapter(state: GraphState) -> Dict[str, Any]:
    """LangGraph-compatible wrapper that loads the configured LLM and invokes the Supervisor.

    Reads SUPERVISOR_LLM_ENDPOINT and SUPERVISOR_LLM_MODEL from environment
    to instantiate the provider, then delegates to supervisor_node for ReAct execution.
    """
    llm = get_supervisor_llm()

    # Use explicit run_id if present in state; otherwise derive from data file basename
    run_id = state.get("run_id")
    if not run_id:
        raw_path = state.get("raw_data_path", "unknown")
        # Handle both forward and backslash path separators
        basename = raw_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        run_id = basename.split(".")[0] if basename else "default_run"

    return supervisor_node(state=state, llm=llm, run_id=run_id)


def query_transformer_node_adapter(state: GraphState) -> Dict[str, Any]:
    """LangGraph node adapter for the Query Transformer.

    Pre-processes the incoming request before the Supervisor:
    - Intent classification (eeg_analysis vs out_of_scope)
    - Conversational condensation (merges multi-turn history into standalone query)
    - Contradiction detection on the 3 mandatory scientific axes

    Preserves exact input semantics:
    - accumulated_query comes from state["user_request"]
    - latest_user_message comes from state.get("latest_user_message") or accumulated_query
    """
    accumulated_query = state.get("user_request", "")
    latest_user_message = state.get("latest_user_message") or accumulated_query

    # NeMo input rail — screens the raw message before the Query Transformer / Supervisor
    rail_update = apply_input_rail(state, latest_user_message)
    if rail_update is not None:
        return rail_update

    qt_result = transform_query(
        accumulated_query=accumulated_query,
        latest_user_message=latest_user_message,
    )

    if qt_result.is_out_of_scope:
        reason = qt_result.condensed
        return {
            "informational_response": (
                f"This pipeline handles EEG functional connectivity analysis "
                f"— I can't help with {reason}, but I'm glad to help "
                f"with frequency bands, channels, conditions, or connectivity metrics."
            ),
            "informational_artifacts": None,
            "plan": None,
            "clarification_question": None,
            "preflight_confirmed": False,
            "gate_1_approved": False,
            "decision_context": None,
        }

    if qt_result.has_contradiction:
        clarification_text = qt_result.clarification
        question = (
            f"## ⚠️ Possible Contradiction Detected\n\n"
            f"> **Contradiction**: {qt_result.contradiction}\n\n"
            f"{clarification_text}\n\n"
            f"---\n_Reply with your clarification, or type `new query: …` to start fresh._"
        )
        clarification = build_clarification(
            "query_contradiction",
            question,
            [{"name": "new_query", "value": "new query:", "label": "↻ Start a new query"}],
        )
        return {
            **legacy_fields(clarification),
            "condition_candidates": [],
            "plan": None,
            "informational_response": None,
            "informational_artifacts": None,
            "preflight_confirmed": False,
            "gate_1_approved": False,
            "decision_context": None,
        }

    # Clean eeg_analysis: pass condensed request to downstream Supervisor
    return {
        "user_request": qt_result.condensed,
        **legacy_fields(None),
        "informational_response": None,
        "informational_artifacts": None,
        "decision_context": None,
    }


def data_prep_node_adapter(state: GraphState) -> Dict[str, Any]:
    """LangGraph node adapter for the deterministic Data Preparation stage.

    Constructs a ``DataPrepInput`` from graph state, calls ``run_data_prep()``,
    and maps the ``DataPrepResult`` fields back to existing GraphState fields.
    """
    from fc_pipeline.deterministic.data_prep.executor import run_data_prep
    from fc_pipeline.deterministic.data_prep.models import DataPrepInput

    plan = state.get("plan")
    manifest = state.get("parameter_manifest")
    run_id = state.get("run_id") or "unknown"
    raw_data_path = state.get("raw_data_path", "")

    # Assemble DataPrepInput from graph state
    try:
        data_prep_input = DataPrepInput(
            raw_data_path=raw_data_path,
            plan=plan,
            parameter_manifest=manifest or [],
            run_id=run_id,
            gate_1_approved=bool(state.get("gate_1_approved", False)),
            preflight_confirmed=bool(state.get("preflight_confirmed", False)),
        )
    except Exception as exc:
        logger.error("Failed to construct DataPrepInput: %s", exc)
        return {
            "data_prep_error": f"INPUT_CONSTRUCTION_FAILED: {type(exc).__name__}: {exc}",
        }

    logger.info("Executing Data Preparation for run_id=%s", run_id)
    result = run_data_prep(data_prep_input)

    # Map DataPrepResult fields → existing GraphState fields
    update: Dict[str, Any] = {
        "bad_channels_dropped": result.bad_channels_dropped or [],
        "channel_plot_paths": result.channel_plot_paths or {},
        "preprocessed_data_path": result.preprocessed_data_path,
        "data_prep_error": result.error,
        "data_prep_summary": result.summary,
    }

    if result.success:
        logger.info(
            "Data Preparation succeeded: %d channels retained, %d epochs, output=%s",
            result.summary.retained_channel_count if result.summary else 0,
            result.summary.epoch_count if result.summary else 0,
            result.preprocessed_data_path,
        )
    else:
        logger.error("Data Preparation failed: %s", result.error)

    return update

