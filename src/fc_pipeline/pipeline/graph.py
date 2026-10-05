"""StateGraph assembly, node addition, edge routing, and checkpoint gates."""

from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from fc_pipeline.schemas.state import GraphState
from fc_pipeline.schemas.clarification import (
    AXIS_KINDS,
    build_clarification,
    get_clarification,
    legacy_fields,
)
from fc_pipeline.agentic.supervisor.hitl_resolution import apply_clarification_reply
from fc_pipeline.agentic.supervisor.action_policy import (
    KIND_INPUT_RAIL,
    RAIL_CONTEXT_KINDS,
    evaluate_action,
)
from fc_pipeline.pipeline.nodes import (
    apply_output_rail,
    data_prep_node_adapter,
    supervisor_node_adapter,
    query_transformer_node_adapter,
)


def route_query_transformer_output(state: GraphState) -> str:
    """Evaluates Query Transformer output to determine next pipeline step.

    Four outcomes:
      - "gate_1_review": The NeMo input rail flagged the message and the decision
        policy said ask_human (decision_context set) — pause via the existing gate.
      - "informational_complete": Request classified as out_of_scope (informational_response set).
      - "clarification_pause": Contradiction detected requiring human resolution (clarification_question set).
      - "supervisor": Valid eeg_analysis request -> route to Supervisor for scientific validation.
    """
    ctx = state.get("decision_context")
    if ctx and not ctx.get("resolution"):
        return "gate_1_review"
    if state.get("informational_response"):
        return "informational_complete"
    if state.get("clarification_question"):
        return "clarification_pause"
    return "supervisor"


def route_supervisor_output(state: GraphState) -> str:
    """Evaluates Supervisor output to determine next pipeline step.

    Four outcomes:
      - "gate_1_review": Valid plan constructed, route to human preflight.
      - "informational_complete": Request fully satisfied by informational/
        diagnostic tools alone (e.g. a bare overview plot) — no AnalysisPlan
        was needed and nothing further to ask. Checked before
        "clarification_pause" since the Supervisor may set neither plan nor
        a genuine clarification_question for this outcome.
      - "clarification_pause": Supervisor halted with a question for the user.
      - "supervisor_error": Unexpected state — neither plan, informational
        response, nor clarification.
    """
    if state.get("plan") is not None:
        return "gate_1_review"
    if state.get("informational_response"):
        return "informational_complete"
    if state.get("clarification_question"):
        return "clarification_pause"
    return "supervisor_error"


def gate_1_review(state: GraphState) -> Dict[str, Any]:
    """Human-in-the-Loop checkpoint node (generalized Gate 1).

    Uses LangGraph's native interrupt() mechanism to suspend graph execution
    and wait for an explicit human decision (approve / reject; "request
    changes" and per-row edits are handled by the Chainlit handler).

    The pause is driven by a `decision_context` produced by
    evaluate_action(): when absent, this is the classic manifest review; when
    set by a NeMo rail (input_rail / output_rail) it presents that pause
    through the same interrupt -> Chainlit -> Command(resume=...) path.

    Manifest review: only an explicit {"action": "approve"} sets
    preflight_confirmed=True and gate_1_approved=True (unchanged behaviour).
    Rail review: the answer is recorded as decision_context["resolution"] and
    never touches the Gate 1 approval flags.
    """
    ctx = state.get("decision_context")
    if not ctx:
        ctx = evaluate_action(
            "gate_1_manifest_review",
            {},
            {"run_id": state.get("run_id")},
        ).decision_context

    decision = interrupt({
        "type": "gate_1_human_review",
        "plan": state.get("plan"),
        "parameter_manifest": state.get("parameter_manifest"),
        "run_id": state.get("run_id"),
        "decision_context": ctx,
    })

    action = decision.get("action") if isinstance(decision, dict) else str(decision)

    if ctx.get("kind") in RAIL_CONTEXT_KINDS:
        approved = action == "approve"
        update: Dict[str, Any] = {
            "decision_context": {**ctx, "resolution": "approved" if approved else "rejected"},
        }
        if approved and ctx.get("kind") == KIND_INPUT_RAIL:
            update["input_rail_cleared"] = True
        return update

    if action == "approve":
        return {
            "preflight_confirmed": True,
            "gate_1_approved": True,
        }
    return {
        "preflight_confirmed": False,
        "gate_1_approved": False,
    }


def route_gate_1_output(state: GraphState) -> str:
    """Routes from Gate 1 to downstream execution only if human approved.

    - If preflight_confirmed=True and gate_1_approved=True:
        Routes to Data Preparation.
    - If a human cleared a NeMo *input* rail pause: loops back to the Query
      Transformer (which skips the rail via input_rail_cleared) so the request
      continues through the normal pipeline.
    - Otherwise: terminates at END without downstream execution.
    """
    ctx = state.get("decision_context") or {}
    if ctx.get("kind") == KIND_INPUT_RAIL and ctx.get("resolution") == "approved":
        return "query_transformer"
    if state.get("preflight_confirmed") and state.get("gate_1_approved"):
        return "data_prep"
    return END


def informational_complete(state: GraphState) -> Dict[str, Any]:
    """Informational/diagnostic-only completions.

    Reached when the request is out-of-scope or when the Supervisor fully
    answered a request that never needed any of the 3 mandatory scientific
    axes (e.g. "show me an overview plot of this dataset").

    The NeMo output rail screens the Supervisor's LLM-authored answer against
    the tool observations before it is shown (no-op when rails are disabled or
    there is nothing to ground against).
    """
    return apply_output_rail(state)


def route_informational_output(state: GraphState) -> str:
    """Routes to the existing gate when the output rail's decision is ask_human."""
    ctx = state.get("decision_context")
    if ctx and not ctx.get("resolution"):
        return "gate_1_review"
    return END


def clarification_pause(state: GraphState) -> Dict[str, Any]:
    """Native HITL pause for Supervisor/query clarification.

    The interrupt carries the ONE authoritative clarification payload
    (``axis``/``question``/``options``/``allow_manual``) so the UI can never
    receive a question without an action list.  On resume the reply is
    validated against the actual dataset, normalised and persisted into the
    resolved-axis GraphState fields *here* (not left to the LLM), and the SAME
    checkpoint continues to the Supervisor, which then skips the resolved axis.
    An invalid reply keeps the axis unresolved and loops back to this node with
    an explanatory question (same checkpoint, same options).
    """
    clarification = get_clarification(state) or build_clarification(
        "clarification", "Please provide the missing analysis parameter(s)."
    )
    decision = interrupt({
        "type": "clarification",
        "clarification": clarification,
        "question": clarification["question"],
        "kind": clarification["kind"],
        "options": clarification["options"],
        "allow_manual": True,
        "condition_candidates": state.get("condition_candidates") or [],
        "run_id": state.get("run_id"),
    })
    if isinstance(decision, dict):
        reply = str(decision.get("reply") or decision.get("value") or "").strip()
    else:
        reply = str(decision).strip()

    if not reply:
        # Nothing usable was supplied: re-ask (never fall through to a blank text box).
        return {**legacy_fields(clarification), "clarification_response": None}

    kind = clarification.get("kind")
    axis_update: Dict[str, Any] = {}
    if kind in AXIS_KINDS:
        axis_update, error = apply_clarification_reply(state, kind, reply)
        if error:
            retry = build_clarification(
                kind,
                f"{error} Please choose one of the options below or type a valid value.",
                clarification.get("options"),
                axis=clarification.get("axis"),
            )
            return {**legacy_fields(retry), "clarification_response": None}

    current_request = state.get("user_request", "")
    updated_request = (
        f"{current_request}\n[User clarification reply]: {reply}"
        if current_request else reply
    )
    return {
        **axis_update,
        "user_request": updated_request,
        "latest_user_message": reply,
        "clarification_response": reply,
        "clarification_resume_kind": kind,
        **legacy_fields(None),
        "condition_candidates": [],
        "informational_response": None,
        "informational_artifacts": None,
    }


def route_clarification_output(state: GraphState) -> str:
    """After clarification, return to the appropriate reasoning stage.

    - Invalid/empty reply (question re-issued, no response recorded): loop back
      to ``clarification_pause`` on the same checkpoint.
    - Query-transformer contradictions re-condense via the transformer.
    - Supervisor axis clarifications go straight back to the Supervisor while
      retaining resolved-axis caches in GraphState.
    """
    if state.get("clarification_response") is None:
        if state.get("clarification_question"):
            return "clarification_pause"
        return END
    if state.get("clarification_resume_kind") == "query_contradiction":
        return "query_transformer"
    return "supervisor"


def supervisor_error(state: GraphState) -> Dict[str, Any]:
    """Terminal error node for unexpected supervisor state.

    Reached only when supervisor_node returns neither a plan, an
    informational response, nor a clarification question — an internal
    invariant violation.
    """
    return {
        "pipeline_error": (
            "INTERNAL: Supervisor completed without producing a plan, "
            "an informational response, or a clarification question. "
            "This indicates an internal logic error."
        ),
    }


def approve_gate_1(state: GraphState) -> Dict[str, Any]:
    """Helper to record explicit human approval of Gate 1.

    Validates that a plan exists and returns state updates setting
    preflight_confirmed=True and gate_1_approved=True. The plan remains unchanged.
    """
    if state.get("plan") is None:
        raise ValueError("Cannot approve Gate 1: no AnalysisPlan exists in state.")
    return {
        "preflight_confirmed": True,
        "gate_1_approved": True,
    }


def reject_gate_1(state: GraphState) -> Dict[str, Any]:
    """Helper to record explicit human rejection of Gate 1."""
    return {
        "preflight_confirmed": False,
        "gate_1_approved": False,
    }


def build_pipeline_graph() -> StateGraph:
    """Constructs and returns the EEG FC Pipeline StateGraph.

    Architecture:
      START → query_transformer → route_query_transformer_output
        → gate_1_review          (if NeMo input rail flagged + ask_human)
        → informational_complete (if out_of_scope)
        → clarification_pause    (if contradiction)
        → supervisor             (if valid eeg_analysis)
            → route_supervisor_output
                → gate_1_review          (if plan is not None)
                → informational_complete (if informational_response is set)
                → clarification_pause    (if clarification_question is set)
                → supervisor_error       (internal error)
                    → gate_1_review (native interrupt checkpoint)
                        → route_gate_1_output (checks preflight_confirmed & gate_1_approved)
                            → downstream boundary (future Data Preparation) / END
                            → query_transformer (only after a human clears an input-rail pause)
        informational_complete → gate_1_review (only if the output rail says ask_human) / END
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("query_transformer", query_transformer_node_adapter)
    graph.add_node("supervisor", supervisor_node_adapter)
    graph.add_node("gate_1_review", gate_1_review)
    graph.add_node("data_prep", data_prep_node_adapter)
    graph.add_node("informational_complete", informational_complete)
    graph.add_node("clarification_pause", clarification_pause)
    graph.add_node("supervisor_error", supervisor_error)

    # Entry point is Query Transformer — structurally impossible to bypass
    graph.set_entry_point("query_transformer")

    # Conditional routing from Query Transformer
    graph.add_conditional_edges(
        "query_transformer",
        route_query_transformer_output,
        {
            "supervisor": "supervisor",
            "informational_complete": "informational_complete",
            "clarification_pause": "clarification_pause",
            "gate_1_review": "gate_1_review",
        },
    )

    # Conditional routing from supervisor output
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor_output,
        {
            "gate_1_review": "gate_1_review",
            "informational_complete": "informational_complete",
            "clarification_pause": "clarification_pause",
            "supervisor_error": "supervisor_error",
        },
    )

    # Gate 1 conditional routing to Data Preparation or downstream
    graph.add_conditional_edges(
        "gate_1_review",
        route_gate_1_output,
        {
            END: END,
            "data_prep": "data_prep",
            # NeMo input-rail override: resume the request at the Query Transformer
            "query_transformer": "query_transformer",
        },
    )

    # Informational completion: output-rail ask_human pauses at the existing gate
    graph.add_conditional_edges(
        "informational_complete",
        route_informational_output,
        {
            "gate_1_review": "gate_1_review",
            END: END,
        },
    )

    # Terminal / clarification edges
    graph.add_edge("data_prep", END)
    graph.add_conditional_edges(
        "clarification_pause",
        route_clarification_output,
        {
            "supervisor": "supervisor",
            "query_transformer": "query_transformer",
            "clarification_pause": "clarification_pause",
            END: END,
        },
    )
    graph.add_edge("supervisor_error", END)

    return graph


def compile_pipeline_app(checkpointer: Optional[Any] = None):
    """Compiles the pipeline StateGraph with a checkpointer (defaults to MemorySaver)."""
    if checkpointer is None:
        checkpointer = MemorySaver()
    return build_pipeline_graph().compile(checkpointer=checkpointer)
