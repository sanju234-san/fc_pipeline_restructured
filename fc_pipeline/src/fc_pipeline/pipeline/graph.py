"""StateGraph assembly, node addition, edge routing, and checkpoint gates."""

from typing import Any, Dict

from langgraph.graph import StateGraph, END

from fc_pipeline.schemas.state import GraphState
from fc_pipeline.pipeline.nodes import supervisor_node_adapter


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
    # Neither plan, informational response, nor clarification: unexpected
    # internal state. Route to a distinct error node so the graph does not
    # silently stall.
    return "supervisor_error"


def gate_1_review(state: GraphState) -> Dict[str, Any]:
    """Placeholder for Gate 1 human review node.

    In the full pipeline, this node presents the parameter manifest
    for human approval before Data Preparation proceeds.
    Currently a pass-through stub.
    """
    return {}


def informational_complete(state: GraphState) -> Dict[str, Any]:
    """Placeholder for informational/diagnostic-only completions.

    Reached when the Supervisor fully answered a request that never needed
    any of the 3 mandatory scientific axes (e.g. "show me an overview plot
    of this dataset", "what conditions does this dataset have"). There is
    no AnalysisPlan to gate and nothing further to ask the user.
    Currently a pass-through stub — the frontend renders
    `informational_response` (and any artifact in `informational_artifacts`,
    e.g. the overview plot image path) directly.
    """
    return {}


def clarification_pause(state: GraphState) -> Dict[str, Any]:
    """Placeholder for clarification pause node.

    In the full pipeline, this node presents the clarification question
    and waits for user input before re-invoking the Supervisor.
    Currently a pass-through stub.
    """
    return {}


def supervisor_error(state: GraphState) -> Dict[str, Any]:
    """Terminal error node for unexpected supervisor state.

    Reached only when supervisor_node returns neither a plan, an
    informational response, nor a clarification question — an internal
    invariant violation.
    Writes strictly to pipeline_error, preserving data_prep_error for Node 2.
    """
    return {
        "pipeline_error": (
            "INTERNAL: Supervisor completed without producing a plan, "
            "an informational response, or a clarification question. "
            "This indicates an internal logic error."
        ),
    }


def build_pipeline_graph() -> StateGraph:
    """Constructs and returns the EEG FC Pipeline StateGraph.

    Current wiring (Step 0-2):
      START → supervisor → route_supervisor_output
        → gate_1_review          (if plan is not None)
        → informational_complete (if informational_response is set)
        → clarification_pause    (if clarification_question is set)
        → supervisor_error       (none of the above — unexpected state)

    All terminal nodes currently route to END.
    Nodes 2-5 and Gate 2 will be wired as implemented.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node_adapter)
    graph.add_node("gate_1_review", gate_1_review)
    graph.add_node("informational_complete", informational_complete)
    graph.add_node("clarification_pause", clarification_pause)
    graph.add_node("supervisor_error", supervisor_error)

    # Entry point
    graph.set_entry_point("supervisor")

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

    # Terminal edges (stubs — will extend as Nodes 2-5 are implemented)
    graph.add_edge("gate_1_review", END)
    graph.add_edge("informational_complete", END)
    graph.add_edge("clarification_pause", END)
    graph.add_edge("supervisor_error", END)

    return graph
