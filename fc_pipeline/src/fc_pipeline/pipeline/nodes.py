"""LangGraph node adapter functions matching Callable[[GraphState], Dict[str, Any]]."""

from typing import Any, Dict

from fc_pipeline.agentic.supervisor.agent import supervisor_node
from fc_pipeline.agentic.supervisor.llm_provider import get_supervisor_llm
from fc_pipeline.schemas.state import GraphState


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
