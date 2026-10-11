"""Deep Agents powered post-run follow-up: context window management + subagent delegation.

Public API (see PIPELINE_PROTOTYPE_DESIGN spec — Section "Post-run follow-ups let the Deep Agents harness with
  - context window manager compresses long chat history and only loads the slice of run context
    needed for the current query
  - declarative subagents (RunContextQA, PlotInterpreter, NewAnalysisDispatcher)
    handle each question, so the orchestrator never reruns the supervisor for a new analysis
"""

from fc_pipeline.agentic.followup.context_window import (
    ContextSelection,
    OffloadedTurn,
    PostRunContextWindowManager,
    DEFAULT_CONTEXT_TOKEN_WINDOW,
)
from fc_pipeline.agentic.followup.agents import (
    FollowUpResult,
    FollowUpResultKind,
    run_followup_agent,
)

__all__ = [
    "ContextSelection",
    "OffloadedTurn",
    "PostRunContextWindowManager",
    "DEFAULT_CONTEXT_TOKEN_BUDGET",
    "DEFAULT_CONTEXT_TOKEN_WINDOW",
    "FollowUpResult",
    "FollowUpResultKind",
    "run_followup_agent",
]
