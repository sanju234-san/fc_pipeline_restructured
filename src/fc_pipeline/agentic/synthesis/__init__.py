"""Re-exports Final Synthesis node runner."""

from fc_pipeline.agentic.synthesis.agent import run_synthesis_agent
from fc_pipeline.agentic.synthesis.report_builder import build_self_contained_report

__all__ = ["run_synthesis_agent", "build_self_contained_report"]
