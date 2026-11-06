"""Synthesis LLM prompt invocation for structured report narrative.

Follows Section 6.5 and Section 7 of PIPELINE_PROTOTYPE_DESIGN.md and AGENTS.md Gate 2 rules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fc_pipeline.agentic.synthesis.prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_user_prompt,
)
from fc_pipeline.agentic.synthesis.report_builder import (
    build_deterministic_executive_summary,
    build_self_contained_report,
)
from fc_pipeline.config.constants import OUTPUT_DIR, REPORT_FILENAME_TEMPLATE
from fc_pipeline.gates.cli.gate2_prepublication import verify_report_prepublication
from fc_pipeline.schemas.state import GraphState

logger = logging.getLogger(__name__)


def run_synthesis_agent(
    state: GraphState,
    *,
    llm: Optional[Any] = None,
    output_dir: Optional[Path] = None,
    embed_base64_images: bool = False,
) -> Dict[str, Any]:
    """Executes Node 5 (Final Synthesis Agent) and enforces Gate 2 verification before writing report_<run_id>.md."""
    run_id = str(state.get("run_id") or "default_run")
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    executive_narrative: Optional[str] = None
    if llm is not None:
        try:
            user_prompt = build_synthesis_user_prompt(state)
            resp = llm.invoke(
                [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            candidate = str(getattr(resp, "content", resp)).strip()
            if candidate:
                executive_narrative = candidate
        except Exception as exc:
            logger.warning("Synthesis LLM narrative call failed (%s); using deterministic summary.", exc)

    report_md = build_self_contained_report(
        state,
        executive_narrative=executive_narrative,
        embed_base64_images=embed_base64_images,
    )

    # Gate 2 mandatory 4-check prepublication verification (PHI, thresholds, numerical_consistency, references)
    gate2_check = verify_report_prepublication(report_md, state)
    if not gate2_check["passed"] and executive_narrative is not None:
        logger.warning(
            "Gate 2 prepublication check flagged LLM narrative (%s); falling back to deterministic narrative.",
            gate2_check["issues"],
        )
        report_md = build_self_contained_report(
            state,
            executive_narrative=build_deterministic_executive_summary(state),
            embed_base64_images=embed_base64_images,
        )
        gate2_check = verify_report_prepublication(report_md, state)

    if not gate2_check["passed"]:
        logger.error("Gate 2 prepublication verification failed: %s", gate2_check["issues"])
        return {
            "pipeline_error": f"GATE2_VERIFICATION_FAILED: {'; '.join(gate2_check['issues'])}",
            "final_report_path": None,
        }

    report_path = out_dir / REPORT_FILENAME_TEMPLATE.format(run_id=run_id)
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "final_report_path": str(report_path.resolve()),
    }
