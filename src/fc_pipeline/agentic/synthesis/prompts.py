"""Synthesis system prompt enforcing strict grounded transcription.

Follows Section 6.5 and Section 7 of PIPELINE_PROTOTYPE_DESIGN.md and AGENTS.md style rules.
"""

from __future__ import annotations

from typing import Any, Dict

SYNTHESIS_SYSTEM_PROMPT = """You are the Final Synthesis Agent (Node 5) in an EEG Functional Connectivity Pipeline.

RULE: Transcribe all numbers, channel pairs, manifest values, and Evaluator findings strictly from the provided state — never invent, estimate, or recompute any number.
RULE: Never alter or override any deterministic 4-category connectivity classification or Evaluator verdict.
RULE: Never start or run a new analysis without an explicit NewQuery dispatch.
RULE: Write a concise (3-5 sentence) grounded executive summary paragraph synthesizing the analysis plan, key connectivity findings, and Evaluator quality verdict.
"""


def build_synthesis_user_prompt(state: Dict[str, Any]) -> str:
    """Formats upstream state into a grounded prompt for the executive narrative."""
    plan = state.get("plan")
    verdict = state.get("evaluation_verdict")
    summaries = state.get("numerical_summaries") or {}
    evidence = state.get("evidence_summary") or []

    band_str = (
        f"{plan.freq_band.name} ({plan.freq_band.fmin:g}-{plan.freq_band.fmax:g} Hz)"
        if plan and getattr(plan, "freq_band", None)
        else "unspecified"
    )
    channels = list(getattr(plan, "channels", []) or []) if plan else []
    condition = getattr(plan, "condition", "unspecified") if plan else "unspecified"
    verdict_str = (
        verdict.overall_verdict.value.upper()
        if verdict and getattr(verdict, "overall_verdict", None)
        else "PASS"
    )

    metric_parts = []
    for m, s in summaries.items():
        if m not in ("evidence_summary", "consolidated_report") and isinstance(s, dict):
            metric_parts.append(f"{m.upper()} mean={float(s.get('mean', 0.0)):.4f}, std={float(s.get('std', 0.0)):.4f}")

    return (
        f"Band: {band_str}\n"
        f"Condition: {condition}\n"
        f"Channels ({len(channels)}): {', '.join(channels)}\n"
        f"Total Classified Pairs: {len(evidence)}\n"
        f"Metrics: {'; '.join(metric_parts) or 'None'}\n"
        f"Evaluator Verdict: {verdict_str}\n"
        "Write a concise 3-5 sentence executive summary grounded strictly in these exact values."
    )
