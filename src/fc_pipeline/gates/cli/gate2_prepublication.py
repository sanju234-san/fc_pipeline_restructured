"""Gate 2 review CLI (PASS/WARNING/FAIL, confidence, Top-5 topology read) and pre-publication checks.

Follows Section 3.2, Section 6.4 Step 3, Section 6.5, and AGENTS.md Gate 2 contract.
Strictly offline-testable: ZERO LLM / LangChain / MLflow imports.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

from fc_pipeline.config.thresholds import TAU_PHASE, TAU_ZEROLAG
from fc_pipeline.schemas.evaluation import EvaluationVerdict

_PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # Email
    re.compile(r"\bpatient\s+name\s*:\s*[A-Z][a-z]+", re.IGNORECASE),
]


def verify_report_prepublication(
    report_markdown: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs the 4 mandatory Gate 2 pre-publication checks before final_report_path is set.

    Checks:
      1. phi: No PHI patterns (SSN, email, explicit patient name) in report body.
      2. thresholds: Canonical tau_phase and tau_zerolag values are documented.
      3. numerical_consistency: Every computed metric's mean from numerical_summaries appears verbatim.
      4. references: Mandatory methodological caveats and artifact/manifest references exist.
    """
    issues: List[str] = []
    text = report_markdown or ""

    # 1. PHI check
    phi_ok = True
    for pat in _PHI_PATTERNS:
        if pat.search(text):
            phi_ok = False
            issues.append(f"PHI pattern detected matching {pat.pattern}")

    # 2. Thresholds check
    tau_phase = TAU_PHASE
    tau_zerolag = TAU_ZEROLAG
    for entry in state.get("parameter_manifest") or []:
        name = getattr(entry, "name", None) if not isinstance(entry, dict) else entry.get("name")
        val = (
            (getattr(entry, "human_approved_value", None) or getattr(entry, "proposed_value", None))
            if not isinstance(entry, dict)
            else (entry.get("human_approved_value") or entry.get("proposed_value"))
        )
        if name == "tau_phase" and val is not None:
            try:
                tau_phase = float(val)
            except (ValueError, TypeError):
                pass
        elif name == "tau_zerolag" and val is not None:
            try:
                tau_zerolag = float(val)
            except (ValueError, TypeError):
                pass

    thresholds_ok = (f"{tau_phase:.2f}" in text) and (f"{tau_zerolag:.2f}" in text)
    if not thresholds_ok:
        issues.append(
            f"Report missing canonical thresholds tau_phase={tau_phase:.2f} or tau_zerolag={tau_zerolag:.2f}."
        )

    # 3. Numerical consistency check
    numerical_ok = True
    summaries = state.get("numerical_summaries") or {}
    for metric, stats in summaries.items():
        if metric in ("evidence_summary", "consolidated_report") or not isinstance(stats, dict):
            continue
        mean_val = stats.get("mean")
        if isinstance(mean_val, (int, float)):
            formatted = f"{float(mean_val):.4f}"
            if formatted not in text:
                numerical_ok = False
                issues.append(f"Metric '{metric}' mean {formatted} not found in report body.")

    # 4. Methodological caveats & references check
    required_phrases = [
        "ZERO-LAG COUPLING, CAUSE UNRESOLVED",
        "Multiple Comparisons",
        "Evaluator Quality Audit",
    ]
    references_ok = all(phrase.lower() in text.lower() for phrase in required_phrases)
    if not references_ok:
        issues.append("Report missing mandatory methodological caveats or Evaluator audit section.")

    passed = phi_ok and thresholds_ok and numerical_ok and references_ok
    return {
        "passed": passed,
        "checks": {
            "phi": phi_ok,
            "thresholds": thresholds_ok,
            "numerical_consistency": numerical_ok,
            "references": references_ok,
        },
        "issues": issues,
    }


def format_gate2_cli_review(verdict: EvaluationVerdict, run_id: str = "") -> str:
    """Formats the Gate 2 Pre-Publication Review screen with Subsections A and B."""
    badge = verdict.overall_verdict.value.upper()
    lines = [
        "=" * 72,
        f"GATE 2 — PRE-PUBLICATION REVIEW (Run: {run_id or 'default'})",
        f"Overall Verdict: [{badge}]",
        "=" * 72,
        "",
        "Subsection A — Deterministic Sanity Checks & Artifact Heuristics",
        "-" * 64,
    ]
    for metric, flags in (verdict.per_metric_sanity or {}).items():
        status = "PASS" if flags.confidence == 1.0 else "FAIL"
        lines.append(
            f"  [Sanity] {metric.upper():<8} {status:<6} "
            f"(bounds={flags.values_in_bounds}, sym={flags.matrix_is_symmetric}, "
            f"flat={flags.is_sparse_or_flat}, conf={flags.confidence:.2f})"
        )

    art = verdict.artifacts
    for label, item in [
        ("Ocular Contamination", art.ocular_contamination),
        ("Muscle Contamination", art.muscle_contamination),
        ("Electrode Pop", art.electrode_pop),
    ]:
        state_lbl = "DETECTED" if item.detected else "CLEAR"
        chs = f" [{', '.join(item.affected_channels)}]" if item.affected_channels else ""
        lines.append(f"  [Artifact] {label:<22} {state_lbl:<8} (conf={item.confidence:.2f}){chs} — {item.detail}")

    lines.extend([
        "",
        "Subsection B — Subjective VLM Topology Reads (Non-Deterministic)",
        "-" * 64,
    ])
    if art.top_unresolved_visual_reads:
        for p in art.top_unresolved_visual_reads:
            pair_str = f"{p.channel_pair[0]}-{p.channel_pair[1]}"
            read_str = p.evaluator_visual_read.value if p.evaluator_visual_read else "indeterminate"
            lines.append(
                f"  [Top-5 Pair] {pair_str:<10} read={read_str} (zero_lag={p.zero_lag_mean:.3f}) — {p.evaluator_visual_note or ''}"
            )
    else:
        lines.append("  No unresolved zero-lag pairs required VLM topology inspection.")

    lines.append("=" * 72)
    return "\n".join(lines)


def run_gate2_cli_review(
    verdict: EvaluationVerdict,
    run_id: str = "",
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Interactive CLI prompt for Gate 2 Pre-Publication approval."""
    _ = format_gate2_cli_review(verdict, run_id=run_id)
    ans = input_fn("Approve report for publication? [approve/reject] (Available: approve, reject): ").strip().lower()
    if ans in {"approve", "a", "yes", "y"}:
        return True
    if ans in {"reject", "r", "no", "n"}:
        return False
    raise ValueError(f"Unrecognized Gate 2 response '{ans}'. Available: approve, reject")
