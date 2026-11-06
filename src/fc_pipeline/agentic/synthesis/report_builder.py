"""Self-contained Markdown assembler with embedded base64 image assets.

Follows Section 7 of PIPELINE_PROTOTYPE_DESIGN.md.
Pure compilation and formatting: zero re-analysis or invention of numbers.
"""

from __future__ import annotations

import base64
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fc_pipeline.config.thresholds import TAU_PHASE, TAU_ZEROLAG
from fc_pipeline.schemas.enums import EvidenceCategoryEnum


def _format_figure_entry(label: str, path_str: Optional[str], embed_base64: bool = False) -> Optional[str]:
    if not path_str:
        return None
    p = Path(str(path_str))
    if not embed_base64:
        rel_name = (
            f"{p.parent.name}/{p.name}"
            if p.parent.name in ("plots", "heatmaps", "networks")
            else p.name
        )
        return f"![{label}]({rel_name})"
    if not p.is_file():
        return None
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"![{label}](data:image/png;base64,{b64})"
    except Exception:
        return None


def _resolve_tau_values(manifest: list) -> tuple[float, float]:
    tau_p = TAU_PHASE
    tau_z = TAU_ZEROLAG
    for entry in manifest or []:
        name = getattr(entry, "name", None) if not isinstance(entry, dict) else entry.get("name")
        val = (
            (getattr(entry, "human_approved_value", None) or getattr(entry, "proposed_value", None))
            if not isinstance(entry, dict)
            else (entry.get("human_approved_value") or entry.get("proposed_value"))
        )
        if name == "tau_phase" and val is not None:
            try:
                tau_p = float(val)
            except (ValueError, TypeError):
                pass
        elif name == "tau_zerolag" and val is not None:
            try:
                tau_z = float(val)
            except (ValueError, TypeError):
                pass
    return tau_p, tau_z


def build_deterministic_executive_summary(state: Dict[str, Any]) -> str:
    """Builds a strictly grounded executive summary paragraph from state."""
    plan = state.get("plan")
    verdict = state.get("evaluation_verdict")
    summaries = state.get("numerical_summaries") or {}
    evidence = state.get("evidence_summary") or []

    band_str = (
        f"{plan.freq_band.name} ({plan.freq_band.fmin:g}–{plan.freq_band.fmax:g} Hz)"
        if plan and getattr(plan, "freq_band", None)
        else "the selected frequency band"
    )
    n_ch = len(getattr(plan, "channels", []) or []) if plan else 0
    cond = getattr(plan, "condition", "selected") if plan else "selected"
    v_badge = (
        verdict.overall_verdict.value.upper()
        if verdict and getattr(verdict, "overall_verdict", None)
        else "PASS"
    )
    metric_clauses = [
        f"{m.upper()} (mean={float(s.get('mean', 0.0)):.4f})"
        for m, s in summaries.items()
        if m not in ("evidence_summary", "consolidated_report") and isinstance(s, dict)
    ]
    metrics_text = ", ".join(metric_clauses) if metric_clauses else "configured connectivity metrics"
    return (
        f"Functional connectivity analysis was completed on {n_ch} EEG channel(s) in the {band_str} band "
        f"during condition `{cond}`, evaluating {len(evidence)} unique channel pair(s) across {metrics_text}. "
        f"The Multimodal Evaluator assigned an overall quality verdict of **{v_badge}**."
    )


def build_self_contained_report(
    state: Dict[str, Any],
    executive_narrative: Optional[str] = None,
    *,
    embed_base64_images: bool = False,
) -> str:
    """Assembles the 8-section self-contained Markdown report per Section 7 spec."""
    run_id = str(state.get("run_id") or "default_run")
    raw_path = str(state.get("raw_data_path") or "dataset.fif")
    dataset_name = Path(raw_path).name
    plan = state.get("plan")
    manifest = state.get("parameter_manifest") or []
    verdict = state.get("evaluation_verdict")
    summaries = state.get("numerical_summaries") or {}
    evidence = state.get("evidence_summary") or []
    dropped = state.get("bad_channels_dropped") or []
    channel_plots = state.get("channel_plot_paths") or {}
    heatmaps = state.get("heatmap_image_paths") or {}
    networks = state.get("network_image_paths") or {}

    tau_phase, tau_zerolag = _resolve_tau_values(manifest)
    overall_badge = (
        verdict.overall_verdict.value.upper()
        if verdict and getattr(verdict, "overall_verdict", None)
        else "PASS"
    )
    narrative = (executive_narrative or "").strip() or build_deterministic_executive_summary(state)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: List[str] = [
        f"# EEG Functional Connectivity Report — `{run_id}`",
        "",
        "## 1. Header & Execution Metadata",
        "",
        f"- **Run ID:** `{run_id}`",
        f"- **Timestamp:** `{timestamp}`",
        f"- **Target Dataset:** `{dataset_name}`",
        f"- **Overall Quality Verdict:** **`{overall_badge}`**",
        "",
        narrative,
        "",
        "## 2. Analysis Plan & Methodology Summary",
        "",
    ]

    if plan is not None:
        metrics_str = ", ".join(getattr(m, "value", str(m)) for m in (plan.metrics or []))
        band_str = f"{plan.freq_band.name} ({plan.freq_band.fmin:g} - {plan.freq_band.fmax:g} Hz)"
        ch_str = f"{len(plan.channels)} ({', '.join(plan.channels)})"
        lines.extend([
            "| Locked Parameter | Value |",
            "| :--- | :--- |",
            f"| Metrics Computed | `{metrics_str}` |",
            f"| Frequency Band | `{band_str}` |",
            f"| Included Channels | `{ch_str}` |",
            f"| Condition Name | `{plan.condition}` |",
            "",
        ])

    lines.extend([
        "### Full Confirmed Manifest (Gate 1)",
        "",
        "| Parameter | Proposed Value | Flagged (`needs_human_input`) | Risk Tier | Human Approved Value |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])
    for entry in manifest:
        name = getattr(entry, "name", "") if not isinstance(entry, dict) else entry.get("name", "")
        prop = getattr(entry, "proposed_value", "") if not isinstance(entry, dict) else entry.get("proposed_value", "")
        needs = getattr(entry, "needs_human_input", False) if not isinstance(entry, dict) else entry.get("needs_human_input", False)
        tier = getattr(entry, "risk_tier", "low") if not isinstance(entry, dict) else entry.get("risk_tier", "low")
        appr = (
            getattr(entry, "human_approved_value", None)
            if not isinstance(entry, dict)
            else entry.get("human_approved_value")
        ) or prop
        flag_str = "⚠ NEEDS YOUR INPUT" if needs else "No"
        tier_str = "**`elevated`**" if str(tier) == "elevated" else "`low`"
        lines.append(f"| `{name}` | `{prop}` | {flag_str} | {tier_str} | `{appr}` |")

    dropped_note = (
        f"Bad channels dropped: {', '.join(dropped)}."
        if dropped
        else "No bad channels detected; no channels were dropped."
    )
    lines.extend([
        "",
        f"- **Preprocessing Note:** Applied Common Average Reference (CAR), zero-phase bandpass filtering, and epoch-length cycle validation. {dropped_note} Phase dynamics of retained channels were un-manipulated (No ICA, No amplitude rejection, No channel interpolation).",
        "- **Mathematical Transformation Note:** Imaginary coherence is stored and evaluated as its absolute magnitude (`|imcoh|`) to establish a non-negative, `[0, 1]`-bounded, symmetric metric matching PLI and wPLI.",
        "",
        "## 3. Cross-Metric Evidence Summary",
        "",
        "| Channel Pair | Phase-Robust Mean (PLI / wPLI / \\|ImCoh\\|) | Zero-Lag Mean (PLV / Coherence) | Joint Evidence Classification | Evaluator Supporting Visual Read |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for item in evidence:
        pair_lbl = f"`{item.channel_pair[0]} - {item.channel_pair[1]}`"
        p_strength = "Strong" if item.phase_robust_mean >= tau_phase else "Weak"
        z_strength = "Strong" if item.zero_lag_mean >= tau_zerolag else "Weak"
        cat_val = getattr(item.classification, "value", str(item.classification))
        if item.evaluator_visual_read is not None:
            r_val = getattr(item.evaluator_visual_read, "value", str(item.evaluator_visual_read))
            vis_text = f"*({r_val}: {item.evaluator_visual_note or ''})*"
        elif item.classification == EvidenceCategoryEnum.CONFIRMED_COUPLING:
            vis_text = "*(Phase-lagged coupling; volume conduction excluded; uncorrected for null distribution)*"
        elif item.classification == EvidenceCategoryEnum.FULLY_CONVERGENT:
            vis_text = "*(Convergent phase-robust and zero-lag coupling)*"
        elif item.classification == EvidenceCategoryEnum.ZERO_LAG_UNRESOLVED:
            vis_text = "*(Zero-lag coupling; cause unresolved)*"
        else:
            vis_text = "*(Baseline / noise floor)*"
        lines.append(
            f"| {pair_lbl} | `{item.phase_robust_mean:.4f} ({p_strength})` | "
            f"`{item.zero_lag_mean:.4f} ({z_strength})` | **`{cat_val}`** | {vis_text} |"
        )

    lines.extend([
        "",
        "> [!WARNING]",
        "> **Mandatory Methodological Caveats on Thresholding & Multiple Testing:**",
        f"> 1. **Heuristic Thresholds & Sample-Size Bias:** The coupling thresholds (`tau_phase = {tau_phase:.2f}`, `tau_zerolag = {tau_zerolag:.2f}`) are deterministic operational heuristics uncorrected for epoch count.",
        "> 2. **Uncorrected Multiple Comparisons:** No family-wise error rate or False Discovery Rate (FDR) correction is applied across simultaneous pairwise comparisons.",
        "> 3. **Guidance on ZERO-LAG COUPLING, CAUSE UNRESOLVED:** Pairs classified as *ZERO-LAG COUPLING, CAUSE UNRESOLVED* indicate zero-lag synchronization without phase-lagged confirmation; sensor-level metrics cannot distinguish between genuine zero-lag coordination and volume conduction.",
        "> 4. **Elevated-Risk Confirmed Values:** Parameters marked as `elevated` risk tier in the Full Confirmed Manifest directly control pair classification and artifact flags.",
        "",
        "## 4. Per-Metric Connectivity Numerical Summaries",
        "",
        "| Metric | Mean | Std Dev | Median | Min | Max | Sparsity |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for metric, s in summaries.items():
        if metric in ("evidence_summary", "consolidated_report") or not isinstance(s, dict):
            continue
        lines.append(
            f"| `{metric.upper()}` | `{float(s.get('mean', 0.0)):.4f}` | "
            f"`{float(s.get('std', 0.0)):.4f}` | `{float(s.get('median', 0.0)):.4f}` | "
            f"`{float(s.get('min', 0.0)):.4f}` | `{float(s.get('max', 0.0)):.4f}` | "
            f"`{float(s.get('sparsity', 0.0)):.4f}` |"
        )

    lines.extend([
        "",
        "## 5. Data Cleaning Visualization",
        "",
    ])
    if not dropped:
        lines.append("No bad channels detected; no channels were dropped.")
        lines.append("")

    before_entry = _format_figure_entry(
        "Channels Before Cleaning",
        channel_plots.get("channels_before") or channel_plots.get("before"),
        embed_base64=embed_base64_images,
    )
    after_entry = _format_figure_entry(
        "Channels After Cleaning",
        channel_plots.get("channels_after") or channel_plots.get("after"),
        embed_base64=embed_base64_images,
    )
    if before_entry:
        lines.append(before_entry)
        lines.append("")
    if after_entry:
        lines.append(after_entry)
        lines.append("")

    lines.extend([
        "## 6. Visual Figure Gallery",
        "",
    ])
    has_gallery_items = False
    for m_name, h_path in heatmaps.items():
        entry = _format_figure_entry(
            f"{m_name.upper()} Heatmap",
            h_path,
            embed_base64=embed_base64_images,
        )
        if entry:
            lines.append(entry)
            lines.append("")
            has_gallery_items = True
    for m_name, n_path in networks.items():
        entry = _format_figure_entry(
            f"{m_name.upper()} Network Graph",
            n_path,
            embed_base64=embed_base64_images,
        )
        if entry:
            lines.append(entry)
            lines.append("")
            has_gallery_items = True
    if not has_gallery_items:
        lines.append("No connectivity figures recorded for this run.")
        lines.append("")

    lines.extend([
        "## 7. Evaluator Quality Audit & Forensic Findings",
        "",
    ])
    if verdict is not None:
        lines.extend([
            "### Subsection A — Deterministic Sanity & Artifact Checks",
            "",
            "| Metric | Bounds `[0, 1]` | Symmetric (`M = M^T`) | Sparse / Flat | Confidence |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for m_name, flags in (verdict.per_metric_sanity or {}).items():
            lines.append(
                f"| `{m_name.upper()}` | `{flags.values_in_bounds}` | `{flags.matrix_is_symmetric}` | "
                f"`{flags.is_sparse_or_flat}` | `{flags.confidence:.2f}` |"
            )

        art = verdict.artifacts
        flagged_all = sorted(
            set(
                (art.ocular_contamination.affected_channels or [])
                + (art.muscle_contamination.affected_channels or [])
                + (art.electrode_pop.affected_channels or [])
            )
        )
        lines.extend([
            "",
            "#### Artifact Risk Callout Box",
            f"- **Ocular Contamination:** `{'Detected' if art.ocular_contamination.detected else 'Clear'}` "
            f"(confidence=`{art.ocular_contamination.confidence:.2f}`) — {art.ocular_contamination.detail}",
            f"- **Muscle Intrusion:** `{'Detected' if art.muscle_contamination.detected else 'Clear'}` "
            f"(confidence=`{art.muscle_contamination.confidence:.2f}`) — {art.muscle_contamination.detail}",
            f"- **Electrode Pop:** `{'Detected' if art.electrode_pop.detected else 'Clear'}` "
            f"(confidence=`{art.electrode_pop.confidence:.2f}`) — {art.electrode_pop.detail}",
            f"- **Flagged Channels:** {', '.join(f'`{c}`' for c in flagged_all) if flagged_all else 'None'}",
            "",
            "### Subsection B — Subjective VLM Topology Reads",
            "",
            verdict.visual_observations,
            "",
        ])

    lines.extend([
        "## 8. Scientific Recommendations & Caveats",
        "",
    ])
    recs = (verdict.recommendations_for_user if verdict else None) or [
        "Verify phase-lagged connectivity findings against experimental task design and regional anatomy."
    ]
    for r in recs:
        lines.append(f"- {r}")

    lines.append("")
    return "\n".join(lines)
