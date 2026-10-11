---
name: Artifact Scoring Review
description: Score ocular / muscle / bad-channel artifact severity for Gate 2 pre-publication, using deterministic Data Prep outputs.
version: 1.0.0
author: fc-pipeline-team
triggers:
  - artifact
  - ocular
  - muscle
  - bad channel
  - severity
  - gate 2 score
  - gate 2
  - prepublication
  - signal quality
  - noise
tags: [data-prep, gate2, evaluation, quality-assurance]
inputs:
  - data_prep_summary
  - bad_channels_dropped
  - channel_plot_paths
  - parameter_manifest
outputs:
  - artifact_severity_score
requires_model: false
---

# Artifact Severity Scoring (Gate 2)

This skill is used by the Evaluator node and the Gate 2 CLI to produce a
semi-deterministic artifact-severity score that is written into the final
report.  All heuristics come from the engineering thresholds defined in
`config/thresholds.py` and the manifest rows `ocular_variance_ratio` and
`muscle_power_threshold`.

## 1. Scoring rubric

**A score is an integer 0 (clean) → 5 (severe), plus a free-text rationale.**

| Band | Score | Criteria (any met gets that band) |
|:---|:---:|:---|
| Clean | 0 | No bad channels dropped; ocular_variance_ratio ≤ 1.2; muscle_power_threshold residual ≤ 0.5× baseline |
| Mild | 1 | 1 bad channel dropped; OR ocular_variance_ratio 1.2–1.5; OR muscle residual 0.5–1.0× |
| Moderate | 2 | 2 bad channels dropped; OR ocular_variance_ratio 1.5–2.0; OR muscle residual 1.0–1.5× |
| Marked | 3 | 3+ bad channels dropped; OR ocular_variance_ratio 2.0–3.0; OR muscle residual 1.5–2.0× |
| Severe | 4 | trial_adequacy advisory was flagged (≤3 trials) AND any band 2+ criterion |
| Extreme | 5 | Either: (a) ≥50% of original channels were dropped, OR (b) ocular_variance_ratio > 3.0 AND muscle residual > 2.0× |

## 2. Rationale template

```
Artifact Severity Score: {score}/5 — {label}

Rationale:
  * Bad channels dropped: {n_dropped}/{original_count} channels.
  * Ocular variance ratio: {ocular_ratio:.2f} (threshold nominal ≤ 1.5).
  * Muscle power residual: {muscle_residual:.2f}× baseline (nominal ≤ 1.0×).
  * Trial adequacy: {trial_count} epochs retained for condition {condition}.
  * Diagnostic plots: {plot_labels}.

Gate 2 recommendation: {recommendation}.
```

## 3. Gate 2 recommendation policy

Given score S:
  - S ≤ 1 → **Pass without notes** (but still report score).
  - 2 ≤ S ≤ 3 → **Pass with caveats** (copy the rationale verbatim into Section 4
    Limitations of the synthesis report).
  - S = 4 → **Conditional pass** — reviewer must see the before/after diagnostic
    plots before final sign-off.
  - S = 5 → **Fail** — report not cleared for public release; re-run Data Prep
    with explicit artifact rejection (manual ICA, stricter bad-channel screen,
    or a different reference choice).
