"""Multimodal prompts for visual artifact and Top-5 topology reads.

Follows Section 6.4 Step 3 of PIPELINE_PROTOTYPE_DESIGN.md and AGENTS.md style rules.
"""

from __future__ import annotations

from typing import Dict, List

from fc_pipeline.schemas.evaluation import ArtifactSignatures, SanityCheckFlags
from fc_pipeline.schemas.evidence import PairEvidenceSynthesis

EVALUATOR_SYSTEM_PROMPT = """You are the Multimodal Evaluator Agent (Node 4) in an EEG Functional Connectivity Pipeline.

RULE: Never invent connectivity values, thresholds, or channel labels.
RULE: Never re-classify any channel pair; deterministic 4-category labels (such as ZERO-LAG COUPLING, CAUSE UNRESOLVED) are locked upstream in Node 3.
RULE: Never start or run a new analysis without an explicit NewQuery dispatch.
RULE: Provide supporting visual topology observations strictly for the Top-5 unresolved zero-lag pairs supplied in the prompt.

Return ONLY valid JSON with this exact structure:
{
  "visual_observations": "2-3 sentence summary of heatmap/network spatial patterns and artifact signatures.",
  "recommendations_for_user": [
    "Actionable scientific caveat 1",
    "Actionable scientific caveat 2"
  ],
  "pair_reads": [
    {
      "pair": ["ChA", "ChB"],
      "read": "plausible_neural_topology | consistent_with_artifact | indeterminate",
      "note": "Concise spatial topology note for this pair"
    }
  ]
}
"""


def build_evaluator_user_prompt(
    top_unresolved: List[PairEvidenceSynthesis],
    artifacts: ArtifactSignatures,
    sanity: Dict[str, SanityCheckFlags],
) -> str:
    """Builds the structured user prompt summarizing deterministic checks and Top-5 unresolved pairs."""
    sanity_lines = [
        f"- {m}: bounds={f.values_in_bounds}, symmetric={f.matrix_is_symmetric}, "
        f"sparse_or_flat={f.is_sparse_or_flat}, confidence={f.confidence:.2f}"
        for m, f in sanity.items()
    ]
    artifact_lines = [
        f"- ocular_contamination: detected={artifacts.ocular_contamination.detected} "
        f"({artifacts.ocular_contamination.detail})",
        f"- muscle_contamination: detected={artifacts.muscle_contamination.detected} "
        f"({artifacts.muscle_contamination.detail})",
        f"- electrode_pop: detected={artifacts.electrode_pop.detected} "
        f"({artifacts.electrode_pop.detail})",
    ]
    pair_lines = [
        f"- {p.channel_pair[0]}-{p.channel_pair[1]}: phase_robust_mean={p.phase_robust_mean:.4f}, "
        f"zero_lag_mean={p.zero_lag_mean:.4f}, classification={p.classification.value}"
        for p in top_unresolved
    ] or ["- None (0 unresolved zero-lag pairs)"]

    return (
        "### Deterministic Sanity Checks\n"
        + "\n".join(sanity_lines or ["- None"])
        + "\n\n### Deterministic Artifact Heuristics\n"
        + "\n".join(artifact_lines)
        + "\n\n### Top-5 Unresolved Zero-Lag Pairs (Ranked by Zero-Lag Mean)\n"
        + "\n".join(pair_lines)
        + "\n\nInspect the attached plots and deterministic findings, then return the JSON response."
    )
