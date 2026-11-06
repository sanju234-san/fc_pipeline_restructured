"""VLM API payload builder, base64 image encoder, and response parser.

Follows Section 6.4 Step 3 of PIPELINE_PROTOTYPE_DESIGN.md.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fc_pipeline.schemas.enums import VisualTopologyReadEnum
from fc_pipeline.schemas.evaluation import ArtifactSignatures
from fc_pipeline.schemas.evidence import PairEvidenceSynthesis

_FRONTAL_ARTIFACT_CHANNELS = {"FP1", "FP2", "AF3", "AF4", "FPZ", "AF7", "AF8", "F7", "F8"}


def encode_image_base64(image_path: str | Path) -> Optional[str]:
    """Reads a PNG file and returns its base64-encoded string, or None if missing."""
    try:
        p = Path(str(image_path))
        if not p.is_file():
            return None
        return base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        return None


def build_vlm_messages(
    system_prompt: str,
    user_prompt: str,
    image_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Constructs multimodal message dicts containing text and base64 PNG data URIs."""
    content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for img_path in (image_paths or [])[:6]:
        b64 = encode_image_base64(img_path)
        if b64:
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_blocks},
    ]


def _coerce_topology_enum(raw_read: str) -> VisualTopologyReadEnum:
    val = str(raw_read or "").strip().lower()
    if val in {"plausible_neural_topology", "plausible_neural"}:
        return VisualTopologyReadEnum.PLAUSIBLE_NEURAL_TOPOLOGY
    if val in {"consistent_with_artifact", "artifact"}:
        return VisualTopologyReadEnum.CONSISTENT_WITH_ARTIFACT
    return VisualTopologyReadEnum.INDETERMINATE


def fallback_topology_reads(
    top_pairs: List[PairEvidenceSynthesis],
    artifacts: ArtifactSignatures,
) -> Tuple[List[PairEvidenceSynthesis], str, List[str]]:
    """Deterministic fallback when VLM is offline or returns unparseable output."""
    flagged_chs = {
        c.upper()
        for c in (
            artifacts.ocular_contamination.affected_channels
            + artifacts.muscle_contamination.affected_channels
            + artifacts.electrode_pop.affected_channels
        )
    }
    updated: List[PairEvidenceSynthesis] = []
    for p in top_pairs:
        ch_a, ch_b = p.channel_pair[0], p.channel_pair[1]
        pair_upper = {ch_a.upper(), ch_b.upper()}
        if pair_upper & flagged_chs or pair_upper <= _FRONTAL_ARTIFACT_CHANNELS:
            read = VisualTopologyReadEnum.CONSISTENT_WITH_ARTIFACT
            note = (
                f"Zero-lag coupling ({p.zero_lag_mean:.3f}) involves fronto-polar or artifact-flagged "
                f"electrodes ({ch_a}-{ch_b}); spatial proximity suggests volume conduction or ocular/muscle leakage."
            )
        else:
            read = VisualTopologyReadEnum.INDETERMINATE
            note = (
                f"Zero-lag coupling ({p.zero_lag_mean:.3f}) on {ch_a}-{ch_b} lacks phase-lagged confirmation "
                f"({p.phase_robust_mean:.3f}); sensor-level topology alone cannot separate volume conduction from zero-lag synchrony."
            )
        updated.append(
            p.model_copy(
                update={
                    "evaluator_visual_read": read,
                    "evaluator_visual_note": note,
                }
            )
        )

    obs = (
        f"Evaluated {len(updated)} top unresolved zero-lag channel pair(s) alongside deterministic "
        "sanity and artifact checks."
    )
    recs = [
        "Interpret CONFIRMED COUPLING and FULLY CONVERGENT pairs as primary phase-lagged connectivity candidates.",
        "Treat ZERO-LAG COUPLING, CAUSE UNRESOLVED pairs cautiously; sensor-level zero-lag synchrony may reflect volume conduction.",
    ]
    if flagged_chs:
        recs.append(
            f"Inspect artifact-implicated channels ({', '.join(sorted(flagged_chs))}) before drawing regional conclusions."
        )
    return updated, obs, recs


def parse_vlm_topology_response(
    raw_text: str,
    top_pairs: List[PairEvidenceSynthesis],
    artifacts: ArtifactSignatures,
) -> Tuple[List[PairEvidenceSynthesis], str, List[str]]:
    """Parses VLM JSON output and attaches visual reads to the Top-5 unresolved pairs."""
    if not raw_text or not raw_text.strip():
        return fallback_topology_reads(top_pairs, artifacts)

    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return fallback_topology_reads(top_pairs, artifacts)
    except Exception:
        return fallback_topology_reads(top_pairs, artifacts)

    read_map: Dict[Tuple[str, str], Tuple[VisualTopologyReadEnum, str]] = {}
    for item in data.get("pair_reads") or []:
        if not isinstance(item, dict):
            continue
        pair = item.get("pair")
        if isinstance(pair, list) and len(pair) == 2:
            key = (str(pair[0]).upper(), str(pair[1]).upper())
            r_enum = _coerce_topology_enum(str(item.get("read", "indeterminate")))
            note = str(item.get("note") or "Visual topology read recorded by Evaluator.")
            read_map[key] = (r_enum, note)
            read_map[(key[1], key[0])] = (r_enum, note)

    fb_pairs, fb_obs, fb_recs = fallback_topology_reads(top_pairs, artifacts)
    updated: List[PairEvidenceSynthesis] = []
    for orig, fb in zip(top_pairs, fb_pairs):
        key = (orig.channel_pair[0].upper(), orig.channel_pair[1].upper())
        if key in read_map:
            r_enum, note = read_map[key]
            updated.append(
                orig.model_copy(
                    update={
                        "evaluator_visual_read": r_enum,
                        "evaluator_visual_note": note,
                    }
                )
            )
        else:
            updated.append(fb)

    obs = str(data.get("visual_observations") or fb_obs).strip()
    raw_recs = data.get("recommendations_for_user")
    recs = (
        [str(r).strip() for r in raw_recs if str(r).strip()]
        if isinstance(raw_recs, list) and raw_recs
        else fb_recs
    )
    return updated, obs, recs
