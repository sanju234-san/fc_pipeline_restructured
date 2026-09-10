"""Tool: resolve_channel_selection (channel name fuzzy matching & validation)."""

import re
from typing import Any, Dict, List, Set, Tuple
from langchain_core.tools import tool

from fc_pipeline.config.thresholds import SUPERVISOR_CONFIDENCE_THRESHOLD


# Standard 10-20 system old-to-modern renamings (Section 6.1)
ALIAS_MAP_10_20: Dict[str, str] = {
    "T3": "T7", "T7": "T3",
    "T4": "T8", "T8": "T4",
    "T5": "P7", "P7": "T5",
    "T6": "P8", "P8": "T6",
}

# Standard 10-20 anatomical scalp region definitions
REGION_MAP: Dict[str, List[str]] = {
    "frontal": ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"],
    "central": ["C3", "C4", "CZ"],
    "motor": ["C3", "C4", "CZ"],
    "parietal": ["P3", "P4", "PZ"],
    "occipital": ["O1", "O2", "OZ"],
    "temporal": ["T7", "T8", "P7", "P8", "T3", "T4", "T5", "T6"],
}


def _clean_channel_name(ch: str) -> str:
    """Stage 1: Cleans channel strings by removing whitespace, prefixes (e.g., EEG), and montage suffixes."""
    cleaned = ch.strip().upper()
    cleaned = re.sub(r"^EEG\s*", "", cleaned)
    cleaned = re.sub(r"[-_](REF|LE)$", "", cleaned)
    return cleaned


@tool
def resolve_channel_selection(requested_channels_or_region: str, available_channels: List[str]) -> Dict[str, Any]:
    """Resolves requested channel labels or anatomical regions against available dataset channels using a 4-stage normalization.
    
    Executes: Clean -> Exact Match -> 10-20 Alias Match -> Region Mapping.
    Returns canonical resolved channel labels, per-channel confidence, and human input flags.
    """
    if not available_channels:
        return {
            "resolved_channels": [],
            "confidence": 0.0,
            "needs_human_input": True,
            "error": "Dataset channel list is empty.",
        }

    # Pre-map cleaned available channel names back to their original raw header labels
    clean_to_raw: Dict[str, str] = {_clean_channel_name(ch): ch for ch in available_channels}
    available_cleaned: Set[str] = set(clean_to_raw.keys())
    
    query = requested_channels_or_region.strip().lower()
    resolved_raw_channels: List[str] = []
    confidences: List[float] = []
    
    # -------------------------------------------------------------
    # Stage 4 Check: Region Mapping (e.g., "frontal", "motor cortex")
    # -------------------------------------------------------------
    matched_region: Optional[str] = None
    for region_name in REGION_MAP:
        if region_name in query:
            matched_region = region_name
            break
            
    if matched_region:
        candidate_channels = REGION_MAP[matched_region]
        for ch in candidate_channels:
            if ch in available_cleaned:
                resolved_raw_channels.append(clean_to_raw[ch])
                confidences.append(0.70)  # Heuristic region inference: 0.70 < 0.80 trips needs_human_input
    
    # -------------------------------------------------------------
    # Stages 1, 2, 3: Individual Channel Parsing & Alias Resolution
    # -------------------------------------------------------------
    else:
        # Split tokens on commas, spaces, or brackets
        raw_tokens = re.findall(r"[A-Za-z0-9]+", requested_channels_or_region)
        for token in raw_tokens:
            cleaned_req = _clean_channel_name(token)
            
            # Stage 2: Exact Match (Confidence = 1.0)
            if cleaned_req in available_cleaned:
                resolved_raw_channels.append(clean_to_raw[cleaned_req])
                confidences.append(1.0)
            
            # Stage 3: 10-20 Alias Match (Confidence = 0.85)
            elif cleaned_req in ALIAS_MAP_10_20 and ALIAS_MAP_10_20[cleaned_req] in available_cleaned:
                alias = ALIAS_MAP_10_20[cleaned_req]
                resolved_raw_channels.append(clean_to_raw[alias])
                confidences.append(0.85)  # 0.85 >= 0.80 passes without human flag
                
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_channels: List[str] = []
    for ch in resolved_raw_channels:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)
            
    min_confidence = min(confidences) if confidences else 0.0
    error: Optional[str] = None
    
    if not unique_channels:
        error = f"No requested channels or region matches found in available dataset channels."
        min_confidence = 0.0

    needs_human_input = (min_confidence < SUPERVISOR_CONFIDENCE_THRESHOLD) or (error is not None)

    return {
        "resolved_channels": unique_channels,
        "confidence": min_confidence,
        "needs_human_input": needs_human_input,
        "error": error,
    }
