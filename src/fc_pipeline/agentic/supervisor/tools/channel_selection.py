"""Tool: resolve_channel_selection (channel name fuzzy matching & validation)."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
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

# Common English / syntax filler tokens to ignore when parsing channel lists
STOP_WORDS: Set[str] = {
    "", "AND", "OR", "CHANNELS", "CHANNEL", "ELECTRODES", "ELECTRODE",
    "REGION", "REGIONS", "LOBES", "LOBE", "THE", "EEG", "RECORDING",
    "ACROSS", "BETWEEN", "FOR", "IN", "ON", "OF", "WITH", "REF", "LE",
}


def normalize_channel_label(ch: str) -> str:
    """Return a canonical electrode label for matching user input to EEG headers.

    EEG/EDF exports commonly decorate the same electrode with transport/montage
    syntax such as ``EEG C3-REF``, ``C3-REF``, or a trailing period (``C3.``).
    Those decorations must not make an otherwise exact electrode lookup fail.
    The original/raw header is still preserved for the value passed to MNE.
    """
    cleaned = str(ch).strip(" \t\n\r,;[](){}").upper()
    cleaned = re.sub(r"^EEG\s*", "", cleaned)
    cleaned = re.sub(r"[-_]((REF|LE))$", "", cleaned)
    # Some EDF exports append a period to every electrode name (C3., FC5., ...).
    cleaned = re.sub(r"\.+$", "", cleaned)
    return cleaned.strip()


# Backwards-compatible private name used internally/tests from older revisions.
_clean_channel_name = normalize_channel_label


@tool
def resolve_channel_selection(requested_channels_or_region: str, available_channels: List[str]) -> Dict[str, Any]:
    """Resolves requested channel labels or anatomical regions against available dataset channels using a 4-stage normalization.
    
    Executes: Clean -> Exact Match -> 10-20 Alias Match -> Region Mapping.
    Returns canonical resolved channel labels, per-channel confidence, and human input flags.
    Never silently drops an unresolvable channel.
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
    unresolved_tokens: List[str] = []
    
    # -------------------------------------------------------------
    # Stage 4 Check: Region Mapping (e.g., "frontal", "motor cortex")
    # -------------------------------------------------------------
    matched_regions: List[str] = [
        region_name for region_name in REGION_MAP if region_name in query
    ]
            
    if matched_regions:
        for matched_region in matched_regions:
            candidate_channels = REGION_MAP[matched_region]
            for ch in candidate_channels:
                if ch in available_cleaned:
                    resolved_raw_channels.append(clean_to_raw[ch])
                    confidences.append(0.70)  # Heuristic region inference: 0.70 < 0.80 trips needs_human_input
    
    # -------------------------------------------------------------
    # Stages 1, 2, 3: Individual Channel Parsing & Alias Resolution
    # -------------------------------------------------------------
    else:
        # Split tokens on commas, semicolons, or whitespace (preserving hyphenated suffixes like F3-REF)
        raw_tokens = [t.strip(" \t\n\r,;[](){}") for t in re.split(r"[,;\s]+", requested_channels_or_region) if t.strip()]
        for token in raw_tokens:
            if token.upper() in STOP_WORDS:
                continue

            cleaned_req = _clean_channel_name(token)
            if not cleaned_req or cleaned_req in STOP_WORDS:
                continue
            
            # Stage 2: Exact Match (Confidence = 1.0)
            if cleaned_req in available_cleaned:
                resolved_raw_channels.append(clean_to_raw[cleaned_req])
                confidences.append(1.0)
            
            # Stage 3: 10-20 Alias Match (Confidence = 0.85)
            elif cleaned_req in ALIAS_MAP_10_20 and ALIAS_MAP_10_20[cleaned_req] in available_cleaned:
                alias = ALIAS_MAP_10_20[cleaned_req]
                resolved_raw_channels.append(clean_to_raw[alias])
                confidences.append(0.85)  # 0.85 >= 0.80 passes without human flag
                
            else:
                # Unresolved token — cannot be mapped to any available dataset channel or alias
                unresolved_tokens.append(token)
                confidences.append(0.0)
                
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_channels: List[str] = []
    for ch in resolved_raw_channels:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)
            
    min_confidence = min(confidences) if confidences else 0.0
    error: Optional[str] = None
    
    if unresolved_tokens:
        error = f"Unrecognized channel label(s) not found in dataset: {', '.join(unresolved_tokens)}."
        min_confidence = 0.0
    elif not unique_channels:
        error = f"No requested channels or region matches found in available dataset channels."
        min_confidence = 0.0

    needs_human_input = (min_confidence < SUPERVISOR_CONFIDENCE_THRESHOLD) or (error is not None)

    return {
        "resolved_channels": unique_channels,
        "confidence": min_confidence,
        "needs_human_input": needs_human_input,
        "unresolved_channels": unresolved_tokens,
        "error": error,
    }
