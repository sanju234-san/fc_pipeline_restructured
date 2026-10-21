"""Section 6.1 hardcoded lookup table mapping metric aliases to MetricEnum."""
from typing import Dict, TypedDict

from fc_pipeline.schemas.enums import MetricEnum


class MetricInfo(TypedDict):
    canonical_id: MetricEnum
    group: str


METRIC_LOOKUP: Dict[str, MetricInfo] = {
    # Phase Lag Index (Phase-Robust)
    "pli": {"canonical_id": MetricEnum.PLI, "group": "Phase-Robust"},
    "phase lag index": {"canonical_id": MetricEnum.PLI, "group": "Phase-Robust"},
    "phase-lag index": {"canonical_id": MetricEnum.PLI, "group": "Phase-Robust"},
    # Weighted Phase Lag Index (Phase-Robust)
    "wpli": {"canonical_id": MetricEnum.WPLI, "group": "Phase-Robust"},
    "weighted phase lag index": {"canonical_id": MetricEnum.WPLI, "group": "Phase-Robust"},
    "weighted pli": {"canonical_id": MetricEnum.WPLI, "group": "Phase-Robust"},
    # Imaginary Coherence (Phase-Robust)
    "imcoh": {"canonical_id": MetricEnum.IMAGINARY_COHERENCE, "group": "Phase-Robust"},
    "imaginary coherence": {"canonical_id": MetricEnum.IMAGINARY_COHERENCE, "group": "Phase-Robust"},
    "icoh": {"canonical_id": MetricEnum.IMAGINARY_COHERENCE, "group": "Phase-Robust"},
    "imaginary part of coherence": {"canonical_id": MetricEnum.IMAGINARY_COHERENCE, "group": "Phase-Robust"},
    # Phase Locking Value (Zero-Lag-Inclusive)
    "plv": {"canonical_id": MetricEnum.PLV, "group": "Zero-Lag-Inclusive"},
    "phase locking value": {"canonical_id": MetricEnum.PLV, "group": "Zero-Lag-Inclusive"},
    "phase lock": {"canonical_id": MetricEnum.PLV, "group": "Zero-Lag-Inclusive"},
    # Spectral Coherence (Zero-Lag-Inclusive)
    "coh": {"canonical_id": MetricEnum.COHERENCE, "group": "Zero-Lag-Inclusive"},
    "coherence": {"canonical_id": MetricEnum.COHERENCE, "group": "Zero-Lag-Inclusive"},
    "spectral coherence": {"canonical_id": MetricEnum.COHERENCE, "group": "Zero-Lag-Inclusive"},
}

CANONICAL_METRICS = [m.value for m in MetricEnum]
METRIC_ALIASES = list(METRIC_LOOKUP.keys())
METRIC_GROUPS = {
    MetricEnum.PLI: "Phase-Robust",
    MetricEnum.WPLI: "Phase-Robust",
    MetricEnum.IMAGINARY_COHERENCE: "Phase-Robust",
    MetricEnum.PLV: "Zero-Lag-Inclusive",
    MetricEnum.COHERENCE: "Zero-Lag-Inclusive",
}


def canonicalize_metric(metric_str: str):
    """Normalizes a raw user input string to a canonical MetricEnum, or None if unsupported."""
    clean = metric_str.strip().lower()
    info = METRIC_LOOKUP.get(clean)
    if info:
        return info["canonical_id"]
    return None

