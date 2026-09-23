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
