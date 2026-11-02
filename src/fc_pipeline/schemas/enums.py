"""MetricEnum, EvidenceCategoryEnum, VisualTopologyReadEnum, OverallVerdictEnum."""

from enum import Enum


class MetricEnum(str, Enum):
    """Canonical string identifiers for the five supported connectivity metrics."""

    PLI = "pli"
    WPLI = "wpli"
    IMAGINARY_COHERENCE = "imcoh"
    PLV = "plv"
    COHERENCE = "coh"


class EvidenceCategoryEnum(str, Enum):
    """Deterministic 4-category classification profiles based on metric synthesis."""

    FULLY_CONVERGENT = "FULLY CONVERGENT"
    CONFIRMED_COUPLING = "CONFIRMED COUPLING"
    ZERO_LAG_UNRESOLVED = "ZERO-LAG COUPLING, CAUSE UNRESOLVED"
    NO_COUPLING = "NO COUPLING"


class VisualTopologyReadEnum(str, Enum):
    """Categorical visual topology reads produced by the multimodal evaluator."""

    PLAUSIBLE_NEURAL_TOPOLOGY = "plausible_neural_topology"
    CONSISTENT_WITH_ARTIFACT = "consistent_with_artifact"
    INDETERMINATE = "indeterminate"


class OverallVerdictEnum(str, Enum):
    """Overall pipeline scientific validity verdict."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
