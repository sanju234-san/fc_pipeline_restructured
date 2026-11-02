"""SanityCheckFlags, ArtifactDetectionResult, ArtifactSignatures, EvaluationVerdict."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from fc_pipeline.schemas.enums import OverallVerdictEnum
from fc_pipeline.schemas.evidence import PairEvidenceSynthesis


class SanityCheckFlags(BaseModel):
    """Sanity assessment flags for a single connectivity metric matrix."""

    values_in_bounds: bool = Field(..., description="All values within [0, 1] (holds for |imcoh|)")
    matrix_is_symmetric: bool = Field(..., description="M = M^T verified for non-directional metrics")
    is_sparse_or_flat: bool = Field(..., description="Flags SNR collapse (all near 0 or all near 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0.0-1.0 confidence in sanity assessment")


class ArtifactDetectionResult(BaseModel):
    """Result of a single deterministic artifact heuristic check."""

    detected: bool = Field(..., description="True if the artifact pattern was found")
    affected_channels: List[str] = Field(default_factory=list, description="Channels implicated")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0.0-1.0 confidence in heuristic detection")
    detail: str = Field(..., description="Human-readable description of measured values vs threshold")


class ArtifactSignatures(BaseModel):
    """Deterministic spatial/statistical artifact detection results."""

    ocular_contamination: ArtifactDetectionResult
    muscle_contamination: ArtifactDetectionResult
    electrode_pop: ArtifactDetectionResult
    top_unresolved_visual_reads: List[PairEvidenceSynthesis] = Field(default_factory=list)


class EvaluationVerdict(BaseModel):
    """Complete evaluation verdict combining sanity, artifacts, and narrative."""

    overall_verdict: OverallVerdictEnum
    per_metric_sanity: Dict[str, SanityCheckFlags]
    artifacts: ArtifactSignatures
    visual_observations: str
    recommendations_for_user: List[str] = Field(default_factory=list)
