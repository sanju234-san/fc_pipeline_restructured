"""PairEvidenceSynthesis data model for 4-category classification."""

from typing import List, Optional
from pydantic import BaseModel, Field

from fc_pipeline.schemas.enums import EvidenceCategoryEnum, VisualTopologyReadEnum


class PairEvidenceSynthesis(BaseModel):
    """Pairwise evidence synthesis classification across metric groups."""

    channel_pair: List[str] = Field(..., description="Pair of channel labels, e.g. ['F3', 'F4']")
    phase_robust_mean: float = Field(..., description="Mean of computed phase-robust metrics (PLI, wPLI, |ImCoh|)")
    zero_lag_mean: float = Field(..., description="Mean of computed zero-lag-inclusive metrics (PLV, Coherence)")
    classification: EvidenceCategoryEnum = Field(..., description="Deterministic 4-category classification from Node 3")
    evaluator_visual_read: Optional[VisualTopologyReadEnum] = Field(
        default=None,
        description="Supporting visual observation for top-N unresolved pairs from Evaluator",
    )
    evaluator_visual_note: Optional[str] = Field(
        default=None,
        description="Supporting visual context note",
    )
