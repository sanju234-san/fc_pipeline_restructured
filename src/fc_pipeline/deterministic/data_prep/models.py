"""Data Preparation execution contracts.

These models wrap the existing AnalysisPlan, ParameterManifestEntry,
and GraphState contracts. They do not replace or duplicate them.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan


class DataPrepInput(BaseModel):
    """Trusted input to the deterministic Data Preparation pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    raw_data_path: str = Field(..., min_length=1)
    plan: AnalysisPlan
    parameter_manifest: List[ParameterManifestEntry] = Field(default_factory=list)
    run_id: str = Field(..., min_length=1)

    # StrictBool: approval flags must be real booleans. Lax pydantic bool
    # coercion would turn values like "yes" or 1 into True (fail-open).
    gate_1_approved: StrictBool = False
    preflight_confirmed: StrictBool = False


# Reference methods Data Prep can execute. Deliberately only "average" (CAR, the
# design-doc default): mastoid/bipolar need electrode choices that neither the
# AnalysisPlan nor the manifest carries, so supporting them would mean inventing
# scientific parameters. Extend this Literal only together with the manifest.
ReferenceMethod = Literal["average"]


class ValidatedDataPrepParams(BaseModel):
    """Immutable, fully-resolved parameters produced ONLY by validation.

    Every downstream Data Prep module consumes this object instead of
    re-reading ``proposed_value`` strings from the manifest, so there is a
    single authoritative place where human-approved values are resolved.
    """

    model_config = ConfigDict(frozen=True)

    raw_data_path: Path
    run_id: str

    channels: Tuple[str, ...]
    condition: str
    fmin: float
    fmax: float

    reference_method: ReferenceMethod
    bad_channel_variance_threshold: float
    min_cycles: float


class DataPrepSummary(BaseModel):
    """Auditable summary of the deterministic Data Preparation run."""

    sampling_frequency: float
    original_channel_count: int
    selected_channel_count: int
    retained_channel_count: int

    dropped_channels: List[str] = Field(default_factory=list)

    reference_detected: Optional[str] = None
    reference_applied: Optional[str] = None

    condition: str

    epoch_count: int
    epoch_duration_seconds: float

    filter_l_freq: float
    filter_h_freq: float

    ica_applied: bool = False
    interpolation_applied: bool = False


class DataPrepResult(BaseModel):
    """Result returned by the deterministic Data Preparation pipeline."""

    success: bool

    preprocessed_data_path: Optional[str] = None

    bad_channels_dropped: List[str] = Field(default_factory=list)

    channel_plot_paths: Dict[str, str] = Field(default_factory=dict)

    summary: Optional[DataPrepSummary] = None

    error: Optional[str] = None
