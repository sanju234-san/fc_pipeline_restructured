"""FrequencyBand and AnalysisPlan data models."""
# purpose : To define the data models for the analysis plan and frequency band.
from typing import List  # Standard typing import for declaring typed lists.
from pydantic import BaseModel , Field # Pydantic BaseModel for creating data models, Field for validation.
from fc_pipeline.schemas.enums import MetricEnum # Enum for metrics.

class FrequencyBand(BaseModel): # this class is for defining the frequency band.
    """A frequency band specification for analysis, with validation."""
    name : str # name of the frequency band.
    fmin: float = Field(..., gt=0.0, description="Lower frequency boundary in Hz") # lower frequency boundary in Hz.
    fmax: float = Field(..., gt=0.0, description="Upper frequency boundary in Hz") # upper frequency boundary in Hz.
class AnalysisPlan(BaseModel): # this class is for defining the analysis plan.
    """An analysis plan specification for analysis, with validation."""
    metrics: List[MetricEnum] = Field(
        default_factory=lambda: [
            MetricEnum.PLI,
            MetricEnum.WPLI,
            MetricEnum.IMAGINARY_COHERENCE,
            MetricEnum.PLV,
            MetricEnum.COHERENCE,
        ],
        description="List of metrics to compute. Defaults to all five foundational metrics to enable complementary evidence synthesis.",
    ) # default factory is used to create a new list for each instance of the class.  
    freq_band: FrequencyBand # frequency band to be analyzed.
    channels: List[str] # list of channels to be analyzed.
    condition: str # condition to be analyzed.