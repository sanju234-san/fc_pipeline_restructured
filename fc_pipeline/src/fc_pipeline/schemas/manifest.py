"""ParameterManifestEntry data model with risk_tier field."""
# purpose : To define the data models for the parameter manifest.
from typing import Optional  # Optional for optional values.
from pydantic import BaseModel # Base model for creating data models.
class ParameterManifestEntry(BaseModel):
    """A parameter manifest entry with risk tier."""
    name: str # name of the parameter.
    category: str # category of the parameter.
    proposed_value: str # proposed value of the parameter.
    confidence: Optional[float] = None # confidence level of the proposed value.
    needs_human_input: bool = False # whether the proposed value needs human input.
    risk_tier: str = "low" # risk tier of the proposed value.
    human_approved_value: Optional[str] = None # human approved value of the proposed value.