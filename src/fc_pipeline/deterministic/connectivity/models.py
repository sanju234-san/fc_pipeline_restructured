"""Models and contracts for deterministic connectivity execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from fc_pipeline.schemas.evidence import PairEvidenceSynthesis
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan


class ConnectivityInput(BaseModel):
    """Input payload required for deterministic connectivity analysis."""

    preprocessed_data_path: str = Field(..., description="Path to preprocessed epoched FIF file")
    plan: AnalysisPlan = Field(..., description="Locked scientific analysis plan")
    parameter_manifest: Optional[List[ParameterManifestEntry]] = Field(
        default_factory=list,
        description="Gate 1 manifest containing threshold entries",
    )
    run_id: str = Field(..., description="Shared pipeline run identifier")


class ConnectivityResult(BaseModel):
    """Complete result returned by the connectivity analysis executor."""

    success: bool = Field(..., description="Whether connectivity computation completed successfully")
    metric_csv_paths: Dict[str, str] = Field(default_factory=dict, description="Metric name to CSV path")
    heatmap_image_paths: Dict[str, str] = Field(default_factory=dict, description="Metric name to heatmap PNG")
    network_image_paths: Dict[str, str] = Field(default_factory=dict, description="Metric name to network PNG")
    numerical_summaries: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Summary stats per metric")
    evidence_summary: List[PairEvidenceSynthesis] = Field(default_factory=list, description="List of pairwise syntheses")
    consolidated_report_path: Optional[str] = Field(default=None, description="Path to consolidated connectivity report CSV")
    error: Optional[str] = Field(default=None, description="Error message if success=False")
