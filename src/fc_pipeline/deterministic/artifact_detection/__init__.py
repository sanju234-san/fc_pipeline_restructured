"""Re-exports statistical artifact detection routines."""

from fc_pipeline.deterministic.artifact_detection.heuristics import (
    detect_electrode_pop,
    detect_muscle_contamination,
    detect_ocular_contamination,
    run_artifact_heuristics,
)

__all__ = [
    "detect_ocular_contamination",
    "detect_muscle_contamination",
    "detect_electrode_pop",
    "run_artifact_heuristics",
]
