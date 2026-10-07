"""Validation toolbox — deterministic, fail-closed EEG pipeline validation.

Canonical implementations for Data Preparation input validation, recording
metadata checks, path validation, and safe output path construction.
Backward-compatible re-export shim:
    fc_pipeline.deterministic.data_prep.validation
"""

from fc_pipeline.toolbox.validation.data_prep import (
    REQUIRED_MANIFEST_PARAMETERS,
    SUPPORTED_OUTPUT_SUFFIXES,
    SUPPORTED_RAW_SUFFIXES,
    DataPrepValidationError,
    build_safe_output_path,
    validate_data_prep_input,
    validate_plan_against_recording,
    validate_raw_data_path,
    validate_run_id,
)

__all__ = [
    "DataPrepValidationError",
    "REQUIRED_MANIFEST_PARAMETERS",
    "SUPPORTED_RAW_SUFFIXES",
    "SUPPORTED_OUTPUT_SUFFIXES",
    "validate_run_id",
    "build_safe_output_path",
    "validate_raw_data_path",
    "validate_data_prep_input",
    "validate_plan_against_recording",
]
