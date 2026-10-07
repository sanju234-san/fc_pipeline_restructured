"""Backward-compatible re-export shim for validation.

Constants and exception class now live in the leaf contract package
(``fc_pipeline.schemas.data_prep_contracts``) to break the circular import
between ``fc_pipeline.toolbox.validation.data_prep`` and
``fc_pipeline.deterministic.data_prep.__init__``.

Functions (validate_data_prep_input, validate_plan_against_recording, etc.)
remain implemented in the canonical toolbox location:
    fc_pipeline.toolbox.validation.data_prep

All existing imports of this module continue to work unchanged.
"""

# Leaf contracts: imported first (before toolbox.*) so any partial-load order
# still sees the names populated.
from fc_pipeline.schemas.data_prep_contracts import (  # noqa: F401
    DataPrepValidationError,
    REFERENCE_ALIASES,
    RECOGNISED_UNSUPPORTED_REFERENCES,
    REQUIRED_MANIFEST_PARAMETERS,
    SUPPORTED_OUTPUT_SUFFIXES,
    SUPPORTED_RAW_SUFFIXES,
    SUPPORTED_REFERENCE_METHODS,
)

# Canonical toolbox functions (still live there; pure re-export).
from fc_pipeline.toolbox.validation.data_prep import (  # noqa: F401
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
    "SUPPORTED_REFERENCE_METHODS",
    "REFERENCE_ALIASES",
    "RECOGNISED_UNSUPPORTED_REFERENCES",
    "validate_run_id",
    "build_safe_output_path",
    "validate_raw_data_path",
    "validate_data_prep_input",
    "validate_plan_against_recording",
]
