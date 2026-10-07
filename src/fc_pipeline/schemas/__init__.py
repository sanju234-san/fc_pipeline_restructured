"""Re-exports all schema classes for clean package-level imports.

Public names are grouped by source module so importers can either use the
package-level shorthand (``from fc_pipeline.schemas import DataPrepInput``) or
import directly from the leaf submodule (``from
fc_pipeline.schemas.data_prep_models import DataPrepInput``).  Both forms are
supported and equivalent.
"""

# ---------------------------------------------------------------------------
# schemas.data_prep_models — Data Prep pydantic contracts (leaf package)
# ---------------------------------------------------------------------------
from fc_pipeline.schemas.data_prep_models import (  # noqa: F401
    DataPrepInput,
    DataPrepResult,
    DataPrepSummary,
    ReferenceMethod,
    ValidatedDataPrepParams,
)

# ---------------------------------------------------------------------------
# schemas.data_prep_contracts — Data Prep shared constants + typed exception
# ---------------------------------------------------------------------------
from fc_pipeline.schemas.data_prep_contracts import (  # noqa: F401
    DataPrepValidationError,
    REFERENCE_ALIASES,
    RECOGNISED_UNSUPPORTED_REFERENCES,
    REQUIRED_MANIFEST_PARAMETERS,
    SUPPORTED_OUTPUT_SUFFIXES,
    SUPPORTED_RAW_SUFFIXES,
    SUPPORTED_REFERENCE_METHODS,
)

__all__ = [
    # data_prep_models
    "DataPrepInput",
    "ValidatedDataPrepParams",
    "DataPrepSummary",
    "DataPrepResult",
    "ReferenceMethod",
    # data_prep_contracts
    "DataPrepValidationError",
    "REQUIRED_MANIFEST_PARAMETERS",
    "SUPPORTED_RAW_SUFFIXES",
    "SUPPORTED_OUTPUT_SUFFIXES",
    "SUPPORTED_REFERENCE_METHODS",
    "REFERENCE_ALIASES",
    "RECOGNISED_UNSUPPORTED_REFERENCES",
]

