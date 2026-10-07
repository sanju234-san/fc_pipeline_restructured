"""Backward-compatible re-export shim for Data Prep contracts (models).

The canonical implementation has moved to:
    fc_pipeline.schemas.data_prep_models

All existing imports of this module continue to work unchanged.

Migration note (zero-logic change):
The 4 Pydantic models (DataPrepInput, ValidatedDataPrepParams, DataPrepSummary,
DataPrepResult) and the ``ReferenceMethod`` Literal previously lived here.
Moving them to the leaf schemas package removes the circular import between
``fc_pipeline.toolbox.validation.data_prep`` and
``fc_pipeline.deterministic.data_prep.__init__``.
"""

from fc_pipeline.schemas.data_prep_models import (  # noqa: F401
    DataPrepInput,
    DataPrepResult,
    DataPrepSummary,
    ReferenceMethod,
    ValidatedDataPrepParams,
)

__all__ = [
    "DataPrepInput",
    "ValidatedDataPrepParams",
    "DataPrepSummary",
    "DataPrepResult",
    "ReferenceMethod",
]
