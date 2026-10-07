"""Backward-compatibility re-export for resolve_frequency_band and CANONICAL_BANDS.

Original implementation has moved to fc_pipeline.toolbox.dataset.frequency_band.
"""

from fc_pipeline.toolbox.dataset.frequency_band import (
    CANONICAL_BANDS,
    resolve_frequency_band,
)

__all__ = ["CANONICAL_BANDS", "resolve_frequency_band"]