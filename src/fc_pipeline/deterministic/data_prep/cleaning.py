"""Backward-compatible re-export shim for cleaning.

The canonical implementation has moved to:
    fc_pipeline.toolbox.preprocessing.cleaning

All existing imports of this module continue to work unchanged.
"""

from fc_pipeline.toolbox.preprocessing.cleaning import (  # noqa: F401
    clean_bad_channels,
    mark_flatline_channels,
)

__all__ = ["clean_bad_channels", "mark_flatline_channels"]
