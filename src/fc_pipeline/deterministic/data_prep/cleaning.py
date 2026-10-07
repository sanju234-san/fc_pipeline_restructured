"""Backward-compatible re-export shim for cleaning.

The canonical implementation has moved to:
    fc_pipeline.toolbox.preprocessing.cleaning

All existing imports of this module continue to work unchanged.
"""

from fc_pipeline.toolbox.preprocessing.cleaning import clean_bad_channels  # noqa: F401

__all__ = ["clean_bad_channels"]
