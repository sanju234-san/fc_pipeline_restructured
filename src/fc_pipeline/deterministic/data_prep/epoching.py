"""Backward-compatible re-export shim for epoching.

The canonical implementation has moved to:
    fc_pipeline.toolbox.preprocessing.epoching

All existing imports of this module continue to work unchanged.
"""

from fc_pipeline.toolbox.preprocessing.epoching import filter_and_epoch  # noqa: F401

__all__ = ["filter_and_epoch"]
