"""Backward-compatible re-export shim for referencing.

The canonical implementation has moved to:
    fc_pipeline.toolbox.preprocessing.referencing

All existing imports of this module continue to work unchanged.
"""

from fc_pipeline.toolbox.preprocessing.referencing import apply_reference  # noqa: F401

__all__ = ["apply_reference"]
