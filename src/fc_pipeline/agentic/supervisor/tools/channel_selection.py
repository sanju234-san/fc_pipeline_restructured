"""Backward-compatibility re-export for resolve_channel_selection and helpers.

Original implementation has moved to fc_pipeline.toolbox.dataset.channel_selection.
"""

from fc_pipeline.toolbox.dataset.channel_selection import (
    ALIAS_MAP_10_20,
    REGION_MAP,
    STOP_WORDS,
    _clean_channel_name,
    normalize_channel_label,
    resolve_channel_selection,
)

__all__ = [
    "ALIAS_MAP_10_20",
    "REGION_MAP",
    "STOP_WORDS",
    "_clean_channel_name",
    "normalize_channel_label",
    "resolve_channel_selection",
]
