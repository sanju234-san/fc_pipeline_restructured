"""Toolbox dataset inspection and parameter resolution tools."""

from fc_pipeline.toolbox.dataset.dataset_info import get_dataset_info
from fc_pipeline.toolbox.dataset.dataset_conditions import get_dataset_conditions
from fc_pipeline.toolbox.dataset.recording_facts import format_recording_facts, get_recording_facts
from fc_pipeline.toolbox.dataset.frequency_band import CANONICAL_BANDS, resolve_frequency_band
from fc_pipeline.toolbox.dataset.channel_selection import (
    ALIAS_MAP_10_20,
    REGION_MAP,
    STOP_WORDS,
    normalize_channel_label,
    resolve_channel_selection,
)

__all__ = [
    "get_dataset_info",
    "get_dataset_conditions",
    "get_recording_facts",
    "format_recording_facts",
    "CANONICAL_BANDS",
    "resolve_frequency_band",
    "ALIAS_MAP_10_20",
    "REGION_MAP",
    "STOP_WORDS",
    "normalize_channel_label",
    "resolve_channel_selection",
]
