"""Preprocessing toolbox — cleaning, referencing, and epoching.

Canonical implementations for EEG signal preprocessing steps.
Agents and deterministic pipelines should import from here.
Backward-compatible re-exports remain at the original
``fc_pipeline.deterministic.data_prep.*`` locations.
"""

from fc_pipeline.toolbox.preprocessing.cleaning import clean_bad_channels
from fc_pipeline.toolbox.preprocessing.epoching import filter_and_epoch
from fc_pipeline.toolbox.preprocessing.referencing import apply_reference

__all__ = [
    "clean_bad_channels",
    "apply_reference",
    "filter_and_epoch",
]
