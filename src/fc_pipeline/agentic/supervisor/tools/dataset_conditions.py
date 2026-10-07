"""Backward-compatibility re-export for get_dataset_conditions.

Original implementation has moved to fc_pipeline.toolbox.dataset.dataset_conditions.
"""

from fc_pipeline.toolbox.dataset.dataset_conditions import get_dataset_conditions

__all__ = ["get_dataset_conditions"]
