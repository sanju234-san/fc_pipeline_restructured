"""Backward-compatibility re-export for get_dataset_info.

Original implementation has moved to fc_pipeline.toolbox.dataset.dataset_info.
"""

from fc_pipeline.toolbox.dataset.dataset_info import get_dataset_info

__all__ = ["get_dataset_info"]
