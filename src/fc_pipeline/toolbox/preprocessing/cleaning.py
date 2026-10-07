"""Variance-based flatline/bad-channel identification and channel removal.

Canonical implementation in the centralized EEG Toolbox.
Backward-compatible re-export shim:
    fc_pipeline.deterministic.data_prep.cleaning
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import mne
import numpy as np

from fc_pipeline.deterministic.data_prep.models import ValidatedDataPrepParams
from fc_pipeline.deterministic.data_prep.validation import DataPrepValidationError

logger = logging.getLogger(__name__)


def clean_bad_channels(
    raw: mne.io.BaseRaw,
    params: ValidatedDataPrepParams,
) -> Tuple[mne.io.BaseRaw, List[str]]:
    """Identify and remove flatline/bad channels from the recording.

    Steps:
      1. Pick only the channels listed in ``params.channels``.
      2. Compute per-channel variance over the full time range.
      3. Flag channels whose variance falls below
         ``params.bad_channel_variance_threshold`` (flatlines / dead channels).
      4. Drop flagged channels from the Raw object.
      5. Fail if fewer than 2 channels remain (connectivity needs ≥ 2).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded, unfiltered EEG recording.  Modified **in-place** (channel
        selection + bad-channel drop).
    params : ValidatedDataPrepParams
        Immutable validated parameters (channels, threshold, …).

    Returns
    -------
    raw : mne.io.BaseRaw
        The same Raw object after channel selection and bad-channel removal.
    dropped : list[str]
        Names of channels that were removed.

    Raises
    ------
    DataPrepValidationError
        If fewer than 2 channels survive cleaning.
    """
    # 1. Select only the requested channels
    raw.pick_channels(list(params.channels), ordered=True)
    logger.info(
        "Channel selection: kept %d / %d requested channels.",
        len(raw.ch_names),
        len(params.channels),
    )

    # 2. Compute variance per channel (full time range, all data loaded)
    data = raw.get_data()  # shape: (n_channels, n_samples)
    variances = np.var(data, axis=1)

    # 3. Identify bad channels
    bad_mask = variances < params.bad_channel_variance_threshold
    bad_channels: List[str] = [
        raw.ch_names[i] for i, is_bad in enumerate(bad_mask) if is_bad
    ]

    dropped: List[str] = []
    if bad_channels:
        logger.info(
            "Flagged %d bad/flatline channel(s): %s (threshold=%.2e)",
            len(bad_channels),
            ", ".join(bad_channels),
            params.bad_channel_variance_threshold,
        )
        # 4. Mark as bad and drop
        raw.info["bads"] = list(set(raw.info.get("bads", []) + bad_channels))
        raw.drop_channels(bad_channels)
        dropped = bad_channels
    else:
        logger.info("No bad/flatline channels detected.")

    # 5. Ensure enough channels remain
    if len(raw.ch_names) < 2:
        raise DataPrepValidationError(
            "INSUFFICIENT_CHANNELS",
            f"Only {len(raw.ch_names)} channel(s) remain after cleaning; "
            "connectivity analysis requires at least 2.",
        )

    return raw, dropped
