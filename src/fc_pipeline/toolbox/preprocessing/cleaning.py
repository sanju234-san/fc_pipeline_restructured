"""Variance-based flatline/bad-channel identification and channel removal.

Canonical implementation in the centralized EEG Toolbox.
Backward-compatible re-export shim:
    fc_pipeline.deterministic.data_prep.cleaning
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import mne
import numpy as np

from fc_pipeline.deterministic.data_prep.models import ValidatedDataPrepParams
from fc_pipeline.deterministic.data_prep.validation import DataPrepValidationError

logger = logging.getLogger(__name__)


def mark_flatline_channels(
    raw: mne.io.BaseRaw,
    params: ValidatedDataPrepParams,
) -> List[str]:
    """Flag flatline/dead EEG channels across the FULL montage, before referencing.

    Why this exists
    ---------------
    The average reference must be computed over the whole recorded montage,
    not over the handful of channels selected for analysis, and it must not
    include dead channels.  A flatline channel also stops being a flatline
    the moment it is re-referenced (it becomes the negated mean of the other
    channels), so flatline screening has to happen on the *unreferenced*
    signal.  This function therefore:

      1. Computes per-channel variance for every EEG channel.
      2. Adds channels below ``params.bad_channel_variance_threshold`` to
         ``raw.info["bads"]`` (MNE excludes ``bads`` from the average reference).
      3. Does NOT drop anything; dropping happens in :func:`clean_bad_channels`.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded, unreferenced recording (all channels still present).
        ``raw.info["bads"]`` is updated **in-place**.
    params : ValidatedDataPrepParams
        Supplies ``bad_channel_variance_threshold``.

    Returns
    -------
    flagged : list[str]
        Channels newly flagged as flatline (pass this to
        :func:`clean_bad_channels` as ``pre_flagged``).
    """
    eeg_idx = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_idx) == 0:
        return []

    variances = np.var(raw.get_data(picks=eeg_idx), axis=1)
    flagged: List[str] = [
        raw.ch_names[int(i)]
        for i, var in zip(eeg_idx, variances)
        if var < params.bad_channel_variance_threshold
    ]

    if flagged:
        existing = list(raw.info.get("bads", []))
        raw.info["bads"] = existing + [c for c in flagged if c not in existing]
        logger.info(
            "Pre-reference screening flagged %d flatline channel(s) across the "
            "full montage: %s (threshold=%.2e). They are excluded from the "
            "average reference.",
            len(flagged),
            ", ".join(flagged),
            params.bad_channel_variance_threshold,
        )
    else:
        logger.info("Pre-reference screening: no flatline channels in the full montage.")

    return flagged


def clean_bad_channels(
    raw: mne.io.BaseRaw,
    params: ValidatedDataPrepParams,
    pre_flagged: Optional[Sequence[str]] = None,
) -> Tuple[mne.io.BaseRaw, List[str]]:
    """Identify and remove flatline/bad channels from the recording.

    Steps:
      1. Pick only the channels listed in ``params.channels``.
      2. Compute per-channel variance over the full time range.
      3. Flag channels whose variance falls below
         ``params.bad_channel_variance_threshold`` (flatlines / dead channels),
         plus any channel in ``pre_flagged``.
      4. Drop flagged channels from the Raw object.
      5. Fail if fewer than 2 channels remain (connectivity needs ≥ 2).

    In the executor this runs AFTER referencing: flatline channels are
    flagged on the unreferenced full montage by :func:`mark_flatline_channels`
    (a dead channel is no longer flat once re-referenced), and passed in via
    ``pre_flagged``.  Called directly on unreferenced data without
    ``pre_flagged`` it behaves exactly as before.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded, unfiltered EEG recording.  Modified **in-place** (channel
        selection + bad-channel drop).
    params : ValidatedDataPrepParams
        Immutable validated parameters (channels, threshold, …).
    pre_flagged : sequence of str, optional
        Channels already flagged as flatline before referencing.  Those that
        are among the selected channels are dropped even though their
        post-reference variance is no longer below the threshold.

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
    pre = set(pre_flagged or ())
    bad_channels: List[str] = [
        raw.ch_names[i]
        for i, is_bad in enumerate(bad_mask)
        if is_bad or raw.ch_names[i] in pre
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
