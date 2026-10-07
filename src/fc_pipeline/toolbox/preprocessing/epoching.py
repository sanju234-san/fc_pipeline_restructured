"""Filtering, epoch segmentation, and epoch duration vs fmin checks.

Canonical implementation in the centralized EEG Toolbox.
Backward-compatible re-export shim:
    fc_pipeline.deterministic.data_prep.epoching
"""

from __future__ import annotations

import logging

import mne
import numpy as np

from fc_pipeline.schemas.data_prep_models import ValidatedDataPrepParams
from fc_pipeline.schemas.data_prep_contracts import DataPrepValidationError

logger = logging.getLogger(__name__)


def filter_and_epoch(
    raw: mne.io.BaseRaw,
    params: ValidatedDataPrepParams,
) -> mne.Epochs:
    """Bandpass-filter the recording and segment into epochs.

    Steps:
      1. Bandpass filter ``[fmin, fmax]`` Hz using MNE's FIR filter.
      2. Extract events from annotations matching ``params.condition``.
      3. Compute epoch duration as ``min_cycles / fmin`` (minimum length
         that provides ``min_cycles`` full cycles of the lowest frequency).
      4. Create fixed-length epochs from the condition-matched events.
      5. Validate that at least one epoch was produced.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        EEG recording after cleaning and re-referencing.
        Modified **in-place** (filtering).
    params : ValidatedDataPrepParams
        Validated parameters including ``fmin``, ``fmax``, ``condition``,
        ``min_cycles``.

    Returns
    -------
    epochs : mne.Epochs
        Epoched data ready for downstream connectivity analysis.

    Raises
    ------
    DataPrepValidationError
        If no events match the condition, or no valid epochs are produced.
    """
    # 1. Bandpass filter
    logger.info(
        "Applying bandpass filter: %.2f – %.2f Hz",
        params.fmin,
        params.fmax,
    )
    raw.filter(
        l_freq=params.fmin,
        h_freq=params.fmax,
        method="fir",
        verbose=False,
    )

    # 2. Extract events from annotations
    #    MNE's events_from_annotations returns events for each unique annotation
    #    description. We filter to keep only the requested condition.
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except ValueError:
        raise DataPrepValidationError(
            "NO_ANNOTATIONS",
            "The recording has no annotations to extract events from.",
        )

    if params.condition not in event_id:
        available = ", ".join(sorted(event_id.keys()))
        raise DataPrepValidationError(
            "CONDITION_NOT_IN_RECORDING",
            f"Condition '{params.condition}' not found in annotations. "
            f"Available: {available}.",
        )

    # Keep only events for the requested condition
    target_event_id = {params.condition: event_id[params.condition]}
    condition_mask = events[:, 2] == event_id[params.condition]
    condition_events = events[condition_mask]

    if len(condition_events) == 0:
        raise DataPrepValidationError(
            "NO_EVENTS_FOR_CONDITION",
            f"No events found for condition '{params.condition}'.",
        )

    # 3. Compute epoch duration from min_cycles constraint
    #    epoch_duration = min_cycles / fmin
    #    This ensures at least min_cycles full cycles of the lowest
    #    frequency fit within each epoch.
    epoch_duration = params.min_cycles / params.fmin
    logger.info(
        "Epoch duration: %.3f s (min_cycles=%.1f / fmin=%.2f Hz)",
        epoch_duration,
        params.min_cycles,
        params.fmin,
    )

    # Validate epoch duration against sampling rate
    sfreq = raw.info["sfreq"]
    min_samples = int(np.ceil(epoch_duration * sfreq))
    if min_samples < 2:
        raise DataPrepValidationError(
            "INVALID_EPOCH_DURATION",
            f"Computed epoch duration ({epoch_duration:.4f} s) is too short "
            f"at {sfreq} Hz sampling rate.",
        )

    # 4. Create epochs
    #    tmin=0, tmax=epoch_duration (epoch starts at event onset)
    #    baseline=None: no baseline correction (band-pass filter already applied)
    try:
        epochs = mne.Epochs(
            raw,
            events=condition_events,
            event_id=target_event_id,
            tmin=0.0,
            tmax=epoch_duration,
            baseline=None,
            preload=True,
            verbose=False,
        )
    except Exception as exc:
        raise DataPrepValidationError(
            "EPOCHING_FAILED",
            f"MNE Epochs creation failed: {type(exc).__name__}: {exc}",
        )

    # Drop bad epochs if any were auto-rejected
    epochs.drop_bad(verbose=False)

    # 5. Validate epoch count
    n_epochs = len(epochs)
    if n_epochs == 0:
        raise DataPrepValidationError(
            "NO_VALID_EPOCHS",
            "No valid epochs remain after segmentation. The epoch duration "
            f"({epoch_duration:.3f} s) may exceed the available annotation "
            f"segments for condition '{params.condition}'.",
        )

    logger.info(
        "Created %d epoch(s) of %.3f s for condition '%s'.",
        n_epochs,
        epoch_duration,
        params.condition,
    )

    return epochs
