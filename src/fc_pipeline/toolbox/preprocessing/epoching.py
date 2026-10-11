"""Filtering, epoch segmentation, and epoch duration vs fmin checks.

Canonical implementation in the centralized EEG Toolbox.
Backward-compatible re-export shim:
    fc_pipeline.deterministic.data_prep.epoching
"""

from __future__ import annotations

import logging

import mne
import numpy as np

from fc_pipeline.deterministic.data_prep.models import ValidatedDataPrepParams
from fc_pipeline.deterministic.data_prep.validation import DataPrepValidationError

logger = logging.getLogger(__name__)


def filter_and_epoch(
    raw: mne.io.BaseRaw,
    params: ValidatedDataPrepParams,
) -> mne.Epochs:
    """Bandpass-filter the recording and segment into epochs.

    Steps:
      1. Bandpass filter ``[fmin, fmax]`` Hz using MNE's FIR filter.
      2. Extract events from annotations matching ``params.condition``.
      3. Compute the epoch length as ``min_cycles / fmin`` rounded UP to a
         whole number of samples (so every epoch holds at least
         ``min_cycles`` full cycles of the lowest frequency).
      4. Build the epochs from the annotated condition segments:

         * a segment with a real duration (more than ~1 sample) is tiled
           with consecutive, non-overlapping epochs that lie fully inside
           the segment (``floor(duration / epoch_length)`` epochs);
         * a segment shorter than one epoch cannot host an epoch and is
           skipped (and counted in the log);
         * an instantaneous marker (duration of at most ~1 sample, e.g. a
           stimulus trigger) keeps the original behaviour: one epoch that
           starts at the marker.

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

    # 3. Compute the epoch length from the min_cycles constraint
    #    epoch_duration = min_cycles / fmin, rounded UP to whole samples so the
    #    final epoch can never contain fewer than min_cycles cycles of fmin.
    epoch_duration = params.min_cycles / params.fmin
    sfreq = raw.info["sfreq"]
    window_samples = int(np.ceil(epoch_duration * sfreq - 1e-9))
    if window_samples < 2:
        raise DataPrepValidationError(
            "INVALID_EPOCH_DURATION",
            f"Computed epoch duration ({epoch_duration:.4f} s) is too short "
            f"at {sfreq} Hz sampling rate.",
        )
    window_seconds = window_samples / sfreq
    logger.info(
        "Epoch length: %.3f s = %d samples (min_cycles=%.1f / fmin=%.2f Hz).",
        window_seconds,
        window_samples,
        params.min_cycles,
        params.fmin,
    )

    # 4. Build epoch start positions from the annotated condition segments.
    #    Annotations are kept sorted by onset by MNE, and
    #    ``events_from_annotations`` emits events in the same order, so the
    #    k-th condition event corresponds to the k-th condition annotation.
    annotations = raw.annotations
    segment_durations = np.asarray(annotations.duration)[
        np.asarray(annotations.description) == params.condition
    ]
    if len(segment_durations) != len(condition_events):
        # Defensive fallback: durations cannot be aligned with events, so
        # keep the original onset-only behaviour instead of guessing.
        logger.warning(
            "Condition '%s': %d event(s) but %d annotation duration(s); "
            "falling back to one epoch per event onset.",
            params.condition,
            len(condition_events),
            len(segment_durations),
        )
        segment_durations = np.zeros(len(condition_events))

    marker_max_seconds = 1.5 / sfreq  # treated as an instantaneous marker
    starts: list[int] = []
    n_markers = 0
    n_segments_used = 0
    segments_too_short: list[float] = []

    for onset_sample, duration in zip(condition_events[:, 0], segment_durations):
        if duration <= marker_max_seconds:
            starts.append(int(onset_sample))
            n_markers += 1
            continue
        segment_samples = int(np.floor(duration * sfreq + 1e-9))
        n_windows = segment_samples // window_samples
        if n_windows == 0:
            segments_too_short.append(float(duration))
            continue
        n_segments_used += 1
        starts.extend(int(onset_sample) + window_samples * k for k in range(n_windows))

    if segments_too_short:
        logger.warning(
            "Skipped %d '%s' segment(s) shorter than one epoch (%.3f s): "
            "longest skipped = %.3f s.",
            len(segments_too_short),
            params.condition,
            window_seconds,
            max(segments_too_short),
        )

    if not starts:
        raise DataPrepValidationError(
            "INSUFFICIENT_EPOCH_LENGTH",
            f"No '{params.condition}' segment is long enough for one epoch "
            f"of {window_seconds:.3f} s ({params.min_cycles:g} cycles of "
            f"{params.fmin:g} Hz). Longest segment: "
            f"{max(segments_too_short):.3f} s.",
        )

    epoch_events = np.column_stack(
        [
            np.asarray(starts, dtype=int),
            np.zeros(len(starts), dtype=int),
            np.full(len(starts), event_id[params.condition], dtype=int),
        ]
    )
    epoch_events = epoch_events[np.argsort(epoch_events[:, 0], kind="stable")]
    target_event_id = {params.condition: event_id[params.condition]}

    # 5. Create epochs
    #    tmin=0; tmax is inclusive in MNE, so (window_samples - 1) / sfreq
    #    gives exactly ``window_samples`` samples per epoch and keeps tiled
    #    epochs contiguous without a one-sample overlap.
    #    baseline=None: no baseline correction (band-pass filter already applied)
    try:
        epochs = mne.Epochs(
            raw,
            events=epoch_events,
            event_id=target_event_id,
            tmin=0.0,
            tmax=(window_samples - 1) / sfreq,
            baseline=None,
            preload=True,
            verbose=False,
        )
    except Exception as exc:
        raise DataPrepValidationError(
            "EPOCHING_FAILED",
            f"MNE Epochs creation failed: {type(exc).__name__}: {exc}",
        )

    # Drop bad epochs if any were auto-rejected (e.g. overlapping BAD annotations)
    epochs.drop_bad(verbose=False)

    # 6. Validate epoch count
    n_epochs = len(epochs)
    if n_epochs == 0:
        raise DataPrepValidationError(
            "NO_VALID_EPOCHS",
            "No valid epochs remain after segmentation. The epoch duration "
            f"({window_seconds:.3f} s) may exceed the available annotation "
            f"segments for condition '{params.condition}'.",
        )

    if n_epochs < 2:
        logger.warning(
            "Only %d epoch(s) for condition '%s'. Phase- and coherence-based "
            "connectivity is estimated across epochs and is not reliable with "
            "so few; downstream connectivity checks should treat this as "
            "insufficient data.",
            n_epochs,
            params.condition,
        )

    logger.info(
        "Created %d epoch(s) of %.3f s for condition '%s' "
        "(%d tiled segment(s), %d marker(s), %d segment(s) too short).",
        n_epochs,
        window_seconds,
        params.condition,
        n_segments_used,
        n_markers,
        len(segments_too_short),
    )

    return epochs
