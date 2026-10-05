"""Data Preparation executor — single orchestrator for the deterministic pipeline.

Chains: validate → load → clean → reference → filter/epoch → plot → save.
All errors are captured into ``DataPrepResult(success=False, error=...)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import mne

from fc_pipeline.deterministic.data_prep.cleaning import clean_bad_channels
from fc_pipeline.deterministic.data_prep.epoching import filter_and_epoch
from fc_pipeline.deterministic.data_prep.models import (
    DataPrepInput,
    DataPrepResult,
    DataPrepSummary,
)
from fc_pipeline.deterministic.data_prep.plotting import generate_diagnostics
from fc_pipeline.deterministic.data_prep.referencing import apply_reference
from fc_pipeline.agentic.supervisor.tracer import SupervisorTracer
from fc_pipeline.deterministic.data_prep.validation import (
    DataPrepValidationError,
    build_safe_output_path,
    validate_data_prep_input,
    validate_plan_against_recording,
)

logger = logging.getLogger(__name__)

# Default output directory (relative to project root). Created on demand.
_DEFAULT_OUTPUT_DIR = Path("outputs")
_DEFAULT_PLOT_DIR = _DEFAULT_OUTPUT_DIR / "plots"


def _trace(run_id: str, event_type: str, payload: dict) -> None:
    """Emit deterministic Data Prep events through the existing UI/audit tracer."""
    try:
        SupervisorTracer(run_id=run_id).log_event(event_type, payload)
    except Exception:
        # Explainability must never change scientific execution.
        pass


def run_data_prep(
    data_prep_input: DataPrepInput,
    *,
    output_dir: Optional[Path] = None,
    plot_dir: Optional[Path] = None,
    allowed_data_roots: Optional[Sequence[str]] = None,
) -> DataPrepResult:
    """Execute the full deterministic Data Preparation pipeline.

    Parameters
    ----------
    data_prep_input : DataPrepInput
        Trusted input assembled from graph state (includes Gate 1 flags,
        plan, manifest, raw_data_path, run_id).
    output_dir : Path, optional
        Directory for the processed ``.fif`` file.  Defaults to ``outputs/``.
    plot_dir : Path, optional
        Directory for diagnostic plots.  Defaults to ``outputs/plots/``.
    allowed_data_roots : sequence of str, optional
        Passed through to ``validate_data_prep_input`` for path containment.

    Returns
    -------
    DataPrepResult
        Always returned (never raises).  On failure, ``success=False`` with
        a machine-readable ``error`` string.
    """
    out = output_dir or _DEFAULT_OUTPUT_DIR
    plots = plot_dir or _DEFAULT_PLOT_DIR

    try:
        return _run(data_prep_input, out, plots, allowed_data_roots)
    except DataPrepValidationError as exc:
        logger.error("Data Prep validation failed: %s", exc)
        _trace(data_prep_input.run_id, "data_prep_failed", {"stage": "validation", "error": str(exc)})
        return DataPrepResult(success=False, error=str(exc))
    except Exception as exc:
        logger.exception("Data Prep unexpected error: %s", exc)
        _trace(data_prep_input.run_id, "data_prep_failed", {"stage": "unexpected", "error": f"{type(exc).__name__}: {exc}"})
        return DataPrepResult(
            success=False,
            error=f"UNEXPECTED_ERROR: {type(exc).__name__}: {exc}",
        )


def _run(
    data_prep_input: DataPrepInput,
    output_dir: Path,
    plot_dir: Path,
    allowed_data_roots: Optional[Sequence[str]],
) -> DataPrepResult:
    """Inner implementation — raises on error (caught by ``run_data_prep``)."""

    # ── 1. Validate (calls existing validation.py) ─────────────────────
    _trace(data_prep_input.run_id, "data_prep_stage", {"stage": "validate", "status": "started"})
    params = validate_data_prep_input(
        data_prep_input, allowed_data_roots=allowed_data_roots
    )
    _trace(data_prep_input.run_id, "data_prep_stage", {
        "stage": "validate", "status": "completed",
        "channels": len(params.channels), "condition": params.condition,
        "fmin": params.fmin, "fmax": params.fmax, "reference": params.reference_method,
    })
    logger.info(
        "Validation passed. run_id=%s, channels=%d, condition=%s, "
        "band=%.1f–%.1f Hz, ref=%s",
        params.run_id,
        len(params.channels),
        params.condition,
        params.fmin,
        params.fmax,
        params.reference_method,
    )

    # ── 2. Load EEG ────────────────────────────────────────────────────
    _trace(params.run_id, "data_prep_stage", {"stage": "load", "status": "started", "file": params.raw_data_path.name})
    logger.info("Loading EEG data from: %s", params.raw_data_path.name)
    try:
        raw = mne.io.read_raw(str(params.raw_data_path), preload=True, verbose=False)
    except Exception as exc:
        raise DataPrepValidationError(
            "EEG_LOAD_FAILED",
            f"Could not load EEG file: {type(exc).__name__}: {exc}",
        )

    original_channel_count = len(raw.ch_names)
    sampling_frequency = raw.info["sfreq"]
    _trace(params.run_id, "data_prep_stage", {
        "stage": "load", "status": "completed",
        "channel_count": original_channel_count, "sampling_frequency": sampling_frequency,
        "duration_seconds": raw.times[-1] if len(raw.times) else 0.0,
    })

    # ── 3. Post-load validation ────────────────────────────────────────
    #    Extract condition labels from annotations for validation.
    condition_labels: set[str] = set()
    if raw.annotations is not None and len(raw.annotations) > 0:
        condition_labels = set(raw.annotations.description)

    validate_plan_against_recording(
        params,
        sfreq=sampling_frequency,
        channel_names=raw.ch_names,
        condition_labels=condition_labels,
    )

    # ── 4. Clean bad channels ──────────────────────────────────────────
    selected_channel_count = len(params.channels)
    _trace(params.run_id, "data_prep_stage", {"stage": "bad_channel_screening", "status": "started"})
    raw, dropped_channels = clean_bad_channels(raw, params)
    retained_channel_count = len(raw.ch_names)
    _trace(params.run_id, "data_prep_stage", {
        "stage": "bad_channel_screening", "status": "completed",
        "dropped_channels": dropped_channels, "retained_channel_count": retained_channel_count,
    })

    # ── 5. Apply reference ─────────────────────────────────────────────
    _trace(params.run_id, "data_prep_stage", {"stage": "reference", "status": "started", "requested_reference": params.reference_method})
    raw, ref_applied = apply_reference(raw, params)
    _trace(params.run_id, "data_prep_stage", {"stage": "reference", "status": "completed", "reference_applied": ref_applied})

    # ── 6. Filter + epoch ──────────────────────────────────────────────
    _trace(params.run_id, "data_prep_stage", {"stage": "filter_and_epoch", "status": "started", "fmin": params.fmin, "fmax": params.fmax, "condition": params.condition})
    epochs = filter_and_epoch(raw, params)
    epoch_duration = epochs.tmax - epochs.tmin
    _trace(params.run_id, "data_prep_stage", {"stage": "filter_and_epoch", "status": "completed", "epoch_count": len(epochs), "epoch_duration_seconds": epoch_duration})

    # ── 7. Diagnostic plots ────────────────────────────────────────────
    _trace(params.run_id, "data_prep_stage", {"stage": "diagnostics", "status": "started"})
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = generate_diagnostics(epochs, params, plot_dir)
    _trace(params.run_id, "data_prep_stage", {"stage": "diagnostics", "status": "completed", "plot_count": len(plot_paths), "plot_names": list(plot_paths.keys())})

    # ── 8. Save processed epochs ───────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_safe_output_path(
        output_dir, params.run_id, "preprocessed_epo", ".fif"
    )
    epochs.save(str(output_path), overwrite=True, verbose=False)
    _trace(params.run_id, "data_prep_stage", {"stage": "save", "status": "completed", "output_file": output_path.name})
    logger.info("Preprocessed epochs saved: %s", output_path.name)

    # ── 9. Build result ────────────────────────────────────────────────
    summary = DataPrepSummary(
        sampling_frequency=sampling_frequency,
        original_channel_count=original_channel_count,
        selected_channel_count=selected_channel_count,
        retained_channel_count=retained_channel_count,
        dropped_channels=dropped_channels,
        reference_detected=None,
        reference_applied=ref_applied,
        condition=params.condition,
        epoch_count=len(epochs),
        epoch_duration_seconds=round(epoch_duration, 4),
        filter_l_freq=params.fmin,
        filter_h_freq=params.fmax,
        ica_applied=False,
        interpolation_applied=False,
    )

    return DataPrepResult(
        success=True,
        preprocessed_data_path=str(output_path),
        bad_channels_dropped=dropped_channels,
        channel_plot_paths=plot_paths,
        summary=summary,
        error=None,
    )
