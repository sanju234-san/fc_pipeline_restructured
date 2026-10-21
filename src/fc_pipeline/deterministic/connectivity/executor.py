"""Deterministic connectivity execution engine orchestrating Node 3 computation.

Follows Section 6.3 and Appendix B of PIPELINE_PROTOTYPE_DESIGN.md.
Pure signal processing and math: zero LLM / VLM / MLflow dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mne

from fc_pipeline.agentic.supervisor.tracer import SupervisorTracer
from fc_pipeline.config.constants import OUTPUT_DIR
from fc_pipeline.config.thresholds import TAU_PHASE, TAU_ZEROLAG
from fc_pipeline.deterministic.connectivity.classification import (
    classify_channel_pairs,
    export_connectivity_matrices_to_csv,
    export_evidence_summary_to_csv,
    export_consolidated_connectivity_report_csv,
)
from fc_pipeline.deterministic.connectivity.metrics import (
    ConnectivityComputationError,
    compute_metric_numerical_summaries,
    compute_spectral_connectivity,
)
from fc_pipeline.deterministic.connectivity.models import (
    ConnectivityInput,
    ConnectivityResult,
)
from fc_pipeline.deterministic.connectivity.plotting import (
    render_all_connectivity_plots,
)

logger = logging.getLogger(__name__)


def _trace(run_id: str, event_type: str, payload: dict) -> None:
    """Emit deterministic connectivity audit event through SupervisorTracer."""
    try:
        SupervisorTracer(run_id=run_id).log_event(event_type, payload)
    except Exception:
        pass


def _extract_tau_thresholds(connectivity_input: ConnectivityInput) -> tuple[float, float]:
    """Extracts tau_phase and tau_zerolag from Gate 1 parameter manifest if present."""
    tau_phase = TAU_PHASE
    tau_zerolag = TAU_ZEROLAG

    manifest = connectivity_input.parameter_manifest or []
    for entry in manifest:
        val_str = entry.human_approved_value or entry.proposed_value
        if not val_str:
            continue
        try:
            val_float = float(val_str)
            if entry.name == "tau_phase":
                tau_phase = val_float
            elif entry.name == "tau_zerolag":
                tau_zerolag = val_float
        except ValueError:
            pass

    return tau_phase, tau_zerolag


def _extract_subject_id(epochs) -> Optional[str]:
    """Subject identifier recorded in the epochs' MNE metadata, if any (never invented)."""
    try:
        info = epochs.info.get("subject_info") or {}
        for key in ("his_id", "last_name", "first_name"):
            val = info.get(key)
            if val not in (None, ""):
                return str(val)
    except Exception:
        pass
    return None


def run_connectivity_analysis(
    connectivity_input: ConnectivityInput,
    *,
    output_dir: Optional[Path] = None,
) -> ConnectivityResult:
    """Executes the full deterministic Connectivity Analysis stage.

    Parameters
    ----------
    connectivity_input : ConnectivityInput
        Validated input payload assembled from GraphState.
    output_dir : Path, optional
        Target directory for artifacts. Defaults to OUTPUT_DIR ('outputs/').

    Returns
    -------
    ConnectivityResult
        Always returned (never raises unhandled exceptions).
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = connectivity_input.run_id

    _trace(
        run_id,
        "connectivity_started",
        {
            "plan_metrics": [m.value for m in connectivity_input.plan.metrics],
            "freq_band": connectivity_input.plan.freq_band.name,
            "fmin": connectivity_input.plan.freq_band.fmin,
            "fmax": connectivity_input.plan.freq_band.fmax,
        },
    )

    fif_path = Path(connectivity_input.preprocessed_data_path)
    if not fif_path.exists():
        err = f"PREPROCESSED_FILE_NOT_FOUND: Preprocessed FIF file not found at: {fif_path}"
        logger.error(err)
        _trace(run_id, "connectivity_failed", {"error": err})
        return ConnectivityResult(success=False, error=err)

    try:
        # 1. Load preprocessed epoched data
        logger.info("Loading preprocessed epochs from %s", fif_path)
        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)

        # 2. Match requested channels
        plan_channels = connectivity_input.plan.channels
        available_ch = epochs.ch_names
        channels_to_use = [ch for ch in plan_channels if ch in available_ch]

        if len(channels_to_use) < 2:
            err = (
                f"INSUFFICIENT_CHANNELS: Only {len(channels_to_use)} requested channels "
                f"({channels_to_use}) are present in preprocessed epochs (available: {available_ch}). "
                "Connectivity analysis requires >= 2 channels."
            )
            logger.error(err)
            _trace(run_id, "connectivity_failed", {"error": err})
            return ConnectivityResult(success=False, error=err)

        if len(channels_to_use) < len(available_ch):
            logger.info("Subselecting %d channels for analysis: %s", len(channels_to_use), channels_to_use)
            epochs = epochs.pick(channels_to_use)

        # 3. Execute single-pass spectral connectivity
        plan = connectivity_input.plan
        matrices = compute_spectral_connectivity(
            epochs=epochs,
            methods=plan.metrics,
            fmin=plan.freq_band.fmin,
            fmax=plan.freq_band.fmax,
            sfreq=epochs.info["sfreq"],
            mode="multitaper",
        )

        # 4. Compute numerical summaries
        summaries = compute_metric_numerical_summaries(matrices)

        # 5. Extract thresholds and perform complementary evidence synthesis
        tau_phase, tau_zerolag = _extract_tau_thresholds(connectivity_input)
        evidence_list = classify_channel_pairs(
            matrices=matrices,
            channel_names=epochs.ch_names,
            tau_phase=tau_phase,
            tau_zerolag=tau_zerolag,
        )

        # 6. Export CSV artifacts
        metric_csv_paths = export_connectivity_matrices_to_csv(
            matrices=matrices,
            channel_names=epochs.ch_names,
            output_dir=out_dir,
            run_id=run_id,
        )
        evidence_csv_path = export_evidence_summary_to_csv(
            evidence_list=evidence_list,
            matrices=matrices,
            channel_names=epochs.ch_names,
            output_dir=out_dir,
            run_id=run_id,
        )
        metric_csv_paths["evidence_summary"] = evidence_csv_path

        consolidated_csv_path = export_consolidated_connectivity_report_csv(
            evidence_list=evidence_list,
            matrices=matrices,
            channel_names=epochs.ch_names,
            output_dir=out_dir,
            run_id=run_id,
            condition=getattr(plan, "condition", None),
            freq_band=getattr(getattr(plan, "freq_band", None), "name", None),
            subject=_extract_subject_id(epochs),
            fmin=getattr(getattr(plan, "freq_band", None), "fmin", None),
            fmax=getattr(getattr(plan, "freq_band", None), "fmax", None),
        )
        metric_csv_paths["consolidated_report"] = consolidated_csv_path

        # 7. Render Heatmaps & Circular Network Graphs
        heatmaps, networks = render_all_connectivity_plots(
            matrices=matrices,
            channel_names=epochs.ch_names,
            output_dir=out_dir,
            run_id=run_id,
        )

        _trace(
            run_id,
            "connectivity_completed",
            {
                "computed_metrics": list(matrices.keys()),
                "pair_count": len(evidence_list),
                "heatmaps": list(heatmaps.values()),
                "networks": list(networks.values()),
                "consolidated_report": consolidated_csv_path,
            },
        )

        logger.info(
            "Connectivity Analysis completed successfully for run_id=%s (%d metrics, %d pairs)",
            run_id,
            len(matrices),
            len(evidence_list),
        )

        return ConnectivityResult(
            success=True,
            metric_csv_paths=metric_csv_paths,
            heatmap_image_paths=heatmaps,
            network_image_paths=networks,
            numerical_summaries=summaries,
            evidence_summary=evidence_list,
            consolidated_report_path=consolidated_csv_path,
            error=None,
        )

    except ConnectivityComputationError as exc:
        err_msg = f"CONNECTIVITY_COMPUTATION_ERROR: {exc}"
        logger.error(err_msg)
        _trace(run_id, "connectivity_failed", {"error": err_msg})
        return ConnectivityResult(success=False, error=err_msg)
    except Exception as exc:
        err_msg = f"UNEXPECTED_CONNECTIVITY_ERROR: {type(exc).__name__}: {exc}"
        logger.exception(err_msg)
        _trace(run_id, "connectivity_failed", {"error": err_msg})
        return ConnectivityResult(success=False, error=err_msg)
