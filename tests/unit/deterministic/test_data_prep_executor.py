"""Focused tests for the Data Preparation implementation.

Tests cover:
  1. Approved Gate 1 → Data Prep executes
  2. Unapproved Gate 1 → Data Prep does not execute (validation error)
  3. Valid EEG → preprocessing completes and output is created
  4. Flatline channel → channel is detected and recorded
  5. Diagnostics → plot paths are produced
  6. Chainlit state fields → Data Prep result populates existing fields
  7. Graph wiring → Gate 1 approval routes to data_prep, not END
"""

from __future__ import annotations

import numpy as np
import pytest
import mne
from pathlib import Path

from fc_pipeline.deterministic.data_prep.models import (
    DataPrepInput,
    DataPrepResult,
    ValidatedDataPrepParams,
)
from fc_pipeline.deterministic.data_prep.cleaning import clean_bad_channels
from fc_pipeline.deterministic.data_prep.referencing import apply_reference
from fc_pipeline.deterministic.data_prep.epoching import filter_and_epoch
from fc_pipeline.deterministic.data_prep.plotting import generate_diagnostics
from fc_pipeline.deterministic.data_prep.executor import run_data_prep
from fc_pipeline.deterministic.data_prep.validation import DataPrepValidationError
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.enums import MetricEnum


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_channels():
    return ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]


@pytest.fixture
def sample_plan(sample_channels):
    return AnalysisPlan(
        metrics=[MetricEnum.PLI, MetricEnum.COHERENCE],
        freq_band=FrequencyBand(name="alpha", fmin=8.0, fmax=13.0),
        channels=sample_channels,
        condition="rest",
    )


@pytest.fixture
def sample_manifest(sample_channels):
    return [
        ParameterManifestEntry(
            name="freq_band",
            category="scientific_axis",
            proposed_value="alpha (8.0 - 13.0 Hz)",
        ),
        ParameterManifestEntry(
            name="channels",
            category="scientific_axis",
            proposed_value=", ".join(sample_channels),
        ),
        ParameterManifestEntry(
            name="condition",
            category="scientific_axis",
            proposed_value="rest",
        ),
        ParameterManifestEntry(
            name="reference",
            category="engineering_threshold",
            proposed_value="average",
        ),
        ParameterManifestEntry(
            name="bad_channel_variance_threshold",
            category="engineering_threshold",
            proposed_value="1e-15",
        ),
        ParameterManifestEntry(
            name="min_cycles",
            category="engineering_threshold",
            proposed_value="3",
        ),
    ]


@pytest.fixture
def valid_params(synthetic_eeg_path, sample_channels):
    return ValidatedDataPrepParams(
        raw_data_path=synthetic_eeg_path,
        run_id="test001",
        channels=tuple(sample_channels),
        condition="rest",
        fmin=8.0,
        fmax=13.0,
        reference_method="average",
        bad_channel_variance_threshold=1e-15,
        min_cycles=3.0,
    )


@pytest.fixture
def raw_eeg(synthetic_eeg_path):
    return mne.io.read_raw_fif(str(synthetic_eeg_path), preload=True, verbose=False)


@pytest.fixture
def data_prep_input_approved(synthetic_eeg_path, sample_plan, sample_manifest):
    return DataPrepInput(
        raw_data_path=str(synthetic_eeg_path),
        plan=sample_plan,
        parameter_manifest=sample_manifest,
        run_id="test001",
        gate_1_approved=True,
        preflight_confirmed=True,
    )


@pytest.fixture
def data_prep_input_unapproved(synthetic_eeg_path, sample_plan, sample_manifest):
    return DataPrepInput(
        raw_data_path=str(synthetic_eeg_path),
        plan=sample_plan,
        parameter_manifest=sample_manifest,
        run_id="test001",
        gate_1_approved=False,
        preflight_confirmed=False,
    )


# ─── Test 1: Approved Gate 1 → Data Prep executes ─────────────────────────

def test_approved_gate1_data_prep_executes(data_prep_input_approved, tmp_path):
    """Approved Gate 1 input → run_data_prep returns success."""
    result = run_data_prep(
        data_prep_input_approved,
        output_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
    )
    assert result.success is True, f"Expected success, got error: {result.error}"
    assert result.preprocessed_data_path is not None
    assert Path(result.preprocessed_data_path).exists()


# ─── Test 2: Unapproved Gate 1 → Data Prep does not execute ───────────────

def test_unapproved_gate1_data_prep_blocked(data_prep_input_unapproved, tmp_path):
    """Unapproved Gate 1 → run_data_prep returns failure with GATE_1_NOT_APPROVED."""
    result = run_data_prep(
        data_prep_input_unapproved,
        output_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
    )
    assert result.success is False
    assert "GATE_1_NOT_APPROVED" in (result.error or "")


# ─── Test 3: Valid EEG → preprocessing completes ──────────────────────────

def test_valid_eeg_preprocessing_completes(data_prep_input_approved, tmp_path):
    """Full pipeline produces a .fif output and a summary."""
    result = run_data_prep(
        data_prep_input_approved,
        output_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
    )
    assert result.success is True
    assert result.summary is not None
    assert result.summary.epoch_count > 0
    assert result.summary.sampling_frequency == 250.0
    assert result.summary.condition == "rest"
    assert result.summary.filter_l_freq == 8.0
    assert result.summary.filter_h_freq == 13.0
    assert result.summary.reference_applied == "average"
    assert result.summary.ica_applied is False


# ─── Test 4: Flatline channel → detected and recorded ─────────────────────

def test_flatline_channel_detected(valid_params, tmp_path):
    """A channel with zero data (flatline) is detected and dropped."""
    # Create a raw with one flatline channel
    ch_names = list(valid_params.channels)
    sfreq = 250.0
    n_samples = int(10.0 * sfreq)
    np.random.seed(42)
    data = np.random.randn(len(ch_names), n_samples) * 1e-6
    # Make O2 (index 7) a flatline
    data[7, :] = 0.0

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    raw, dropped = clean_bad_channels(raw, valid_params)
    assert "O2" in dropped
    assert "O2" not in raw.ch_names
    assert len(raw.ch_names) >= 2  # enough channels remain


# ─── Test 5: Diagnostics → plot paths produced ────────────────────────────

def test_diagnostic_plots_produced(data_prep_input_approved, tmp_path):
    """run_data_prep produces diagnostic plot files."""
    result = run_data_prep(
        data_prep_input_approved,
        output_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
    )
    assert result.success is True
    assert len(result.channel_plot_paths) > 0
    for plot_name, plot_path in result.channel_plot_paths.items():
        assert Path(plot_path).exists(), f"Plot '{plot_name}' not found at {plot_path}"


# ─── Test 6: Data Prep result populates existing state fields ──────────────

def test_data_prep_result_maps_to_state_fields(data_prep_input_approved, tmp_path):
    """DataPrepResult fields match the existing GraphState field names."""
    result = run_data_prep(
        data_prep_input_approved,
        output_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
    )
    # These must match existing GraphState field names
    assert hasattr(result, "bad_channels_dropped")
    assert hasattr(result, "channel_plot_paths")
    assert hasattr(result, "preprocessed_data_path")
    assert hasattr(result, "error")

    # Simulate what data_prep_node_adapter does
    update = {
        "bad_channels_dropped": result.bad_channels_dropped or [],
        "channel_plot_paths": result.channel_plot_paths or {},
        "preprocessed_data_path": result.preprocessed_data_path,
        "data_prep_error": result.error,
    }
    assert isinstance(update["bad_channels_dropped"], list)
    assert isinstance(update["channel_plot_paths"], dict)
    assert update["data_prep_error"] is None  # success case


# ─── Test 7: Graph wiring → approval routes to data_prep ──────────────────

def test_graph_routes_approval_to_data_prep():
    """route_gate_1_output returns 'data_prep' when approved (not END)."""
    from fc_pipeline.pipeline.graph import route_gate_1_output

    state = {
        "preflight_confirmed": True,
        "gate_1_approved": True,
        "decision_context": None,
    }
    result = route_gate_1_output(state)
    assert result == "data_prep", f"Expected 'data_prep', got '{result}'"


def test_graph_routes_rejection_to_end():
    """route_gate_1_output returns END when not approved."""
    from fc_pipeline.pipeline.graph import route_gate_1_output
    from langgraph.graph import END

    state = {
        "preflight_confirmed": False,
        "gate_1_approved": False,
        "decision_context": None,
    }
    result = route_gate_1_output(state)
    assert result == END


def test_graph_has_data_prep_node():
    """The compiled graph contains a 'data_prep' node."""
    from fc_pipeline.pipeline.graph import build_pipeline_graph

    graph = build_pipeline_graph()
    assert "data_prep" in graph.nodes, (
        f"'data_prep' not found in graph nodes: {list(graph.nodes.keys())}"
    )


# ─── Cleaning module unit tests ───────────────────────────────────────────

def test_clean_no_bad_channels(valid_params, raw_eeg):
    """Normal data (random noise) → no channels dropped."""
    raw_eeg, dropped = clean_bad_channels(raw_eeg, valid_params)
    assert dropped == []
    assert len(raw_eeg.ch_names) == len(valid_params.channels)


# ─── Referencing module unit tests ────────────────────────────────────────

def test_apply_average_reference(valid_params, raw_eeg):
    """Average reference applies without error."""
    raw_eeg.pick_channels(list(valid_params.channels), ordered=True)
    raw_eeg, method = apply_reference(raw_eeg, valid_params)
    assert method == "average"


# ─── Epoching module unit tests ───────────────────────────────────────────

def test_filter_and_epoch(valid_params, raw_eeg):
    """Bandpass filter + epoching produces valid Epochs."""
    raw_eeg.pick_channels(list(valid_params.channels), ordered=True)
    epochs = filter_and_epoch(raw_eeg, valid_params)
    assert len(epochs) > 0
    assert epochs.info["sfreq"] == 250.0
