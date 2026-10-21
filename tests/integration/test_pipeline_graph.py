"""End-to-end LangGraph DAG execution test with deterministic nodes."""

from pathlib import Path
import mne
import numpy as np
import pytest
from langgraph.types import Command

from fc_pipeline.pipeline.graph import compile_pipeline_app
from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand


@pytest.fixture
def synthetic_raw_dataset(tmp_path: Path):
    """Creates a synthetic raw EEG file with events suitable for data prep and connectivity."""
    sfreq = 250.0
    n_channels = 4
    ch_names = ["F3", "F4", "C3", "C4"]
    duration_s = 20.0
    n_samples = int(sfreq * duration_s)

    np.random.seed(42)
    # 10 Hz sine wave + noise to ensure distinct alpha connectivity
    t = np.arange(n_samples) / sfreq
    alpha_signal = 1e-5 * np.sin(2 * np.pi * 10 * t)
    data = np.random.randn(n_channels, n_samples) * 5e-6
    data[0] += alpha_signal
    data[1] += 0.8 * alpha_signal  # correlated with F3

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    # Add annotations for condition 'eyes_closed'
    # 5 epochs of 3 seconds each
    onsets = [1.0, 5.0, 9.0, 13.0, 17.0]
    durations = [3.0] * len(onsets)
    descriptions = ["eyes_closed"] * len(onsets)
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(annotations)

    raw_path = tmp_path / "synthetic_raw.fif"
    raw.save(raw_path, overwrite=True, verbose=False)
    return str(raw_path.resolve())


def test_data_prep_to_connectivity_dag(synthetic_raw_dataset, tmp_path: Path):
    """Tests DAG execution from Gate 1 approval through Data Preparation to Connectivity Analysis."""
    plan = AnalysisPlan(
        freq_band=FrequencyBand(name="alpha", fmin=8.0, fmax=12.0),
        channels=["F3", "F4", "C3", "C4"],
        condition="eyes_closed",
        metrics=[
            MetricEnum.PLI,
            MetricEnum.WPLI,
            MetricEnum.IMAGINARY_COHERENCE,
            MetricEnum.PLV,
            MetricEnum.COHERENCE,
        ],
    )

    manifest = [
        ParameterManifestEntry(
            name="freq_band",
            category="scientific_axis",
            proposed_value="alpha (8.0 - 12.0 Hz)",
            human_approved_value="alpha (8.0 - 12.0 Hz)",
            risk_tier="low",
        ),
        ParameterManifestEntry(
            name="channels",
            category="scientific_axis",
            proposed_value="F3, F4, C3, C4",
            human_approved_value="F3, F4, C3, C4",
            risk_tier="low",
        ),
        ParameterManifestEntry(
            name="condition",
            category="scientific_axis",
            proposed_value="eyes_closed",
            human_approved_value="eyes_closed",
            risk_tier="low",
        ),
        ParameterManifestEntry(
            name="reference",
            category="scientific_axis",
            proposed_value="average",
            human_approved_value="average",
            risk_tier="elevated",
        ),
        ParameterManifestEntry(
            name="bad_channel_variance_threshold",
            category="engineering_threshold",
            proposed_value="1e-15",
            human_approved_value="1e-15",
            risk_tier="elevated",
        ),
        ParameterManifestEntry(
            name="min_cycles",
            category="engineering_threshold",
            proposed_value="3",
            human_approved_value="3",
            risk_tier="elevated",
        ),
        ParameterManifestEntry(
            name="tau_phase",
            category="scientific_axis",
            proposed_value="0.20",
            human_approved_value="0.20",
            risk_tier="elevated",
        ),
        ParameterManifestEntry(
            name="tau_zerolag",
            category="scientific_axis",
            proposed_value="0.35",
            human_approved_value="0.35",
            risk_tier="elevated",
        ),
    ]

    thread_id = "test_thread_node3"
    config = {"configurable": {"thread_id": thread_id}}
    app = compile_pipeline_app()

    # Seed state as if Supervisor just produced a valid AnalysisPlan
    app.update_state(
        config,
        {
            "pipeline_error": None,
            "run_id": thread_id,
            "raw_data_path": synthetic_raw_dataset,
            "user_request": "Run alpha connectivity analysis on eyes_closed condition.",
            "latest_user_message": "Run alpha connectivity analysis on eyes_closed condition.",
            "plan": plan,
            "parameter_manifest": manifest,
            "preflight_confirmed": False,
            "gate_1_approved": False,
            "input_rail_cleared": True,
        },
        as_node="supervisor",
    )

    # 1. Stream into gate_1_review to trigger human checkpoint interrupt
    events = list(app.stream(None, config=config, stream_mode=["updates"]))
    assert any(
        isinstance(e, tuple) and len(e) > 1 and "__interrupt__" in e[1]
        for e in events
    )

    # Check paused state snapshot
    before_state = app.get_state(config)
    assert before_state.next == ("gate_1_review",)
    assert before_state.values["preflight_confirmed"] is False
    assert before_state.values["gate_1_approved"] is False

    # 2. Human clicks Approve -> resume into Data Prep & Connectivity Analysis
    resume_events = list(
        app.stream(
            Command(resume={"action": "approve"}),
            config=config,
            stream_mode=["updates"],
        )
    )

    after_state = app.get_state(config)
    assert after_state.next == ()  # Full DAG execution completed

    # 3. Verify Data Prep results
    assert after_state.values.get("data_prep_error") is None
    preprocessed_path = after_state.values.get("preprocessed_data_path")
    assert preprocessed_path is not None
    assert Path(preprocessed_path).exists()

    # 4. Verify Connectivity outputs (Node 3)
    assert after_state.values.get("pipeline_error") is None

    metric_csvs = after_state.values.get("metric_csv_paths")
    assert metric_csvs is not None
    assert "pli" in metric_csvs
    assert "wpli" in metric_csvs
    assert "imcoh" in metric_csvs
    assert "plv" in metric_csvs
    assert "coh" in metric_csvs
    assert "evidence_summary" in metric_csvs

    for m, p in metric_csvs.items():
        assert Path(p).exists(), f"CSV file for {m} does not exist at {p}"

    heatmaps = after_state.values.get("heatmap_image_paths")
    assert heatmaps is not None
    assert len(heatmaps) == 5
    for m, p in heatmaps.items():
        assert Path(p).exists()

    networks = after_state.values.get("network_image_paths")
    assert networks is not None
    assert len(networks) == 5
    for m, p in networks.items():
        assert Path(p).exists()

    summaries = after_state.values.get("numerical_summaries")
    assert summaries is not None
    assert len(summaries) == 5
    assert "mean" in summaries["pli"]
    assert "top_5_pairs_mean" in summaries["pli"]

    evidence = after_state.values.get("evidence_summary")
    assert evidence is not None
    assert len(evidence) == 6  # 4 channels -> 6 pairs
    for item in evidence:
        assert item.classification is not None
        assert 0.0 <= item.phase_robust_mean <= 1.0
        assert 0.0 <= item.zero_lag_mean <= 1.0
