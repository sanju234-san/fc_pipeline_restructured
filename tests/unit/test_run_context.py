"""Run context: build / persist / restore, and deterministic answers about a completed run."""

from __future__ import annotations

import json

import pytest

from fc_pipeline.deterministic.data_prep.models import DataPrepSummary
from fc_pipeline.pipeline.run_context import (
    answer_run_question,
    build_run_context,
    build_run_context_from_state,
    load_run_context,
    looks_like_new_analysis,
    metric_label,
    restore_state,
    run_context_path,
    save_run_context,
)
from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand


@pytest.fixture()
def state():
    plan = AnalysisPlan(
        freq_band=FrequencyBand(name="theta", fmin=4.0, fmax=8.0),
        channels=["C3..", "C4..", "Cz.."],
        condition="T0",
    )
    summary = DataPrepSummary(
        sampling_frequency=160.0,
        original_channel_count=64,
        selected_channel_count=3,
        retained_channel_count=3,
        dropped_channels=[],
        reference_applied="average",
        condition="T0",
        epoch_count=80,
        epoch_duration_seconds=0.75,
        filter_l_freq=4.0,
        filter_h_freq=8.0,
    )
    manifest = [
        ParameterManifestEntry(
            name="channels", category="scientific_axis", proposed_value="C3.., C4.., Cz.."
        )
    ]
    return {
        "run_id": "chainlit_20261008_130000_001",
        "plan": plan,
        "data_prep_summary": summary,
        "parameter_manifest": manifest,
        "preprocessed_data_path": "/tmp/out/preprocessed_epo_x.fif",
        "raw_data_path": "/tmp/raw.edf",
        "channel_plot_paths": {"channels_before": "/tmp/p/b.png", "psd_overview": "/tmp/p/psd.png"},
        "bad_channels_dropped": [],
    }


@pytest.fixture()
def ctx(state):
    return build_run_context_from_state(state)


# --- build / persist / restore ---------------------------------------------


def test_context_is_plain_json(ctx):
    json.dumps(ctx)  # must not raise
    assert ctx["plan"]["freq_band"]["name"] == "theta"
    assert ctx["plan"]["metrics"] == ["pli", "wpli", "imaginary_coherence", "plv", "coherence"] or len(
        ctx["plan"]["metrics"]
    ) == 5
    assert ctx["connectivity"] is None


def test_save_load_roundtrip_and_restore_types(ctx, tmp_path):
    path = save_run_context(ctx, run_context_path(str(tmp_path / "e_epo.fif"), ctx["run_id"]))
    assert path.exists() and not list(tmp_path.glob(".run_context_*.tmp"))
    loaded = load_run_context(path)
    assert loaded == json.loads(json.dumps(ctx))
    restored = restore_state(loaded)
    assert isinstance(restored["plan"], AnalysisPlan)
    assert isinstance(restored["data_prep_summary"], DataPrepSummary)
    assert restored["plan"].channels == ["C3..", "C4..", "Cz.."]


def test_load_rejects_missing_corrupt_or_wrong_version(tmp_path, ctx):
    assert load_run_context(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_run_context(bad) is None
    old = tmp_path / "old.json"
    old.write_text(json.dumps({**ctx, "schema_version": 999}), encoding="utf-8")
    assert load_run_context(old) is None


# --- answers ----------------------------------------------------------------


def test_the_reported_question_is_answered_from_the_run(ctx):
    answer = answer_run_question("what was the channel selected in this process", ctx)
    assert "C3, C4, Cz" in answer
    assert "64-channel" in answer
    assert "None were dropped" in answer


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Which frequency band was used?", "theta (4-8 Hz)"),
        ("what condition did we analyse", "T0"),
        ("how many epochs are there", "80 of 0.750 s"),
        ("what was the sampling rate", "160 Hz"),
        ("which reference was applied", "average"),
        ("which metrics were requested", "PLI, wPLI, |ImCoh|, PLV, Coherence"),
        ("were any channels dropped", "none"),
        ("what filter was applied", "4-8 Hz"),
        ("what is the run id", "chainlit_20261008_130000_001"),
    ],
)
def test_topic_answers(ctx, question, expected):
    assert expected.lower() in answer_run_question(question, ctx).lower()


def test_sampling_rate_question_does_not_return_the_band(ctx):
    answer = answer_run_question("what is the sampling rate in Hz", ctx)
    assert "160 Hz" in answer and "theta" not in answer


def test_metrics_answer_says_connectivity_is_not_computed_yet(ctx):
    assert "not been computed" in answer_run_question("which metrics were requested", ctx)


def test_summary_covers_the_run(ctx):
    answer = answer_run_question("give me a summary of the run settings", ctx)
    for fragment in ("theta", "C3", "T0", "80", "160 Hz", "average"):
        assert fragment in answer


def test_few_epochs_are_flagged_as_unreliable(state):
    state["data_prep_summary"] = state["data_prep_summary"].model_copy(update={"epoch_count": 1})
    answer = answer_run_question("how many epochs", build_run_context_from_state(state))
    assert "too few" in answer


@pytest.mark.parametrize(
    "request_text",
    [
        "Plot the PSD for C4",
        "Show me the EEG signal for C3",
        "Compare C3 and C4",
        "show the dataset before and after Data Prep",
        "explain the plots",
        "explain the dataplots",
    ],
)
def test_artifact_requests_are_left_to_the_plot_handler(ctx, request_text):
    assert answer_run_question(request_text, ctx) is None


def test_supervisor_and_connectivity_context_roundtrip_and_grounding(state):
    state["user_request"] = "Analyze alpha connectivity in motor channels"
    state["resolved_frequency_band_info"] = {
        "name": "alpha",
        "fmin": 8.0,
        "fmax": 13.0,
        "source": "user_explicit",
        "confidence": 0.95,
        "nyquist_ok": True,
    }
    state["resolved_channel_info"] = {
        "resolved_channels": ["C3..", "C4..", "Cz.."],
        "source": "region_keyword",
        "confidence": 0.90,
        "matched_region": "central / motor",
    }
    state["dataset_sfreq"] = 160.0
    state["heatmap_image_paths"] = {"pli": "outputs/plots/heatmap_pli.png"}
    state["network_image_paths"] = {"pli": "outputs/plots/network_pli.png"}
    state["numerical_summaries"] = {"pli": {"mean": 0.24, "max": 0.41}}

    rich_ctx = build_run_context_from_state(state)
    assert rich_ctx["supervisor"]["dataset_sfreq"] == 160.0
    assert rich_ctx["connectivity"]["heatmap_image_paths"] == {"pli": "outputs/plots/heatmap_pli.png"}

    band_ans = answer_run_question("why this band?", rich_ctx)
    assert "4-8 Hz" in band_ans
    assert "user_explicit" in band_ans
    assert "Nyquist limit (80 Hz at 160 Hz sampling rate)" in band_ans

    ch_ans = answer_run_question("which channels were selected?", rich_ctx)
    assert "central / motor" in ch_ans
    assert "confidence 0.90" in ch_ans

    metric_ans = answer_run_question("what were the connectivity metrics?", rich_ctx)
    assert "mean=0.240" in metric_ans

    restored = restore_state(rich_ctx)
    assert restored["resolved_frequency_band_info"]["name"] == "alpha"
    assert restored["heatmap_image_paths"] == {"pli": "outputs/plots/heatmap_pli.png"}
    assert restored["numerical_summaries"] == {"pli": {"mean": 0.24, "max": 0.41}}


def test_unrelated_text_and_missing_data_return_none(ctx):
    assert answer_run_question("hello there", ctx) is None
    assert answer_run_question("which channels", {}) is None
    empty = build_run_context(run_id="r", plan=None, data_prep_summary=None)
    assert answer_run_question("which channels were selected", empty) is None
    assert answer_run_question("how many epochs", empty) is None


def test_dropped_channels_are_reported_and_excluded_from_retained(state):
    state["bad_channels_dropped"] = ["Cz.."]
    answer = answer_run_question("which channels were selected", build_run_context_from_state(state))
    assert "Dropped as bad:** Cz." in answer and "Retained:** C3, C4." in answer


# --- new analysis vs question ------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "analyze alpha band on frontal channels",
        "Now compute wPLI for beta",
        "please run it again with a different band",
        "try the occipital region",
    ],
)
def test_new_analysis_requests_are_detected(text):
    assert looks_like_new_analysis(text)


@pytest.mark.parametrize(
    "text",
    [
        "what was the channel selected in this process",
        "which band was used?",
        "how many epochs are there",
        "show me the PSD",
        "analyze this",  # names nothing to analyse: not a concrete new request
    ],
)
def test_questions_are_not_new_analyses(text):
    assert not looks_like_new_analysis(text)


def test_metric_labels_accept_enum_value_and_repr():
    assert metric_label(MetricEnum.PLI) == "PLI"
    assert metric_label("MetricEnum.WPLI") == "wPLI"
    assert metric_label("imaginary_coherence") == "|ImCoh|"
