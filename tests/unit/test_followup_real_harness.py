"""Post-run follow-up: the REAL Deep Agents harness, run offline with a fake model.

The earlier follow-up tests only mocked an import failure, so a broken harness
(`DeclarativeSubagent` does not exist in deepagents; `chat_model=` is not a
`create_deep_agent` argument) went unnoticed and every follow-up silently fell
back to a canned reply. These tests run the real `create_deep_agent`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("deepagents")

import mne  # noqa: E402
import numpy as np  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from fc_pipeline.agentic.followup import agents as ag  # noqa: E402
from fc_pipeline.agentic.followup.agents import (  # noqa: E402
    FollowUpResultKind,
    run_followup_agent,
)
from fc_pipeline.agentic.followup.context_window import PostRunContextWindowManager  # noqa: E402


class FakeChat(GenericFakeChatModel):
    """Scripted chat model that tolerates ``bind_tools`` and records its inputs."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self

    def _generate(self, messages, *a, **k):  # noqa: ANN001
        type(self).seen = list(messages)
        return super()._generate(messages, *a, **k)


class ExplodingChat:
    def invoke(self, *a, **k):  # noqa: ANN002
        raise AssertionError("the language model must not be used for this question")

    def bind_tools(self, *a, **k):  # noqa: ANN002
        raise AssertionError("the language model must not be used for this question")


def _ctx(tmp_path, *, raw=None, plots=None):
    return {
        "schema_version": 1,
        "run_id": "run_x",
        "raw_data_path": str(raw) if raw else None,
        "preprocessed_data_path": None,
        "plan": {
            "freq_band": {"name": "theta", "fmin": 4.0, "fmax": 8.0},
            "condition": "T0",
            "channels": ["C3", "C4", "Cz"],
            "metrics": ["pli", "wpli", "imcoh", "plv", "coh"],
        },
        "data_prep_summary": {
            "original_channel_count": 3, "retained_channel_count": 3,
            "dropped_channels": [], "epoch_count": 80, "epoch_duration_seconds": 0.75,
            "sampling_frequency": 160.0, "filter_l_freq": 4.0, "filter_h_freq": 8.0,
            "reference_applied": "average", "condition": "T0",
        },
        "bad_channels_dropped": [],
        "channel_plot_paths": plots or {},
        "parameter_manifest": [],
        "connectivity": None,
    }


def _mgr(ctx):
    return PostRunContextWindowManager(ctx, token_window=4000)


def _fif(tmp_path):
    raw = mne.io.RawArray(
        np.random.RandomState(0).randn(3, 160 * 10) * 1e-5,
        mne.create_info(["C3", "C4", "Cz"], 160.0, "eeg"),
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([0.0], [10.0], ["T0"]))
    path = tmp_path / "rec_raw.fif"
    raw.save(str(path), overwrite=True, verbose=False)
    return path


def test_real_harness_runs_and_returns_the_model_answer(tmp_path):
    model = FakeChat(messages=iter([AIMessage(content="The setup follows the approved plan.")]))
    res = run_followup_agent("why was the analysis set up this way", _mgr(_ctx(tmp_path)), chat_model=model)
    assert res.error is None, res.error
    assert res.kind == FollowUpResultKind.ANSWERED
    assert res.assistant_text == "The setup follows the approved plan."
    assert res.delegated_to == "Coordinator"


def test_real_harness_delegates_to_the_run_context_subagent(tmp_path):
    call = AIMessage(
        content="",
        tool_calls=[{
            "name": "task", "id": "t1",
            "args": {"description": "Answer the question.", "subagent_type": "RunContextQAAgent"},
        }],
    )
    model = FakeChat(messages=iter([call, AIMessage(content="Specialist answer."), AIMessage(content="Specialist answer.")]))
    res = run_followup_agent("why was the analysis set up this way", _mgr(_ctx(tmp_path)), chat_model=model)
    assert res.error is None, res.error
    assert res.delegated_to == "RunContextQAAgent"
    assert "Specialist answer." in res.assistant_text


def test_harness_failure_falls_back_and_says_so(tmp_path, monkeypatch):
    import deepagents

    def boom(**kwargs):
        raise RuntimeError("harness exploded")

    monkeypatch.setattr(deepagents, "create_deep_agent", boom)
    res = run_followup_agent("why was the analysis set up this way", _mgr(_ctx(tmp_path)), chat_model=object())
    assert res.kind == FollowUpResultKind.ERROR
    assert res.delegated_to == "ErrorFallback"
    assert "harness exploded" in res.error


def test_dataset_question_is_answered_from_the_file_without_the_llm(tmp_path):
    ctx = _ctx(tmp_path, raw=_fif(tmp_path))
    res = run_followup_agent("what is this dataset about", _mgr(ctx), chat_model=ExplodingChat())
    assert res.delegated_to == "RecordingFacts"
    assert "3 EEG channels" in res.assistant_text
    assert "160 Hz" in res.assistant_text
    assert "T0" in res.assistant_text
    assert "only one annotation label" in res.assistant_text.lower()
    assert "did not look up any external dataset documentation" in res.assistant_text


def test_dataset_question_with_missing_file_still_answers(tmp_path):
    res = run_followup_agent("what is this dataset about", _mgr(_ctx(tmp_path)), chat_model=ExplodingChat())
    assert res.delegated_to == "RecordingFacts"
    assert "could not be inspected" in res.assistant_text


def test_plot_question_sends_the_stored_images_to_the_model(tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    png2 = tmp_path / "b.png"
    png2.write_bytes(b"\x89PNG\r\n\x1a\nfake2")
    ctx = _ctx(tmp_path, plots={"channels_before": str(png), "psd_overview": str(png2), "x": str(tmp_path / "gone.png")})
    model = FakeChat(messages=iter([AIMessage(content="The PSD shows power concentrated at low frequencies.")]))
    res = run_followup_agent("what does the plot tell us", _mgr(ctx), chat_model=model)
    assert res.kind == FollowUpResultKind.NEEDS_PLOT
    assert res.delegated_to == "PlotInterpreterAgent"
    assert res.image_paths == ["channels_before", "psd_overview"]
    human = FakeChat.seen[-1]
    blocks = [b for b in human.content if isinstance(b, dict)]
    assert sum(b["type"] == "image_url" for b in blocks) == 2


def test_plot_question_without_plots_says_so(tmp_path):
    res = run_followup_agent("what does the plot tell us", _mgr(_ctx(tmp_path)), chat_model=ExplodingChat())
    assert res.assistant_text == "No diagnostic plots were generated for this run."


def test_plot_model_failure_is_reported_not_hidden(tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    res = run_followup_agent("what does the plot tell us", _mgr(_ctx(tmp_path, plots={"channels_before": str(png)})), chat_model=ExplodingChat())
    assert res.kind == FollowUpResultKind.ERROR
    assert res.delegated_to == "ErrorFallback"


def test_load_plot_images_uses_the_real_plot_keys(tmp_path):
    png = tmp_path / "p.png"
    png.write_bytes(b"x")
    txt = tmp_path / "p.txt"
    txt.write_text("x")
    out = ag._load_plot_images({"channel_plot_paths": {"channels_after": str(png), "other": str(txt)}})
    assert list(out) == ["channels_after"] and out["channels_after"].startswith("data:image/png;base64,")


def test_extract_final_text_handles_langchain_messages_and_block_content():
    resp = {"messages": [AIMessage(content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}])]}
    assert ag._extract_final_text(resp) == "Hello\nworld"
    assert ag._extract_final_text({"messages": []}) == ""


@pytest.mark.parametrize("text,expected", [
    ("what is this dataset about", True),
    ("tell me about the recording", True),
    ("describe the dataset", True),
    ("which channels were used", False),
    ("explain the dataset preprocessing", False),
    ("plot the dataset before and after", False),
])
def test_recording_question_detector(text, expected):
    assert ag._is_recording_question(text) is expected
