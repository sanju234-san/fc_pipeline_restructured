"""Chainlit routing after a completed run: follow-ups answer from the run, never restart it.

Regression for: asking "what was the channel selected in this process" after Data Prep
finished restarted the Supervisor and asked a channel clarification.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import chainlit_app as app
from fc_pipeline.deterministic.data_prep.models import DataPrepSummary
from fc_pipeline.pipeline.run_context import (
    build_run_context_from_state,
    run_context_path,
    save_run_context,
)
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand

RUN_ID = "chainlit_20261008_130000_001"


class FakeSession(dict):
    def set(self, key, value):
        self[key] = value


class Recorder:
    sent: list[str] = []

    def __init__(self, content: str = "", **kwargs):
        self.content = content

    async def send(self):
        Recorder.sent.append(self.content)
        return self


@pytest.fixture()
def run_state(tmp_path):
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
        reference_applied="average",
        condition="T0",
        epoch_count=80,
        epoch_duration_seconds=0.75,
        filter_l_freq=4.0,
        filter_h_freq=8.0,
    )
    return {
        "run_id": RUN_ID,
        "plan": plan,
        "data_prep_summary": summary,
        "parameter_manifest": [],
        "preprocessed_data_path": str(tmp_path / "outputs" / "preprocessed_epo_x.fif"),
        "channel_plot_paths": {},
        "bad_channels_dropped": [],
    }


@pytest.fixture()
def chat(monkeypatch):
    session = FakeSession()
    Recorder.sent = []
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setattr(app.cl, "Message", Recorder)

    def _no_pipeline(*args, **kwargs):
        raise AssertionError("pipeline must not start for a follow-up")

    monkeypatch.setattr(app, "_run_pipeline_sync", _no_pipeline)
    monkeypatch.setattr(app, "transform_query", _no_pipeline)
    return session


def _say(text: str):
    asyncio.run(app.on_message(SimpleNamespace(content=text)))


def test_question_after_completed_run_is_answered_without_restarting(chat, run_state):
    chat.update(completed_run_active=True, completed_run_state=run_state)
    _say("what was the channel selected in this process")
    assert len(Recorder.sent) == 1
    assert "C3, C4, Cz" in Recorder.sent[0]
    assert "Clarification" not in Recorder.sent[0]


def test_lost_flag_is_recovered_from_the_persisted_run_context(chat, run_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_run_context(
        build_run_context_from_state(run_state),
        run_context_path(run_state["preprocessed_data_path"], RUN_ID),
    )
    # Exactly the failure seen live: Data Prep finished, but the session lost its flags.
    chat.update(
        completed_run_active=False,
        completed_run_state=None,
        original_query=None,
        last_data_prep_run_id=RUN_ID,
    )
    _say("what was the channel selected in this process")
    assert len(Recorder.sent) == 1 and "C3, C4, Cz" in Recorder.sent[0]
    assert chat["completed_run_active"] is True  # run re-activated from disk


def test_flag_true_but_state_missing_is_also_recovered(chat, run_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_run_context(
        build_run_context_from_state(run_state),
        run_context_path(run_state["preprocessed_data_path"], RUN_ID),
    )
    chat.update(completed_run_active=True, completed_run_state=None, last_data_prep_run_id=RUN_ID)
    _say("which frequency band was used")
    assert "theta" in Recorder.sent[0]


def test_new_analysis_request_is_not_run_silently(chat, run_state):
    chat.update(completed_run_active=True, completed_run_state=run_state)
    _say("analyze alpha band on frontal channels")
    assert len(Recorder.sent) == 1
    assert "different analysis" in Recorder.sent[0].lower()
    assert "New query" in Recorder.sent[0]


def test_unknown_run_id_cannot_be_restored(chat, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert app._restore_completed_run("does_not_exist") is False
    assert not chat.get("completed_run_active")


def test_guards_precede_the_pipeline_and_respect_explicit_new_query():
    text = Path(app.__file__).read_text(encoding="utf-8")
    primary = text.index('completed_run_active", False) and not is_explicit_new')
    safety_net = text.index("last_run_id = cl.user_session.get(\"last_data_prep_run_id\")")
    pipeline = text.index("# --- Invoke Pipeline ---")
    assert primary < safety_net < pipeline
    # the safety net also honours the explicit "new query:" unlock
    assert "and not is_explicit_new" in text[safety_net:pipeline]


# --- follow-up routing added with the harness fix ----------------------------


def test_named_channel_plot_request_is_drawn_deterministically_before_any_llm(chat, run_state, monkeypatch):
    chat.update(completed_run_active=True, completed_run_state=run_state)
    calls = []

    async def fake_channel_request(text, low, state):
        calls.append(text)
        return True

    def must_not_run(*a, **k):
        raise AssertionError("the language model harness must not run for a channel plot")

    monkeypatch.setattr(app, "_followup_channel_request", fake_channel_request)
    monkeypatch.setattr(app, "run_followup_agent", must_not_run)
    _say("Show me the EEG signal for C3")
    assert calls == ["Show me the EEG signal for C3"]


def test_plot_interpreter_failure_falls_back_to_the_deterministic_explanation(chat, run_state, monkeypatch):
    from fc_pipeline.agentic.followup.agents import FollowUpResult, FollowUpResultKind

    chat.update(completed_run_active=True, completed_run_state=run_state)
    explained = []

    async def no_channels(text, low, state):
        return False

    async def fake_explain(state):
        explained.append(True)

    monkeypatch.setattr(app, "_followup_channel_request", no_channels)
    monkeypatch.setattr(app, "_followup_explain", fake_explain)
    monkeypatch.setattr(
        app, "run_followup_agent",
        lambda text, manager: FollowUpResult(
            kind=FollowUpResultKind.ERROR, assistant_text="x", delegated_to="ErrorFallback", error="no vision"
        ),
    )
    _say("what does the plot tell us")
    assert explained == [True]
    assert not any("Cannot process" in m for m in Recorder.sent)
