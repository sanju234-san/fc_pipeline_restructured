"""Chainlit review of a NeMo guardrail pause: a human decides, nothing auto-approves.

Regression for: every rail pause was shown as a hard "Cannot process this query."
with no way to clear it, so benign requests such as "analyze this EEG" were lost.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import chainlit_app as app

RUN_ID = "chainlit_20261009_120000_001"


def _ctx(kind="input_rail", status="blocked", reason="flagged"):
    return {
        "kind": kind,
        "reason": reason,
        "details": {"rail_status": status, "flagged_text": "analyze this EEG"},
        "allowed_responses": ["approve", "reject"],
    }


class _Msg:
    def __init__(self, content=""):
        self.content = content
        self.updated = 0

    async def update(self):
        self.updated += 1


@pytest.fixture()
def ui(monkeypatch):
    rec = SimpleNamespace(sent=[], resumed=[], rendered=[], reset=0, choice="approve", ask_content=[])

    class FakeMessage:
        def __init__(self, content="", **kw):
            self.content = content

        async def send(self):
            rec.sent.append(self.content)
            return self

    class FakeAsk:
        def __init__(self, content="", actions=None, timeout=0):
            rec.ask_content.append(content)
            rec.actions = [a.name for a in actions]

        async def send(self):
            return None if rec.choice is None else {"name": rec.choice}

    def fake_make_async(fn):
        async def wrapper(*a, **k):
            return fn(*a, **k)

        return wrapper

    def fake_resume(run_id, action, *a, **k):
        rec.resumed.append(action)
        return rec.resume_result

    async def fake_output(state, nodes, query, run_id, msg):
        rec.rendered.append(state)

    rec.resume_result = ({"_executed_nodes": ["supervisor"], "informational_response": "held text"}, None)
    monkeypatch.setattr(app.cl, "Message", FakeMessage)
    monkeypatch.setattr(app.cl, "AskActionMessage", FakeAsk)
    monkeypatch.setattr(app.cl, "Action", lambda name, payload, label, description="": SimpleNamespace(name=name))
    monkeypatch.setattr(app.cl, "make_async", fake_make_async)
    monkeypatch.setattr(app, "_resume_pipeline_sync", fake_resume)
    monkeypatch.setattr(app, "_handle_pipeline_output", fake_output)
    monkeypatch.setattr(app, "_reset_conversation_state", lambda: setattr(rec, "reset", rec.reset + 1))
    return rec


def _run(ctx, accumulated="analyze this EEG"):
    msg = _Msg()
    asyncio.run(app._handle_rail_decision({"decision_context": ctx}, RUN_ID, msg, accumulated))
    return msg


def test_input_rail_offers_allow_and_block_not_a_hard_stop(ui):
    msg = _run(_ctx())
    assert ui.actions == ["approve", "reject"]
    assert "Cannot process" not in msg.content
    assert "analyze this EEG" in msg.content


def test_human_approval_resumes_and_continues_the_pipeline(ui):
    _run(_ctx())
    assert ui.resumed == ["approve"]
    assert len(ui.rendered) == 1


def test_reject_keeps_it_blocked(ui):
    ui.choice = "reject"
    _run(_ctx())
    assert ui.resumed == ["reject"]
    assert not ui.rendered and ui.reset == 1
    assert any("blocked" in s.lower() for s in ui.sent)


def test_timeout_is_never_an_approval(ui):
    ui.choice = None
    _run(_ctx())
    assert ui.resumed == ["reject"]
    assert not ui.rendered
    assert any("timed out" in s.lower() for s in ui.sent)


def test_check_failure_shows_the_real_error_not_a_refusal(ui):
    msg = _run(_ctx(status="error", reason="The input guardrail check could not run (RateLimitError: 429); failing closed."))
    assert "unavailable" in msg.content.lower()
    assert "RateLimitError" in msg.content
    assert "not a judgement about your request" in msg.content


def test_output_rail_approval_shows_the_held_response(ui):
    _run(_ctx(kind="output_rail"))
    assert ui.resumed == ["approve"]
    assert "held text" in ui.sent
    assert not ui.rendered


def test_resume_failure_is_reported(ui):
    ui.resume_result = ({}, "Graph is not initialized.")
    _run(_ctx())
    assert any("Could not resume" in s for s in ui.sent)
    assert not ui.rendered
