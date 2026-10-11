"""INPUT_RAIL_BACKEND selects which classifier backs the input rail."""

import pytest

from fc_pipeline.pipeline import nodes
from fc_pipeline.research import dspy_rail


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("NEMO_GUARDRAILS_ENABLED", "true")
    monkeypatch.delenv("NEMO_INPUT_FASTPATH", raising=False)
    monkeypatch.delenv("INPUT_RAIL_BACKEND", raising=False)


def _nemo_must_not_run(*a, **k):
    raise AssertionError("NeMo input check must not run with the dspy backend")


def _dspy_must_not_run(*a, **k):
    raise AssertionError("DSPy check must not run with the default nemo backend")


def test_dspy_backend_is_used_and_nemo_is_not_called(monkeypatch):
    monkeypatch.setenv("INPUT_RAIL_BACKEND", "dspy")
    seen = []
    monkeypatch.setattr(nodes, "_check_rail", _nemo_must_not_run)
    monkeypatch.setattr(
        dspy_rail, "check_input_with_dspy",
        lambda text: seen.append(text) or {"status": "passed", "content": text, "rail": "x", "error": None},
    )
    assert nodes.apply_input_rail({"run_id": "r1"}, "give me a pasta recipe") is None
    assert seen == ["give me a pasta recipe"]


def test_dspy_block_pauses_for_a_human_like_nemo_does(monkeypatch):
    monkeypatch.setenv("INPUT_RAIL_BACKEND", "dspy")
    monkeypatch.setattr(
        dspy_rail, "check_input_with_dspy",
        lambda text: {"status": "blocked", "content": "off topic", "rail": "dspy input scope", "error": None},
    )
    out = nodes.apply_input_rail({"run_id": "r1"}, "give me a pasta recipe")
    assert out["decision_context"]["kind"] == "input_rail"
    assert out["gate_1_approved"] is False


def test_dspy_failure_pauses_with_the_error_recorded(monkeypatch):
    monkeypatch.setenv("INPUT_RAIL_BACKEND", "dspy")
    monkeypatch.setattr(
        dspy_rail, "check_input_with_dspy",
        lambda text: {"status": "error", "content": "", "rail": None, "error": "TimeoutError: down"},
    )
    out = nodes.apply_input_rail({"run_id": "r1"}, "give me a pasta recipe")
    assert out["decision_context"]["details"]["rail_status"] == "error"
    assert "TimeoutError" in out["decision_context"]["reason"]


def test_default_backend_is_still_nemo(monkeypatch):
    monkeypatch.setattr(dspy_rail, "check_input_with_dspy", _dspy_must_not_run)
    called = []
    monkeypatch.setattr(
        nodes, "_check_rail",
        lambda rt, m: called.append(rt) or {"status": "passed", "content": "", "rail": None, "error": None},
    )
    assert nodes.apply_input_rail({"run_id": "r1"}, "give me a pasta recipe") is None
    assert called == ["input"]


def test_prefilter_still_runs_first_with_the_dspy_backend(monkeypatch):
    monkeypatch.setenv("INPUT_RAIL_BACKEND", "dspy")
    monkeypatch.setattr(dspy_rail, "check_input_with_dspy", _dspy_must_not_run)
    assert nodes.apply_input_rail({"run_id": "r1"}, "analyze this EEG") is None
