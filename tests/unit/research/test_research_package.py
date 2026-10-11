"""Research package: network-free tests (ddgs, Deep Agents, DSPy)."""

import sys
import types

import pytest

from fc_pipeline.research import research_agent, web_search
from fc_pipeline.research.web_search import search_web, web_search_tool


class _FakeDDGS:
    def __init__(self, timeout=5):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def text(self, q, max_results=5):
        return [{"title": "wPLI", "href": "https://example.org/wpli", "body": "x" * 1000}] * 3


def _install_fake(monkeypatch, cls):
    mod = types.ModuleType("ddgs")
    mod.DDGS = cls
    monkeypatch.setitem(sys.modules, "ddgs", mod)


def test_search_normalises_and_truncates(monkeypatch):
    _install_fake(monkeypatch, _FakeDDGS)
    out = search_web("wpli", max_results=2)
    assert out["ok"] and len(out["results"]) == 2
    assert out["results"][0]["url"] == "https://example.org/wpli"
    assert len(out["results"][0]["snippet"]) == web_search.SNIPPET_CHARS


def test_search_failure_never_raises(monkeypatch):
    class Boom(_FakeDDGS):
        def text(self, *a, **k):
            raise TimeoutError("offline")

    _install_fake(monkeypatch, Boom)
    out = search_web("wpli")
    assert out["ok"] is False and "TimeoutError" in out["error"]
    assert web_search_tool("wpli").startswith("SEARCH_UNAVAILABLE")


def test_empty_query_rejected():
    assert search_web("   ")["error"] == "empty_query"


def test_research_agent_is_off_by_default(monkeypatch):
    monkeypatch.delenv("FC_RESEARCH_ENABLED", raising=False)
    with pytest.raises(RuntimeError):
        research_agent.build_research_agent(llm=object())


def test_research_prompt_forbids_dataset_claims():
    p = research_agent.RESEARCH_SYSTEM_PROMPT
    assert "untrusted" in p and "never state or guess" in p and "ICA" in p


def test_dspy_scope_module_and_metric():
    dspy = pytest.importorskip("dspy")
    from dspy.utils import DummyLM

    from fc_pipeline.research.dspy_scope import InputScopeClassifier, labelled_examples, scope_metric

    examples = labelled_examples()
    assert {e.verdict for e in examples} == {"allow", "block"}
    with dspy.context(lm=DummyLM([{"verdict": "allow", "reason": "EEG request"}])):
        pred = InputScopeClassifier()(message="analyze this EEG")
    allow = next(e for e in examples if e.verdict == "allow")
    block = next(e for e in examples if e.verdict == "block")
    assert scope_metric(allow, pred) == 1.0
    assert scope_metric(block, pred) == 0.0  # a missed attack scores worst
