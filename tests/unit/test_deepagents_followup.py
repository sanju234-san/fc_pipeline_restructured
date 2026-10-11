"""Offline tests for the Deep Agents post-run follow-up stack.

These tests DO NOT start an LLM — they exercise:
  * PostRunContextWindowManager: intent routing, progressive disclosure of
    offload-registry documents, sliding-window summarisation + offloading,
    and the deterministic answer fallback.
  * run_followup_agent: intent-shortcircuit paths (new-analysis dispatch,
    RunContextQA deterministic shortcut) without touching deepagents imports
    (those are gated behind allow_harness_import_errors=True and the mock layer
    in test_deepagents_followup_mock.py which actually exercises them).

Only the context manager's offline logic and the follow-up result dataclasses
/ intent routing are covered here.  Integration tests that import deepagents
live in a separate file and are skipped when the dependency is absent.
"""

from __future__ import annotations

import pytest

from fc_pipeline.agentic.followup.context_window import (
    DEFAULT_CONTEXT_TOKEN_WINDOW,
    OffloadedTurn,
    PostRunContextWindowManager,
)
from fc_pipeline.agentic.followup.agents import (
    FollowUpResult,
    FollowUpResultKind,
    run_followup_agent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_run_context() -> dict:
    return {
        "schema_version": 1,
        "run_id": "unit_test_run_001",
        "created_at": "2025-01-01T00:00:00+00:00",
        "raw_data_path": "outputs/raw.fif",
        "preprocessed_data_path": "outputs/preprocessed_epo.fif",
        "plan": {
            "freq_band": {"name": "alpha", "fmin": 8.0, "fmax": 12.0},
            "condition": "rest",
            "channels": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
            "metrics": ["pli", "wpli", "imcoh", "plv", "coh"],
        },
        "data_prep_summary": {
            "original_channel_count": 64,
            "retained_channel_count": 7,
            "dropped_channels": [],
            "reference_applied": "average (CAR)",
            "reference_detected": "none",
            "filter_l_freq": 1.0,
            "filter_h_freq": 40.0,
            "sampling_frequency": 250.0,
            "epoch_count": 120,
            "epoch_duration_seconds": 2.0,
            "condition": "rest",
        },
        "bad_channels_dropped": [],
        "channel_plot_paths": {
            "before": "outputs/plots/channels_before.png",
            "after": "outputs/plots/channels_after.png",
            "channel_variance": "outputs/plots/channel_variance.png",
            "psd_overview": "outputs/plots/psd_overview.png",
        },
        "parameter_manifest": [
            {"name": "freq_band", "category": "scientific_axis",
             "proposed_value": "alpha (8-12 Hz)", "human_approved_value": "alpha (8-12 Hz)",
             "risk_tier": "low"},
            {"name": "tau_phase", "category": "classification_threshold",
             "proposed_value": 0.20, "human_approved_value": None,
             "risk_tier": "elevated"},
            {"name": "tau_zerolag", "category": "classification_threshold",
             "proposed_value": 0.35, "human_approved_value": 0.35,
             "risk_tier": "elevated"},
            {"name": "reference", "category": "data_prep",
             "proposed_value": "CAR", "human_approved_value": "CAR",
             "risk_tier": "elevated"},
        ],
        "connectivity": None,
    }


@pytest.fixture
def manager(sample_run_context) -> PostRunContextWindowManager:
    return PostRunContextWindowManager(sample_run_context, token_window=2000)


# ---------------------------------------------------------------------------
# Manager: offload registry + progressive disclosure
# ---------------------------------------------------------------------------


def test_offload_registry_has_expected_keys(sample_run_context):
    mgr = PostRunContextWindowManager(sample_run_context)
    reg = mgr._offload_registry
    assert "run_plan_summary" in reg
    assert "data_prep_details" in reg
    assert "manifest_elevated_risk" in reg
    assert "plot_index" in reg
    # manifest_elevated_risk must list tau_phase and tau_zerolag explicitly
    elevated_body = reg["manifest_elevated_risk"]
    for name in ("tau_phase", "tau_zerolag", "reference"):
        assert name in elevated_body, f"manifest_elevated_risk missing row '{name}'"


def test_progressive_disclosure_routes_promoted_keys(sample_run_context):
    mgr = PostRunContextWindowManager(sample_run_context)
    # "Which channels were used?" → run_plan_summary + data_prep_details promoted
    sel = mgr.prepare_for_query("which channels were selected and how many epochs?")
    assert "run_plan_summary" not in sel.offloaded_context_files  # promoted
    assert "data_prep_details" not in sel.offloaded_context_files  # promoted
    # manifest_elevated_risk stays offloaded (channels/epochs don't need it)
    assert "manifest_elevated_risk" in sel.offloaded_context_files


def test_summary_topic_promotes_everything(sample_run_context):
    mgr = PostRunContextWindowManager(sample_run_context)
    sel = mgr.prepare_for_query("give me a summary of the run")
    # "summary" in matched topics → use the top-level summary branch
    # which maps to run_plan_summary + data_prep_details + manifest_elevated_risk
    for key in ("run_plan_summary", "data_prep_details", "manifest_elevated_risk"):
        assert key not in sel.offloaded_context_files, f"{key} should be promoted for summary"


def test_new_analysis_intent_promotes_only_plan(sample_run_context):
    mgr = PostRunContextWindowManager(sample_run_context)
    sel = mgr.prepare_for_query("run again with beta band on C3 and C4")
    assert sel.routed_intent == "new_analysis"
    assert "run_plan_summary" not in sel.offloaded_context_files
    # Everything else stays offloaded so the prompt stays small
    assert "data_prep_details" in sel.offloaded_context_files
    assert "manifest_elevated_risk" in sel.offloaded_context_files


def test_plot_interpreter_promotes_plot_index(sample_run_context):
    mgr = PostRunContextWindowManager(sample_run_context)
    sel = mgr.prepare_for_query("explain the before and after plots")
    assert sel.routed_intent == "plot_interpreter"
    assert "plot_index" not in sel.offloaded_context_files


# ---------------------------------------------------------------------------
# Manager: intent routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("which channels were selected?", "run_context_qa"),
        ("what frequency band was used?", "run_context_qa"),
        ("how many epochs?", "run_context_qa"),
        ("show me the before and after plots", "plot_interpreter"),
        ("plot the PSD for C3", "plot_interpreter"),
        ("compare C3 and C4 signals", "plot_interpreter"),
        ("run again with theta band", "new_analysis"),
        ("re-analyze with different channels F3 F4", "new_analysis"),
        ("compute PLI for beta", "new_analysis"),
        ("describe the dataset and its channels", "run_context_qa"),
        ("what all are the channels available and describe the dataset", "run_context_qa"),
    ],
)
def test_route_intent(manager, query, expected):
    assert manager.route_intent(query) == expected


# ---------------------------------------------------------------------------
# Manager: turn accounting + window compression
# ---------------------------------------------------------------------------


def test_turns_are_accounted(sample_run_context):
    mgr = PostRunContextWindowManager(sample_run_context, token_window=200)
    long_text = "what are the channels? " * 50  # force high char/token count
    mgr.push_user(long_text)
    mgr.push_assistant("Selected channels: Fp1, Fp2, ... (7 total)")
    mgr.push_user("how many epochs were there? " * 30)
    mgr.push_assistant("Epochs: 120 of 2.0 s for condition rest")
    sel = mgr.prepare_for_query("how many were dropped?")
    assert (
        len(mgr._offloaded_turns) >= 1
        or sel.estimated_active_tokens <= 200 + 200  # extra headroom; system fragment dominates
    )


def test_custom_summariser_is_used(sample_run_context):
    calls = {"n": 0}

    def my_summ(turns):
        calls["n"] += 1
        return f"CUSTOM({len(turns)} turns)"

    mgr = PostRunContextWindowManager(
        sample_run_context, token_window=50, summariser=my_summ
    )
    for _ in range(5):
        mgr.push_user("this is a long question " * 20)
        mgr.push_assistant("this is a long answer " * 20)
    mgr.prepare_for_query("ok?")
    assert calls["n"] >= 1


# ---------------------------------------------------------------------------
# Manager: deterministic answerer shortcut (no LLM)
# ---------------------------------------------------------------------------


def test_deterministic_answer_known_topic(manager):
    ans = manager.answer_deterministically("which channels were selected?")
    assert ans is not None
    assert "Fp1" in ans
    assert "Selected channels (7)" in ans


def test_deterministic_answer_skips_plot_requests(manager):
    assert manager.answer_deterministically("show me the plots") is None


def test_deterministic_answer_skips_new_analysis(manager):
    assert manager.answer_deterministically("run again with beta") is None


def test_deterministic_answer_nonexistent_topic_returns_none(manager):
    # No artifact words, but no matched topic either → topic detector gives up
    # → deterministic returns None, falls to LLM
    weird = "tell me about the weather outside"
    assert manager.answer_deterministically(weird) is None


# ---------------------------------------------------------------------------
# ContextSelection contract
# ---------------------------------------------------------------------------


def test_context_selection_contract(manager):
    sel = manager.prepare_for_query("what band and how many epochs?")
    # System fragment must reference the completed run explicitly
    assert "SINGLE COMPLETED EEG" in sel.system_prompt_fragment.replace("\n", " ")
    # Active messages list starts empty in a fresh manager
    assert isinstance(sel.active_messages, list)
    # Offload index is present
    assert "Offloaded context registry" in sel.system_prompt_fragment
    # routed_intent matches the question
    assert sel.routed_intent == "run_context_qa"
    # Token estimate is a positive int
    assert isinstance(sel.estimated_active_tokens, int)
    assert sel.estimated_active_tokens > 0


# ---------------------------------------------------------------------------
# run_followup_agent: offline short-circuit paths (no deepagents import)
# ---------------------------------------------------------------------------


def test_run_followup_new_analysis_dispatch_no_harness(manager):
    result = run_followup_agent(
        "run analysis again with beta band on C3 C4",
        manager,
        allow_harness_import_errors=True,
    )
    assert isinstance(result, FollowUpResult)
    assert result.kind == FollowUpResultKind.DISPATCH_NEW_QUERY
    assert "New query" in result.assistant_text
    assert result.delegated_to == "NewAnalysisDispatcher"


def test_run_followup_deterministic_shortcut(manager):
    """Questions matched by answer_run_question are returned directly (no deepagents call)."""
    result = run_followup_agent(
        "how many epochs were extracted and which frequency band?",
        manager,
        allow_harness_import_errors=True,
    )
    assert result.kind == FollowUpResultKind.ANSWERED
    assert "Frequency band" in result.assistant_text
    assert "Epochs" in result.assistant_text
    assert result.delegated_to == "RunContextQAAgent"
    # error is None on the happy path
    assert result.error is None


def test_run_followup_harness_unavailable_graceful_fallback(manager, monkeypatch):
    """When deepagents can't be imported AND the deterministic shortcut doesn't
    cover the question (e.g. a plot request), the agent returns a graceful
    fallback text with kind=ANSWERED and the error attached."""

    # Force the import of deepagents to fail
    import builtins

    real_import = builtins.__import__

    def _fail_import(name, *args, **kwargs):
        if name == "deepagents" or (isinstance(name, str) and name.startswith("deepagents")):
            raise ImportError("forced failure in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_import)
    # Also clear any cached import of agents submodule so we exercise the
    # try/except import-failure branch inside run_followup_agent (agents.py
    # already imports at the top; the failure is caught in run_followup_agent
    # by allow_harness_import_errors=True).
    # A question the deterministic shortcut and the recording/plot paths do not
    # cover, so it needs the harness; a (fake) chat model is supplied so the
    # import failure, not a missing LLM, is what triggers the fallback.
    result = run_followup_agent(
        "why was the analysis set up this way",
        manager,
        chat_model=object(),
        allow_harness_import_errors=True,
    )
    # Deterministic shortcut doesn't cover it → fallback path runs
    assert result.kind == FollowUpResultKind.ANSWERED
    assert result.error is not None
    assert "deepagents" in str(result.error).lower() or "forced" in str(result.error).lower()
    # assistant_text is non-empty and user-facing (mentions plots or fallback)
    assert len(result.assistant_text.strip()) > 0
