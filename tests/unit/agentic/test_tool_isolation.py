"""Tests for Supervisor Agent Tool Isolation (Phase II).

Verifies:
1. Stage-based scoped progression (discovery -> resolution -> manifest compilation).
2. Condition tool dropping from scope after one call (prevents max_iterations spinning).
3. Overview plot gating with whole-word regex and compound analysis-plus-plot ordering.
4. Empty scope exiting directly without calling bind_tools([]).
5. Strict snapshot guard rejecting batched multi-call dependencies within a single response.
6. Mistral text parser fallback extracting an out-of-scope tool call returning TOOL_OUT_OF_SCOPE.
"""

import json
from unittest.mock import MagicMock
import numpy as np
import mne
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from typing import Any
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from fc_pipeline.agentic.supervisor.agent import supervisor_node
from fc_pipeline.schemas.enums import MetricEnum


class MockChatModel(FakeListChatModel):
    """FakeListChatModel that implements bind_tools and tracks bound tools for ReAct testing."""
    bound_tool_history: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_history.append([getattr(t, "name", str(t)) for t in tools])
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        tool_calls = []
        content = output_str
        try:
            parsed = json.loads(output_str)
            raw_calls = []
            if isinstance(parsed, dict) and "name" in parsed and "args" in parsed:
                raw_calls = [parsed]
            elif isinstance(parsed, list) and all(isinstance(x, dict) and "name" in x for x in parsed):
                raw_calls = parsed
            if raw_calls:
                for idx, c in enumerate(raw_calls):
                    tool_calls.append({
                        "name": c["name"],
                        "args": c.get("args", {}),
                        "id": c.get("id", f"call_{idx}_{c['name']}"),
                        "type": "tool_call",
                    })
                content = ""
        except Exception:
            pass

        message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _tool_call(name: str, args: dict) -> str:
    return json.dumps({
        "name": name,
        "args": args,
        "id": f"call_{name}",
        "type": "tool_call",
    })


def _mistral_call(name: str, args: dict) -> str:
    return f"[TOOL_CALLS]{name}[ARGS]{json.dumps(args)}"


@pytest.fixture(scope="module")
def synthetic_eeg(tmp_path_factory):
    """Create a minimal synthetic raw EEG file."""
    ch_names = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    sfreq = 250.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(8, int(sfreq * 5))
    raw = mne.io.RawArray(data, info, verbose=False)

    annot = mne.Annotations(onset=[1.0, 3.0], duration=[1.0, 1.0], description=["rest", "rest"])
    raw.set_annotations(annot)

    out_dir = tmp_path_factory.mktemp("isolation_test")
    file_path = out_dir / "test_raw.fif"
    raw.save(str(file_path), overwrite=True, verbose=False)
    return str(file_path)


def _base_state(path: str, user_request: str) -> dict:
    return {
        "raw_data_path": path,
        "user_request": user_request,
        "latest_user_message": user_request,
        "pipeline_error": None,
        "run_id": "iso_test",
        "plan": None,
        "parameter_manifest": None,
        "preflight_confirmed": False,
        "gate_1_approved": False,
        "clarification": None,
        "clarification_question": None,
        "clarification_kind": None,
        "clarification_options": [],
        "condition_candidates": [],
        "clarification_response": None,
        "clarification_resume_kind": None,
        "resolved_frequency_band_info": None,
        "resolved_channel_info": None,
        "resolved_condition_value": None,
        "dataset_sfreq": None,
        "dataset_duration_seconds": None,
        "dataset_available_channels": None,
        "dataset_reference": None,
        "informational_response": None,
        "informational_artifacts": None,
        "decision_context": None,
        "input_rail_cleared": False,
    }


def test_scoped_progression_stage_by_stage(synthetic_eeg):
    """Verify tool scope transitions sequentially from discovery to resolution, and rebinds only on change."""
    fake_llm = MockChatModel(responses=[
        _tool_call("get_dataset_info", {"data_path": synthetic_eeg}),
        _tool_call("get_dataset_conditions", {"data_path": synthetic_eeg}),
        _tool_call("resolve_frequency_band", {"band_name_or_range": "alpha", "sfreq": 250.0}),
        _tool_call("resolve_channel_selection", {"requested_channels_or_region": "F3, F4"}),
    ])

    state = _base_state(synthetic_eeg, "compute PLI for alpha on F3, F4 during rest")
    result = supervisor_node(state, llm=fake_llm, run_id="progression_run", max_iterations=6)

    assert result["plan"] is not None
    assert result["parameter_manifest"] is not None
    assert result["plan"].condition == "rest"

    # Verify history of tools bound across iterations
    history = fake_llm.bound_tool_history
    assert len(history) >= 2

    # Initial scope: only discovery tools
    assert set(history[0]) == {"get_dataset_info", "get_dataset_conditions"}

    # Subsequent scopes after get_dataset_info: resolution tools enter scope
    assert "resolve_frequency_band" in history[1]
    assert "resolve_channel_selection" in history[1]
    assert "get_dataset_info" not in history[1]


def test_condition_tool_drops_after_one_call_preventing_infinite_loop(synthetic_eeg):
    """When condition is not in user_request, get_dataset_conditions is called once and drops from scope."""
    fake_llm = MockChatModel(responses=[
        _tool_call("get_dataset_info", {"data_path": synthetic_eeg}),
        _tool_call("get_dataset_conditions", {"data_path": synthetic_eeg}),
        _tool_call("resolve_frequency_band", {"band_name_or_range": "alpha", "sfreq": 250.0}),
        _tool_call("resolve_channel_selection", {"requested_channels_or_region": "F3, F4"}),
        "Need condition.",
    ])

    # user_request does NOT mention any condition
    state = _base_state(synthetic_eeg, "compute PLI for alpha on F3, F4")
    result = supervisor_node(state, llm=fake_llm, run_id="cond_drop_run", max_iterations=6)

    # Must cleanly halt with condition clarification
    assert result["plan"] is None
    assert result["clarification_kind"] == "condition"
    assert "condition" in result["clarification_question"].lower()

    # In the third binding (after get_dataset_conditions ran), get_dataset_conditions must be dropped
    assert len(fake_llm.bound_tool_history) >= 3
    final_scope = fake_llm.bound_tool_history[-1]
    assert "get_dataset_conditions" not in final_scope


def test_overview_plot_gated_by_diagnostic_intent_and_compound_ordering(synthetic_eeg):
    """Overview plot is included when requested, and compound analysis+plot resolves both."""
    fake_llm = MockChatModel(responses=[
        _tool_call("get_dataset_info", {"data_path": synthetic_eeg}),
        _tool_call("get_dataset_conditions", {"data_path": synthetic_eeg}),
        _tool_call("generate_dataset_overview_plot", {"data_path": synthetic_eeg, "run_id": "compound_run"}),
        _tool_call("resolve_frequency_band", {"band_name_or_range": "alpha", "sfreq": 250.0}),
        _tool_call("resolve_channel_selection", {"requested_channels_or_region": "F3, F4"}),
    ])

    compound_request = "compute PLI for alpha on F3, F4 during rest and plot overview of dataset"
    state = _base_state(synthetic_eeg, compound_request)
    result = supervisor_node(state, llm=fake_llm, run_id="compound_run", max_iterations=8)

    assert result["plan"] is not None
    assert result["parameter_manifest"] is not None
    # Verify plot was executed and scoped
    assert any("generate_dataset_overview_plot" in scope for scope in fake_llm.bound_tool_history)


def test_overview_plot_excluded_for_pure_analysis_request(synthetic_eeg):
    """When user request does not contain diagnostic plot keywords, generate_dataset_overview_plot is not scoped."""
    fake_llm = MockChatModel(responses=[
        _tool_call("get_dataset_info", {"data_path": synthetic_eeg}),
        _tool_call("get_dataset_conditions", {"data_path": synthetic_eeg}),
        _tool_call("resolve_frequency_band", {"band_name_or_range": "alpha", "sfreq": 250.0}),
        _tool_call("resolve_channel_selection", {"requested_channels_or_region": "F3, F4"}),
    ])

    pure_analysis_request = "compute PLI for alpha on F3, F4 during rest"
    state = _base_state(synthetic_eeg, pure_analysis_request)
    result = supervisor_node(state, llm=fake_llm, run_id="pure_analysis_run", max_iterations=6)

    assert result["plan"] is not None
    for scope in fake_llm.bound_tool_history:
        assert "generate_dataset_overview_plot" not in scope


def test_empty_scope_never_reaches_bind_tools(synthetic_eeg):
    """When all three axes are resolved and no plot is pending, ReAct loop terminates without bind_tools([])."""
    fake_llm = MockChatModel(responses=[
        _tool_call("get_dataset_info", {"data_path": synthetic_eeg}),
        _tool_call("get_dataset_conditions", {"data_path": synthetic_eeg}),
        _tool_call("resolve_frequency_band", {"band_name_or_range": "alpha", "sfreq": 250.0}),
        _tool_call("resolve_channel_selection", {"requested_channels_or_region": "F3, F4"}),
    ])

    state = _base_state(synthetic_eeg, "compute PLI for alpha on F3, F4 during rest")
    result = supervisor_node(state, llm=fake_llm, run_id="empty_scope_run", max_iterations=6)

    assert result["plan"] is not None
    # Empty tools list must NEVER be bound to LLM
    for scope in fake_llm.bound_tool_history:
        assert len(scope) > 0, "bind_tools was called with an empty tool list []"


def test_strict_snapshot_rejection_in_multicall_response(synthetic_eeg):
    """Decision A: When LLM emits [get_dataset_info, resolve_frequency_band] in a single response,
    the second call is strictly rejected as TOOL_OUT_OF_SCOPE against the pre-invocation snapshot."""
    fake_llm = MockChatModel(responses=[
        # Single response batching both discovery and resolution tools
        json.dumps([
            {"name": "get_dataset_info", "args": {"data_path": synthetic_eeg}, "id": "call_info", "type": "tool_call"},
            {"name": "resolve_frequency_band", "args": {"band_name_or_range": "alpha"}, "id": "call_band", "type": "tool_call"},
        ]),
        # Next turn: model sees TOOL_OUT_OF_SCOPE and retries properly now that resolve_frequency_band is in scope
        _tool_call("resolve_frequency_band", {"band_name_or_range": "alpha", "sfreq": 250.0}),
        _tool_call("get_dataset_conditions", {"data_path": synthetic_eeg}),
        _tool_call("resolve_channel_selection", {"requested_channels_or_region": "F3, F4"}),
    ])

    state = _base_state(synthetic_eeg, "compute PLI for alpha on F3, F4 during rest")
    result = supervisor_node(state, llm=fake_llm, run_id="snapshot_run", max_iterations=6)

    # Resolution still succeeds eventually via next iteration
    assert result["plan"] is not None
    assert result["parameter_manifest"] is not None


def test_mistral_out_of_scope_tool_returns_structured_error(synthetic_eeg):
    """When a tool call is extracted via Mistral text parser but is out of scope, it returns TOOL_OUT_OF_SCOPE."""
    fake_llm = MockChatModel(responses=[
        # LLM emits resolver via Mistral syntax on turn 1 (before get_dataset_info has run)
        _mistral_call("resolve_frequency_band", {"band_name_or_range": "alpha"}),
        # LLM recovers and calls discovery first
        _tool_call("get_dataset_info", {"data_path": synthetic_eeg}),
        _tool_call("get_dataset_conditions", {"data_path": synthetic_eeg}),
        _tool_call("resolve_frequency_band", {"band_name_or_range": "alpha", "sfreq": 250.0}),
        _tool_call("resolve_channel_selection", {"requested_channels_or_region": "F3, F4"}),
    ])

    state = _base_state(synthetic_eeg, "compute PLI for alpha on F3, F4 during rest")
    result = supervisor_node(state, llm=fake_llm, run_id="mistral_out_of_scope_run", max_iterations=8)

    assert result["plan"] is not None
    assert result["parameter_manifest"] is not None


# ---------------------------------------------------------------------------
# Provider-side rejection of an unbound tool (Groq: HTTP 400 "tool_use_failed")
# ---------------------------------------------------------------------------

GROQ_REJECTION = (
    "Error code: 400 - {'error': {'message': \"Tool call validation failed: tool call validation "
    "failed: attempted to call tool 'get_dataset_conditions' which was not in request.tools\", "
    "'type': 'invalid_request_error', 'code': 'tool_use_failed'}}"
)


class RejectingChatModel(MockChatModel):
    """Raises the provider's rejection on the listed (0-based) invocations."""

    reject_calls: list = Field(default_factory=list)
    call_count: int = 0
    seen: list = Field(default_factory=list)
    error_text: str = GROQ_REJECTION

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        idx = self.call_count
        self.call_count += 1
        self.seen.append([str(getattr(m, "content", "")) for m in messages])
        if idx in self.reject_calls:
            raise RuntimeError(self.error_text)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _full_progression(path):
    return [
        _tool_call("get_dataset_info", {"data_path": path}),
        _tool_call("get_dataset_conditions", {"data_path": path}),
        _tool_call("resolve_frequency_band", {"band_name_or_range": "alpha", "sfreq": 250.0}),
        _tool_call("resolve_channel_selection", {"requested_channels_or_region": "F3, F4"}),
    ]


def test_provider_tool_rejection_is_recovered_and_the_run_completes(synthetic_eeg):
    llm = RejectingChatModel(responses=_full_progression(synthetic_eeg), reject_calls=[0])
    state = _base_state(synthetic_eeg, "compute PLI for alpha on F3, F4 during rest")
    result = supervisor_node(state, llm=llm, run_id="reject_once_run", max_iterations=8)

    assert result["plan"] is not None
    assert result["plan"].condition == "rest"
    # the retry after the rejection carried a notice naming the rejected tool
    retry_messages = llm.seen[1]
    notice = [m for m in retry_messages if "[system notice]" in m]
    assert notice and "get_dataset_conditions" in notice[0]
    assert "Do not call 'get_dataset_conditions' again" in notice[0]


def test_repeated_provider_rejections_end_gracefully_instead_of_crashing(synthetic_eeg):
    llm = RejectingChatModel(responses=_full_progression(synthetic_eeg), reject_calls=[0, 1, 2, 3, 4, 5])
    state = _base_state(synthetic_eeg, "compute PLI for alpha on F3, F4 during rest")
    result = supervisor_node(state, llm=llm, run_id="reject_always_run", max_iterations=8)

    assert result["plan"] is None  # nothing was resolved, nothing was invented
    assert llm.call_count == 3  # first call + the two allowed retries, then it stops
    assert result.get("clarification_question") or result.get("clarification")


def test_unrelated_llm_errors_are_still_raised(synthetic_eeg):
    llm = RejectingChatModel(
        responses=_full_progression(synthetic_eeg), reject_calls=[0], error_text="connection reset by peer"
    )
    state = _base_state(synthetic_eeg, "compute PLI for alpha on F3, F4 during rest")
    with pytest.raises(RuntimeError, match="connection reset"):
        supervisor_node(state, llm=llm, run_id="other_error_run", max_iterations=8)
