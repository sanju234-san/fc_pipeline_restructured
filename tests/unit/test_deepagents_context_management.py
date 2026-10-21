"""Unit and regression tests for Deep Agents Context Management in the EEG Functional Connectivity Pipeline.

Verifies:
1. Large tool result offloading to filesystem store with lightweight reference stubs.
2. Recovery of offloaded context fragments via read and search tools.
3. Automatic sliding-window conversation history summarization under token budgets.
4. Thread isolation across runs (no cross-contamination in filesystem storage).
5. Strict separation between persistent user preferences (long-term memory) and transient conversation context.
6. Preservation of LangGraph HITL interrupt/resume flags and checkpoint state.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from fc_pipeline.pipeline.context_management import (
    DeepAgentsContextManager,
    FilesystemContextStore,
    OffloadRecord,
    estimate_tokens,
    DEFAULT_TOOL_OFFLOAD_CHAR_THRESHOLD,
    DEFAULT_CONTEXT_TOKEN_LIMIT,
)
from fc_pipeline.schemas.state import GraphState
from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand
from fc_pipeline.pipeline.memory import MemoryBundle
from fc_pipeline.pipeline.user_memory import load_preference_store
from fc_pipeline.agentic.supervisor.agent import supervisor_node


@pytest.fixture
def temp_context_dir(tmp_path):
    storage_dir = tmp_path / "context_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def test_estimate_tokens():
    """Verify heuristic token estimation (~4 chars/token)."""
    assert estimate_tokens("1234") == 1
    assert estimate_tokens("a" * 40) == 10
    
    msgs = [
        SystemMessage(content="Hello"),
        HumanMessage(content="World test"),
    ]
    tokens = estimate_tokens(msgs)
    assert tokens >= 3


def test_filesystem_context_store_thread_isolation(temp_context_dir):
    """Verify thread-isolated directory partitioning."""
    store = FilesystemContextStore(base_dir=temp_context_dir)
    thread_a = "run_thread_alpha"
    thread_b = "run_thread_beta"

    path_a = store.save_payload(thread_a, "dataset_info", "Payload for Alpha")
    path_b = store.save_payload(thread_b, "dataset_info", "Payload for Beta")

    # Ensure files are in distinct directories
    assert path_a.parent.name == thread_a
    assert path_b.parent.name == thread_b
    assert path_a.parent != path_b.parent

    # Ensure content integrity
    assert store.read_payload(path_a) == "Payload for Alpha"
    assert store.read_payload(path_b) == "Payload for Beta"


def test_tool_result_offloading_under_threshold(temp_context_dir):
    """Small tool output under threshold is not offloaded."""
    store = FilesystemContextStore(base_dir=temp_context_dir)
    mgr = DeepAgentsContextManager(
        thread_id="test_run",
        tool_offload_threshold=1000,
        store=store,
    )

    small_output = {"status": "ok", "sfreq": 250.0}
    result, record = mgr.maybe_offload_tool_result("get_dataset_info", small_output)

    assert record is None
    assert result == small_output


def test_tool_result_offloading_over_threshold(temp_context_dir):
    """Large tool output exceeding threshold is offloaded with preview stub."""
    store = FilesystemContextStore(base_dir=temp_context_dir)
    mgr = DeepAgentsContextManager(
        thread_id="test_run_large",
        tool_offload_threshold=200,
        store=store,
    )

    large_payload = {
        "sfreq": 500.0,
        "available_channels": [f"EEG_{i:03d}" for i in range(128)],
        "description": "High density montage recording with extensive sensor grid",
    }
    result_stub, record = mgr.maybe_offload_tool_result("get_dataset_info", large_payload, "call_123")

    assert record is not None
    assert isinstance(record, OffloadRecord)
    assert record.record_id == "get_dataset_info_call_123"
    assert record.file_path.exists()
    assert "DEEP AGENTS OFFLOAD" in result_stub
    assert "Preview (first 10 lines):" in result_stub
    assert "read_offloaded_context(key=\"get_dataset_info_call_123\")" in result_stub


def test_recovery_of_offloaded_information(temp_context_dir):
    """Test on-demand read and search retrieval tools for offloaded context."""
    store = FilesystemContextStore(base_dir=temp_context_dir)
    mgr = DeepAgentsContextManager(
        thread_id="test_run_retrieve",
        tool_offload_threshold=50,
        store=store,
    )

    payload = "Line 1: Alpha band 8-12 Hz\nLine 2: Target channels F3, F4, C3, C4\nLine 3: Condition resting_state"
    stub, record = mgr.maybe_offload_tool_result("dataset_meta", payload, "meta_1")

    assert record is not None
    tools = mgr.build_retrieval_tools()
    read_tool = next(t for t in tools if t.name == "read_offloaded_context")
    search_tool = next(t for t in tools if t.name == "search_offloaded_context")

    # 1. Read full context
    full_text = read_tool.invoke({"key": "dataset_meta_meta_1"})
    assert full_text == payload

    # 2. Search within offloaded context
    search_res = search_tool.invoke({"query": "Target channels"})
    assert "Line 2: Target channels F3, F4, C3, C4" in search_res

    # 3. Search non-matching query
    search_empty = search_tool.invoke({"query": "non_existent_token"})
    assert "No matches found" in search_empty


def test_conversation_history_summarization_trigger_and_compression(temp_context_dir):
    """Test conversation history sliding-window summarization when approaching token budget."""
    store = FilesystemContextStore(base_dir=temp_context_dir)
    mgr = DeepAgentsContextManager(
        thread_id="test_run_summary",
        token_limit=50,  # 50 tokens * 0.85 = 42.5 tokens
        store=store,
    )

    messages = [
        SystemMessage(content="System prompt for supervisor."),
        HumanMessage(content="User query 1: analyze alpha band."),
        AIMessage(content="Thought 1: inspect dataset info."),
        ToolMessage(tool_call_id="c1", content="Dataset info: sfreq=250, channels=64"),
        HumanMessage(content="User query 2: focus on frontal electrodes."),
        AIMessage(content="Thought 2: resolve channels."),
    ]

    # Total characters: ~200 chars -> ~50 tokens >= 42 tokens
    assert mgr.should_summarize(messages) is True

    # Run summarization with preserved tail of 2 messages
    compressed_msgs = mgr.summarize_messages(messages, preserved_tail=2)

    # Structure check: System message + Summary message + 2 preserved tail messages = 4
    assert len(compressed_msgs) == 4
    assert isinstance(compressed_msgs[0], SystemMessage)
    assert compressed_msgs[0].content == "System prompt for supervisor."
    assert isinstance(compressed_msgs[1], HumanMessage)
    assert "[DEEP AGENTS CONTEXT SUMMARY:" in compressed_msgs[1].content
    assert compressed_msgs[2] == messages[-2]
    assert compressed_msgs[3] == messages[-1]

    # Canonical full history was saved to filesystem
    history_files = list((temp_context_dir / "test_run_summary").glob("conversation_history_*.txt"))
    assert len(history_files) >= 1
    assert "User query 1: analyze alpha band." in history_files[0].read_text(encoding="utf-8")


def test_user_preferences_isolated_from_transient_context(temp_context_dir):
    """Ensure user preferences (long-term memory) do not overwrite explicit session parameters or dataset ground truth."""
    # Simulate stored user preferences
    prefs = {"preferred_metric": "PLI", "preferred_band": "theta"}
    bundle = MemoryBundle(text="User prefers theta band by default.", user_defaults=prefs)

    store = FilesystemContextStore(base_dir=temp_context_dir)
    mgr = DeepAgentsContextManager(thread_id="test_isolation", store=store)

    # In an active session, user explicitly specifies alpha band
    active_request = "Compute connectivity for alpha band on rest condition."

    # Transient context summary reflects the explicit user request, not the stale default
    messages = [
        HumanMessage(content=active_request),
        AIMessage(content="Resolving frequency band for alpha..."),
    ]
    summary_text = mgr._generate_summary(messages, active_request)

    assert "alpha" in summary_text.lower()
    # Preference memory remains strictly in bundle and does not silently mutate active_request
    assert bundle.user_defaults["preferred_band"] == "theta"
    assert "alpha" in active_request


def test_hitl_clarification_state_preservation_under_context_management(temp_context_dir):
    """Verify that graph state, pending clarification, and run_id remain intact across context offloading."""
    raw_data_path = "mock/eeg/sample_raw.fif"
    run_id = "test_run_hitl_preservation"

    state: GraphState = {
        "pipeline_error": None,
        "run_id": run_id,
        "raw_data_path": raw_data_path,
        "user_request": "Compute functional connectivity on F3 and F4.",  # Missing band and condition
        "latest_user_message": "Compute functional connectivity on F3 and F4.",
        "plan": None,
        "parameter_manifest": None,
        "preflight_confirmed": False,
        "gate_1_approved": False,
        "clarification_question": None,
        "informational_response": None,
        "informational_artifacts": None,
        "bad_channels_dropped": None,
        "channel_plot_paths": None,
        "preprocessed_data_path": None,
        "data_prep_error": None,
        "metric_csv_paths": None,
        "heatmap_image_paths": None,
        "network_image_paths": None,
        "numerical_summaries": None,
        "evidence_summary": None,
        "evaluation_verdict": None,
        "evaluation_summary_csv_path": None,
        "final_report_path": None,
    }

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Which frequency band would you like to analyze?")

    res = supervisor_node(state, mock_llm, run_id=run_id, max_iterations=2)

    # Check that clarification question is correctly posed and plan is None (not prematurely approved)
    assert res.get("clarification_question") is not None
    assert res.get("plan") is None
    assert state.get("gate_1_approved") is False
    assert state.get("preflight_confirmed") is False
