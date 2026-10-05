"""Unit tests for Query Transformer module."""

import pytest
from unittest.mock import MagicMock

from fc_pipeline.agentic.supervisor.query_transformer import (
    QUERY_TRANSFORMER_SYSTEM_PROMPT,
    QueryTransformerResult,
    _parse_transformer_output,
    _build_fallback,
    transform_query,
)


class TestQueryTransformerResultProperties:
    """Test property helpers on QueryTransformerResult dataclass."""

    def test_clean_result_properties(self):
        res = QueryTransformerResult(
            intent="eeg_analysis",
            condensed="compute PLI for alpha on F3, F4 during rest",
            contradiction="none",
            clarification="none",
        )
        assert not res.is_out_of_scope
        assert not res.has_contradiction
        assert not res.is_blocked
        assert not res.has_clarification

    def test_out_of_scope_properties(self):
        res = QueryTransformerResult(
            intent="out_of_scope",
            condensed="Request is about weather",
            contradiction="none",
            clarification="none",
        )
        assert res.is_out_of_scope
        assert not res.has_contradiction
        assert not res.is_blocked
        assert not res.has_clarification

    def test_contradiction_with_correction_properties(self):
        res = QueryTransformerResult(
            intent="eeg_analysis",
            condensed="compute PLI for alpha on F3, F4 during task",
            contradiction="Condition axis — rest vs task",
            clarification="You switched condition from rest to task — is that correct?",
        )
        assert not res.is_out_of_scope
        assert res.has_contradiction
        assert not res.is_blocked
        assert res.has_clarification

    def test_blocked_contradiction_properties(self):
        res = QueryTransformerResult(
            intent="eeg_analysis",
            condensed="[BLOCKED — see CLARIFICATION]",
            contradiction="Condition axis conflict without correction language",
            clarification="Original said rest, follow-up said task. Which condition?",
        )
        assert not res.is_out_of_scope
        assert res.has_contradiction
        assert res.is_blocked
        assert res.has_clarification


class TestOutputParser:
    """Test _parse_transformer_output regex extraction."""

    def test_parse_valid_output(self):
        raw = (
            "INTENT: eeg_analysis\n"
            "CONDENSED: compute PLI for alpha band on F3, F4 during rest\n"
            "CONTRADICTION: none\n"
            "CLARIFICATION: none\n"
        )
        fields = _parse_transformer_output(raw)
        assert fields is not None
        assert fields["INTENT"] == "eeg_analysis"
        assert fields["CONDENSED"] == "compute PLI for alpha band on F3, F4 during rest"
        assert fields["CONTRADICTION"] == "none"
        assert fields["CLARIFICATION"] == "none"

    def test_parse_with_surrounding_text(self):
        raw = (
            "Here is the classification:\n\n"
            "INTENT: out_of_scope\n"
            "CONDENSED: Request is about cooking recipes.\n"
            "CONTRADICTION: none\n"
            "CLARIFICATION: none\n\n"
            "Hope this helps!"
        )
        fields = _parse_transformer_output(raw)
        assert fields is not None
        assert fields["INTENT"] == "out_of_scope"
        assert fields["CONDENSED"] == "Request is about cooking recipes."

    def test_parse_missing_field_returns_none(self):
        raw = (
            "INTENT: eeg_analysis\n"
            "CONDENSED: compute PLI\n"
            "CONTRADICTION: none\n"
        )
        assert _parse_transformer_output(raw) is None

    def test_parse_empty_string_returns_none(self):
        assert _parse_transformer_output("") is None


class TestBuildFallback:
    """Test fallback generator."""

    def test_build_fallback_structure(self):
        fb = _build_fallback("my raw message")
        assert fb.intent == "eeg_analysis"
        assert fb.condensed == "my raw message"
        assert fb.contradiction == "none"
        assert fb.clarification == "none"
        assert not fb.has_contradiction
        assert not fb.is_blocked


class TestTransformQuery:
    """Test end-to-end transform_query with mocked LLMs."""

    def _mock_llm(self, response_text: str):
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        llm.invoke.return_value = mock_response
        return llm

    def test_transform_clean_query(self):
        response_text = (
            "INTENT: eeg_analysis\n"
            "CONDENSED: compute PLI for alpha band on F3, F4 during rest\n"
            "CONTRADICTION: none\n"
            "CLARIFICATION: none\n"
        )
        llm = self._mock_llm(response_text)
        result = transform_query(
            "compute PLI for alpha band on F3, F4 during rest",
            "compute PLI for alpha band on F3, F4 during rest",
            llm=llm,
        )
        assert result.intent == "eeg_analysis"
        assert result.condensed == "compute PLI for alpha band on F3, F4 during rest"
        assert not result.has_contradiction
        assert not result.is_blocked
        assert not result.is_out_of_scope

    def test_transform_out_of_scope(self):
        response_text = (
            "INTENT: out_of_scope\n"
            "CONDENSED: Request is about weather, not EEG functional connectivity analysis.\n"
            "CONTRADICTION: none\n"
            "CLARIFICATION: none\n"
        )
        llm = self._mock_llm(response_text)
        result = transform_query(
            "what's the weather in Mumbai?",
            "what's the weather in Mumbai?",
            llm=llm,
        )
        assert result.is_out_of_scope
        assert "weather" in result.condensed.lower()

    def test_transform_contradiction_with_correction(self):
        response_text = (
            "INTENT: eeg_analysis\n"
            "CONDENSED: compute PLI for alpha band on F3, F4 during task\n"
            "CONTRADICTION: Condition axis — 'rest' in first turn vs 'task' in follow-up.\n"
            "CLARIFICATION: You switched the condition from rest to task — is that correct?\n"
        )
        llm = self._mock_llm(response_text)
        result = transform_query(
            "compute PLI for alpha band on F3, F4 during rest\n[User follow-up refinement]: actually do this for the task condition",
            "actually do this for the task condition",
            llm=llm,
        )
        assert result.intent == "eeg_analysis"
        assert result.has_contradiction
        assert not result.is_blocked
        assert result.has_clarification
        assert "task" in result.condensed

    def test_transform_contradiction_without_correction_blocked(self):
        response_text = (
            "INTENT: eeg_analysis\n"
            "CONDENSED: [BLOCKED — see CLARIFICATION]\n"
            "CONTRADICTION: Condition axis — 'rest' in first turn vs 'task' in follow-up, no correction language present.\n"
            "CLARIFICATION: Your original request specified condition 'rest' but your follow-up says 'task'. Which condition should I use?\n"
        )
        llm = self._mock_llm(response_text)
        result = transform_query(
            "compute PLI for alpha band on F3, F4 during rest\n[User follow-up refinement]: do this for the task condition",
            "do this for the task condition",
            llm=llm,
        )
        assert result.intent == "eeg_analysis"
        assert result.has_contradiction
        assert result.is_blocked
        assert result.has_clarification
        assert "Which condition" in result.clarification

    def test_transform_llm_exception_falls_back(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("API connection timeout")
        result = transform_query(
            "accumulated context",
            "latest user message",
            llm=llm,
        )
        assert result.intent == "eeg_analysis"
        assert result.condensed == "latest user message"
        assert not result.has_contradiction

    def test_transform_malformed_llm_response_falls_back(self):
        """When LLM returns unparseable text, gracefully fall back to latest user message."""
        llm = self._mock_llm("I am unable to classify this request because of XYZ reasons.")
        result = transform_query(
            "accumulated context",
            "latest user message",
            llm=llm,
        )
        assert result.intent == "eeg_analysis"
        assert result.condensed == "latest user message"
        assert not result.has_contradiction

    def test_transform_resolve_contradiction_via_confirmation(self):
        """Example 8: User confirms previously flagged contradiction with bare 'yes'."""
        response_text = (
            "INTENT: eeg_analysis\n"
            "CONDENSED: compute PLI for alpha band on F3, F4 during task\n"
            "CONTRADICTION: none\n"
            "CLARIFICATION: none\n"
        )
        llm = self._mock_llm(response_text)
        accumulated = (
            "compute PLI for alpha band on F3, F4 during rest\n"
            "[User follow-up refinement]: actually do this for the task condition\n"
            "[User clarification reply]: yes"
        )
        result = transform_query(accumulated, "yes", llm=llm)
        assert result.intent == "eeg_analysis"
        assert not result.has_contradiction
        assert not result.is_blocked
        assert not result.has_clarification
        assert "task" in result.condensed


class TestChainlitSessionSimulation:
    """Test full multi-turn block-then-resolve loop in chainlit_app.py."""

    @pytest.mark.skip(
        reason="Superseded: the Query Transformer now runs inside the LangGraph "
        "(query_transformer node -> clarification_pause). Contradiction HITL is covered "
        "by tests/unit/agentic/test_hitl_clarification.py."
    )
    @pytest.mark.asyncio
    async def test_block_then_resolve_round_trip(self):
        import chainlit_app

        # In-memory session store
        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)

        def _set(k, v):
            session_store[k] = v

        mock_user_session.set.side_effect = _set

        # Mock messages sent
        sent_messages = []

        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                self.elements = elements or []
                sent_messages.append(self)

            async def send(self):
                pass

            async def update(self):
                pass

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return _wrapper

        supervisor_invocations = []
        pipeline_call_idx = 0

        def mock_run_pipeline(effective_query, run_id, tags=None, **kwargs):
            nonlocal pipeline_call_idx
            # In the graph architecture, the graph executes the Query Transformer node first
            from fc_pipeline.pipeline.nodes import query_transformer_node_adapter
            qt_res = query_transformer_node_adapter({"user_request": effective_query, "run_id": run_id})
            if qt_res.get("clarification_question"):
                return {
                    "_routed_node": "clarification_pause",
                    "clarification_question": qt_res["clarification_question"],
                }, ["query_transformer", "clarification_pause"]
            pipeline_call_idx += 1
            condensed = qt_res.get("user_request", effective_query)
            supervisor_invocations.append((condensed, run_id))
            if pipeline_call_idx == 1:
                # Turn 1: Intermediate supervisor pass before Gate 1
                return {
                    "_routed_node": "supervisor_preview",
                    "user_request": condensed,
                    "plan": None,
                    "parameter_manifest": [],
                }, ["query_transformer", "supervisor"]
            # Turn 3: Final supervisor pass lands at Gate 1
            return {
                "_routed_node": "gate_1_review",
                "user_request": condensed,
                "plan": MagicMock(metrics=[], channels=["F3", "F4"], condition="rest"),
                "parameter_manifest": [],
            }, ["query_transformer", "supervisor", "gate_1_review"]

        # Mock the underlying LLM passed to get_supervisor_llm()
        mock_llm = MagicMock()
        r1 = MagicMock()
        r1.content = (
            "INTENT: eeg_analysis\n"
            "CONDENSED: compute PLI for alpha band on F3, F4 during rest\n"
            "CONTRADICTION: none\n"
            "CLARIFICATION: none\n"
        )
        r2 = MagicMock()
        r2.content = (
            "INTENT: eeg_analysis\n"
            "CONDENSED: compute PLI for alpha band on F3, F4 during task\n"
            "CONTRADICTION: Condition axis — rest vs task\n"
            "CLARIFICATION: You switched condition from rest to task — is that correct?\n"
        )
        r3 = MagicMock()
        r3.content = (
            "INTENT: eeg_analysis\n"
            "CONDENSED: compute PLI for alpha band on F3, F4 during task\n"
            "CONTRADICTION: none\n"
            "CLARIFICATION: none\n"
        )
        mock_llm.invoke.side_effect = [r1, r2, r3]

        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None):
                self.content = content
                self.actions = actions
            async def send(self):
                return {"name": "approve"}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app, "_run_pipeline_sync", mock_run_pipeline)
            # Patch get_supervisor_llm so transform_query uses mock_llm
            mp.setattr("fc_pipeline.agentic.supervisor.query_transformer.get_supervisor_llm", lambda: mock_llm)

            # --- TURN 1: Initial query ---
            msg1 = MagicMock()
            msg1.content = "compute PLI for alpha band on F3, F4 during rest"
            await chainlit_app.on_message(msg1)

            assert len(supervisor_invocations) == 1
            assert supervisor_invocations[0][0] == "compute PLI for alpha band on F3, F4 during rest"
            assert session_store["awaiting_transformer_clarification"] is False

            # --- TURN 2: Follow-up refinement with contradiction ---
            msg2 = MagicMock()
            msg2.content = "actually do this for the task condition"
            await chainlit_app.on_message(msg2)

            # Supervisor should NOT have been invoked again (blocked)
            assert len(supervisor_invocations) == 1
            assert session_store["awaiting_transformer_clarification"] is True
            assert any("Possible Contradiction Detected" in m.content for m in sent_messages)

            # --- TURN 3: User replies "yes" (resolves contradiction) ---
            msg3 = MagicMock()
            msg3.content = "yes"
            await chainlit_app.on_message(msg3)

            # 1. ASSERT WHAT WAS ACTUALLY SENT TO THE LLM ON TURN 3:
            assert mock_llm.invoke.call_count == 3
            third_turn_call_args = mock_llm.invoke.call_args_list[2]
            sent_messages_to_llm = third_turn_call_args[0][0]  # [SystemMessage, HumanMessage]
            system_msg, human_msg = sent_messages_to_llm[0], sent_messages_to_llm[1]

            # Verify prompt content sent to the LLM has the full accumulated history including clarification reply:
            expected_accumulated = (
                "compute PLI for alpha band on F3, F4 during rest\n"
                "[User follow-up refinement]: actually do this for the task condition\n"
                "[User clarification reply]: yes"
            )
            assert human_msg.content == expected_accumulated
            assert "[User clarification reply]: yes" in human_msg.content

            # 2. Assert Supervisor was invoked with the resolved query
            assert len(supervisor_invocations) == 2
            assert supervisor_invocations[1][0] == "compute PLI for alpha band on F3, F4 during task"
            assert session_store["awaiting_transformer_clarification"] is False

    @pytest.mark.asyncio
    async def test_transformer_clarification_does_not_clear_supervisor_clarification(self):
        """Confirm awaiting_transformer_clarification does not clear awaiting_clarification."""
        import chainlit_app

        session_store = {
            "original_query": "compute PLI",
            "accumulated_query": "compute PLI",
            "awaiting_clarification": True,  # Supervisor previously requested clarification
            "awaiting_transformer_clarification": True,
            "counter": 1,
            "data_path": "test_path.fif",
            "is_real_data": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
            async def send(self): pass
            async def update(self): pass

        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self): return {"name": "approve"}

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return _wrapper

        mock_transform = MagicMock(return_value=QueryTransformerResult(
            intent="eeg_analysis",
            condensed="compute PLI for alpha on F3, F4",
            contradiction="none",
            clarification="none",
        ))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", mock_transform)
            mp.setattr(chainlit_app, "_run_pipeline_sync", MagicMock(return_value=({"_routed_node": "gate_1_review"}, [])))

            msg = MagicMock()
            msg.content = "yes"
            await chainlit_app.on_message(msg)

            # awaiting_transformer_clarification was cleared
            assert session_store["awaiting_transformer_clarification"] is False
            # awaiting_clarification was NOT prematurely wiped out by transformer handling
            assert session_store["awaiting_clarification"] is False  # cleared by gate_1_review

    @pytest.mark.asyncio
    async def test_gate_1_approve_flow(self):
        """Approve action sets gate_1_approved to True."""
        import chainlit_app

        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        sent_messages = []
        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                sent_messages.append(self)
            async def send(self): pass
            async def update(self): pass

        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self): return {"name": "approve"}

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", MagicMock(return_value=QueryTransformerResult("eeg_analysis", "test", "none", "none")))
            mp.setattr(chainlit_app, "_run_pipeline_sync", MagicMock(return_value=({"_routed_node": "gate_1_review", "parameter_manifest": []}, [])))

            msg = MagicMock()
            msg.content = "compute PLI"
            await chainlit_app.on_message(msg)

            assert session_store["gate_1_approved"] is True
            assert any("Gate 1 Approved" in m.content for m in sent_messages)

    @pytest.mark.asyncio
    async def test_gate_1_reject_flow(self):
        """Reject action halts execution and leaves gate_1_approved False."""
        import chainlit_app

        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        sent_messages = []
        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                sent_messages.append(self)
            async def send(self): pass
            async def update(self): pass

        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self): return {"name": "reject"}

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", MagicMock(return_value=QueryTransformerResult("eeg_analysis", "test", "none", "none")))
            mp.setattr(chainlit_app, "_run_pipeline_sync", MagicMock(return_value=({"_routed_node": "gate_1_review", "parameter_manifest": []}, [])))

            msg = MagicMock()
            msg.content = "compute PLI"
            await chainlit_app.on_message(msg)

            assert session_store["gate_1_approved"] is False
            assert any("Gate 1 Rejected" in m.content for m in sent_messages)

    @pytest.mark.asyncio
    async def test_gate_1_timeout_aborts(self):
        """Timeout (res=None) strictly aborts and never approves."""
        import chainlit_app

        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        sent_messages = []
        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                sent_messages.append(self)
            async def send(self): pass
            async def update(self): pass

        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self): return None  # Timeout!

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", MagicMock(return_value=QueryTransformerResult("eeg_analysis", "test", "none", "none")))
            mp.setattr(chainlit_app, "_run_pipeline_sync", MagicMock(return_value=({"_routed_node": "gate_1_review", "parameter_manifest": []}, [])))

            msg = MagicMock()
            msg.content = "compute PLI"
            await chainlit_app.on_message(msg)

            assert session_store["gate_1_approved"] is False
            assert any("timed out after 5 minutes with no response" in m.content for m in sent_messages)

    @pytest.mark.asyncio
    async def test_gate_1_request_changes_skips_transformer(self):
        """Request Changes prompts user, appends change, skips query transformer, and re-invokes Supervisor with rev suffix."""
        import chainlit_app

        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
            async def send(self): pass
            async def update(self): pass

        # First prompt asks action: return request_changes.
        # Second prompt (on re-entry): return approve so loop finishes.
        action_call_count = 0
        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self):
                nonlocal action_call_count
                action_call_count += 1
                if action_call_count == 1:
                    return {"name": "request_changes"}
                return {"name": "approve"}

        class MockAskUserMessage:
            def __init__(self, content="", timeout=None): pass
            async def send(self):
                return {"output": "switch condition to task"}

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        supervisor_invocations = []
        def mock_run_pipeline(effective_query, run_id, tags=None, **kwargs):
            # Graph enters query_transformer node on initial request (revision == 0)
            if not tags or tags.get("revision", 0) == 0:
                mock_transform(effective_query, effective_query)
            supervisor_invocations.append((effective_query, run_id, tags))
            return {"_routed_node": "gate_1_review", "parameter_manifest": []}, ["supervisor"]

        mock_transform = MagicMock(return_value=QueryTransformerResult("eeg_analysis", "compute PLI on alpha", "none", "none"))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "AskUserMessage", MockAskUserMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", mock_transform)
            mp.setattr(chainlit_app, "_run_pipeline_sync", mock_run_pipeline)

            msg = MagicMock()
            msg.content = "compute PLI on alpha"
            await chainlit_app.on_message(msg)

            # transform_query must only be called ONCE (for the initial turn) — NOT on the Request Changes path!
            assert mock_transform.call_count == 1

            # Supervisor was invoked twice:
            # 1. Initial run
            # 2. Re-invocation with revision suffix
            assert len(supervisor_invocations) == 2
            initial_req, initial_id, initial_tags = supervisor_invocations[0]
            rev_req, rev_id, rev_tags = supervisor_invocations[1]

            assert "_rev1" in rev_id
            assert rev_tags["revision"] == 1
            assert rev_tags["parent_scenario_id"] == initial_id
            assert "[User Gate 1 Change Request]: switch condition to task" in rev_req
            assert session_store["gate_1_approved"] is True  # approved on second pass
            # Approved resets conversation state for the next user query
            assert session_store["original_query"] is None
            assert session_store["accumulated_query"] is None

    @pytest.mark.asyncio
    async def test_gate_1_request_changes_routes_to_clarification_pause(self):
        """Confirm that if Supervisor re-invocation routes to clarification_pause, it pauses with clarification prompt."""
        import chainlit_app

        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        sent_messages = []
        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                sent_messages.append(self)
            async def send(self): pass
            async def update(self): pass

        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self):
                return {"name": "request_changes"}

        class MockAskUserMessage:
            def __init__(self, content="", timeout=None): pass
            async def send(self):
                return {"output": "change frequency band"}  # underspecified feedback

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        call_idx = 0
        def mock_run_pipeline(effective_query, run_id, tags=None, **kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                # Initial run reaches Gate 1
                return {"_routed_node": "gate_1_review", "parameter_manifest": []}, ["supervisor"]
            # Re-run after Request Changes routes to clarification_pause!
            return {
                "_routed_node": "clarification_pause",
                "clarification_question": "Which frequency band would you like to analyze?",
            }, ["supervisor"]

        mock_transform = MagicMock(return_value=QueryTransformerResult("eeg_analysis", "compute PLI", "none", "none"))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "AskUserMessage", MockAskUserMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", mock_transform)
            mp.setattr(chainlit_app, "_run_pipeline_sync", mock_run_pipeline)

            msg = MagicMock()
            msg.content = "compute PLI"
            await chainlit_app.on_message(msg)

            # Assert awaiting_clarification is True (paused for user clarification)
            assert session_store["awaiting_clarification"] is True
            assert session_store["gate_1_approved"] is False
            # Assert clarification question was rendered to the user
            assert any("Clarification Required" in m.content for m in sent_messages)
            assert any("Which frequency band would you like to analyze?" in m.content for m in sent_messages)


class TestGate1PerRowEdits:
    """Test Task 2 Phase B — per-row Edit actions on Gate 1 manifest entries."""

    @pytest.mark.asyncio
    async def test_edit_advisory_entry_direct_accept(self):
        """Edit an advisory row → direct accept writes human_approved_value, no Supervisor re-run."""
        import chainlit_app
        from fc_pipeline.schemas.manifest import ParameterManifestEntry

        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        sent_messages = []
        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                sent_messages.append(self)
            async def send(self): pass
            async def update(self): pass

        manifest_initial = [
            ParameterManifestEntry(
                name="trial_adequacy",
                category="advisory",
                proposed_value="medium (20 trials, borderline)",
                confidence=0.62,
                needs_human_input=True,
                risk_tier="elevated",
            ),
            ParameterManifestEntry(
                name="freq_band",
                category="scientific_axis",
                proposed_value="alpha (8-13 Hz)",
                confidence=1.0,
                needs_human_input=False,
                risk_tier="low",
            ),
        ]
        manifest_captured = {"ref": None}

        action_call_count = 0
        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None):
                self.content = content
                self.actions = actions
            async def send(self):
                nonlocal action_call_count
                action_call_count += 1
                if action_call_count == 1:
                    return {"name": "edit_trial_adequacy", "payload": {"entry_name": "trial_adequacy"}}
                return {"name": "approve"}

        ask_user_call_count = 0
        class MockAskUserMessage:
            def __init__(self, content="", timeout=None):
                self.content = content
            async def send(self):
                nonlocal ask_user_call_count
                ask_user_call_count += 1
                return {"output": "high (accept borderline)"}

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        supervisor_call_count = 0
        def mock_run_pipeline(effective_query, run_id, tags=None, **kwargs):
            nonlocal supervisor_call_count
            supervisor_call_count += 1
            manifest_captured["ref"] = manifest_initial
            return {"_routed_node": "gate_1_review", "parameter_manifest": manifest_initial}, ["supervisor"]

        mock_transform = MagicMock(return_value=QueryTransformerResult("eeg_analysis", "compute PLI", "none", "none"))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "AskUserMessage", MockAskUserMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", mock_transform)
            mp.setattr(chainlit_app, "_run_pipeline_sync", mock_run_pipeline)

            msg = MagicMock()
            msg.content = "compute PLI"
            await chainlit_app.on_message(msg)

        # Supervisor only called ONCE (initial pass — no re-run for advisory edit)
        assert supervisor_call_count == 1
        # AskUserMessage called once (for the edit prompt)
        assert ask_user_call_count == 1
        # Gate 1 approved on second action prompt
        assert session_store["gate_1_approved"] is True
        # Directly assert the advisory entry has the human_approved_value written
        trial_entry = next(e for e in manifest_captured["ref"] if e.name == "trial_adequacy")
        assert trial_entry.human_approved_value == "high (accept borderline)"
        # Scientific axis entry left untouched
        freq_entry = next(e for e in manifest_captured["ref"] if e.name == "freq_band")
        assert freq_entry.human_approved_value is None
        # Confirm approval success message
        assert any("Gate 1 Approved" in m.content for m in sent_messages)

    @pytest.mark.asyncio
    async def test_edit_scientific_axis_routes_to_supervisor_revalidation(self):
        """Edit a scientific_axis row → REJECT direct edit, route through Request Changes (Supervisor re-run with rev suffix)."""
        import chainlit_app
        from fc_pipeline.schemas.manifest import ParameterManifestEntry

        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
            "base_run_id": None,
            "gate_1_revision": 0,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        sent_messages = []
        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                sent_messages.append(self)
            async def send(self): pass
            async def update(self): pass

        manifest_v1 = [
            ParameterManifestEntry(
                name="freq_band",
                category="scientific_axis",
                proposed_value="alpha (8-13 Hz)",
                confidence=0.70,
                needs_human_input=True,
                risk_tier="elevated",
            ),
        ]
        manifest_v2 = [
            ParameterManifestEntry(
                name="freq_band",
                category="scientific_axis",
                proposed_value="beta (13-30 Hz)",
                confidence=1.0,
                needs_human_input=False,
                risk_tier="low",
            ),
        ]

        action_call_count = 0
        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self):
                nonlocal action_call_count
                action_call_count += 1
                if action_call_count == 1:
                    return {"name": "edit_freq_band", "payload": {"entry_name": "freq_band"}}
                return {"name": "approve"}

        ask_user_call_count = 0
        class MockAskUserMessage:
            def __init__(self, content="", timeout=None):
                self.content = content
            async def send(self):
                nonlocal ask_user_call_count
                ask_user_call_count += 1
                return {"output": "beta (13-30 Hz)"}

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        supervisor_invocations = []
        def mock_run_pipeline(effective_query, run_id, tags=None, **kwargs):
            # Graph enters query_transformer node on initial request (revision == 0)
            if not tags or tags.get("revision", 0) == 0:
                mock_transform(effective_query, effective_query)
            supervisor_invocations.append((effective_query, run_id, tags))
            if len(supervisor_invocations) == 1:
                return {"_routed_node": "gate_1_review", "parameter_manifest": manifest_v1}, ["supervisor"]
            return {"_routed_node": "gate_1_review", "parameter_manifest": manifest_v2}, ["supervisor"]

        mock_transform = MagicMock(return_value=QueryTransformerResult("eeg_analysis", "compute PLI", "none", "none"))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "AskUserMessage", MockAskUserMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", mock_transform)
            mp.setattr(chainlit_app, "_run_pipeline_sync", mock_run_pipeline)

            msg = MagicMock()
            msg.content = "compute PLI"
            await chainlit_app.on_message(msg)

        # Transform query only called ONCE (initial — skipped on re-validation path)
        assert mock_transform.call_count == 1
        # Supervisor called TWICE (initial + re-validation)
        assert len(supervisor_invocations) == 2
        initial_req, initial_id, initial_tags = supervisor_invocations[0]
        rev_req, rev_id, rev_tags = supervisor_invocations[1]

        # Rev suffixed run id + lineage tags
        assert "_rev1" in rev_id
        assert rev_tags["revision"] == 1
        assert rev_tags["parent_scenario_id"] == initial_id

        # Change-request text tagged correctly
        assert "[User Gate 1 Change Request]" in rev_req
        assert "freq_band" in rev_req
        assert "beta" in rev_req

        # Gate 1 approved on second pass
        assert session_store["gate_1_approved"] is True

    @pytest.mark.asyncio
    async def test_multiple_sequential_edits_persist(self):
        """Two sequential advisory edits in one Gate 1 turn → both human_approved_value persist in final manifest."""
        import chainlit_app
        from fc_pipeline.schemas.manifest import ParameterManifestEntry

        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        sent_messages = []
        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                sent_messages.append(self)
            async def send(self): pass
            async def update(self): pass

        manifest_initial = [
            ParameterManifestEntry(
                name="trial_adequacy",
                category="advisory",
                proposed_value="medium",
                confidence=0.6,
                needs_human_input=True,
                risk_tier="elevated",
            ),
            ParameterManifestEntry(
                name="bad_channel_threshold",
                category="engineering_threshold",
                proposed_value="0.4",
                confidence=0.75,
                needs_human_input=True,
                risk_tier="low",
            ),
        ]
        manifest_captured = {"ref": None}

        action_call_count = 0
        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self):
                nonlocal action_call_count
                action_call_count += 1
                if action_call_count == 1:
                    return {"name": "edit_trial_adequacy", "payload": {"entry_name": "trial_adequacy"}}
                elif action_call_count == 2:
                    return {"name": "edit_bad_channel_threshold", "payload": {"entry_name": "bad_channel_threshold"}}
                return {"name": "approve"}

        ask_user_responses = ["high (>= 20 trials OK)", "0.6 stricter threshold"]
        ask_user_call_count = 0
        class MockAskUserMessage:
            def __init__(self, content="", timeout=None): pass
            async def send(self):
                nonlocal ask_user_call_count
                ask_user_call_count += 1
                return {"output": ask_user_responses[ask_user_call_count - 1]}

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        def mock_run_pipeline(effective_query, run_id, tags=None, **kwargs):
            manifest_captured["ref"] = manifest_initial
            return {"_routed_node": "gate_1_review", "parameter_manifest": manifest_initial}, ["supervisor"]

        mock_transform = MagicMock(return_value=QueryTransformerResult("eeg_analysis", "compute", "none", "none"))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "AskUserMessage", MockAskUserMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", mock_transform)
            mp.setattr(chainlit_app, "_run_pipeline_sync", mock_run_pipeline)

            msg = MagicMock()
            msg.content = "compute"
            await chainlit_app.on_message(msg)

        # Gate 1 approved, 3 action prompts (edit1, edit2, approve)
        assert action_call_count == 3
        assert ask_user_call_count == 2
        assert session_store["gate_1_approved"] is True

        # BOTH edits persisted on the same manifest reference
        entries = {e.name: e for e in manifest_captured["ref"]}
        assert entries["trial_adequacy"].human_approved_value == "high (>= 20 trials OK)"
        assert entries["bad_channel_threshold"].human_approved_value == "0.6 stricter threshold"

    @pytest.mark.asyncio
    async def test_edit_prompt_timeout_aborts_cleanly(self):
        """During per-row edit, AskUserMessage times out (returns None) → pipeline aborts cleanly, no approve, state reset."""
        import chainlit_app
        from fc_pipeline.schemas.manifest import ParameterManifestEntry

        session_store = {
            "original_query": "compute PLI",
            "accumulated_query": "compute PLI",
            "base_run_id": "run_abc",
            "gate_1_revision": 2,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 1,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        sent_messages = []
        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                sent_messages.append(self)
            async def send(self): pass
            async def update(self): pass

        manifest_initial = [
            ParameterManifestEntry(
                name="trial_adequacy",
                category="informational",
                proposed_value="low",
                needs_human_input=True,
                risk_tier="low",
            ),
        ]

        action_call_count = 0
        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self):
                nonlocal action_call_count
                action_call_count += 1
                return {"name": "edit_trial_adequacy", "payload": {"entry_name": "trial_adequacy"}}

        class MockAskUserMessage:
            def __init__(self, content="", timeout=None): pass
            async def send(self):
                return None  # Timeout!

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        def mock_run_pipeline(effective_query, run_id, tags=None, **kwargs):
            return {"_routed_node": "gate_1_review", "parameter_manifest": manifest_initial}, ["supervisor"]

        mock_transform = MagicMock(return_value=QueryTransformerResult("eeg_analysis", "compute", "none", "none"))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "AskUserMessage", MockAskUserMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", mock_transform)
            mp.setattr(chainlit_app, "_run_pipeline_sync", mock_run_pipeline)

            msg = MagicMock()
            msg.content = "compute"
            await chainlit_app.on_message(msg)

        # Not approved, conversation state was reset (per _reset_conversation_state contract)
        assert session_store["gate_1_approved"] is False
        assert session_store["original_query"] is None
        assert session_store["accumulated_query"] is None
        assert session_store["base_run_id"] is None
        assert session_store["gate_1_revision"] == 0

        # Timeout message shown
        assert any("Edit timed out after 5 minutes with no response" in m.content for m in sent_messages)
        assert any("Pipeline execution aborted" in m.content for m in sent_messages)

    def test_format_manifest_markdown_seventh_column_masked(self):
        """format_manifest_markdown renders 7th 'Approved Value' column; runs human_approved_value through mask_text()."""
        import chainlit_app
        from fc_pipeline.schemas.manifest import ParameterManifestEntry

        manifest = [
            ParameterManifestEntry(
                name="data_source_path",
                category="informational",
                proposed_value="C:\\Users\\john\\datasets\\eeg.fif",
                confidence=1.0,
                needs_human_input=True,
                risk_tier="low",
                human_approved_value="C:\\Users\\john\\datasets\\override\\eeg.fif",
            ),
            ParameterManifestEntry(
                name="freq_band",
                category="scientific_axis",
                proposed_value="alpha",
                confidence=1.0,
                needs_human_input=False,
                risk_tier="low",
                human_approved_value=None,
            ),
        ]

        md = chainlit_app.format_manifest_markdown(manifest)
        lines = md.strip().split("\n")

        # Header: 7 columns total
        header = lines[0]
        assert header.count("|") == 8  # 7 fields = 8 pipes (|A|B|C|D|E|F|G|)
        assert "Approved Value" in header

        # Row 1: Extract ONLY the Approved Value cell (index 7, 0-based after leading empty from leading |)
        # cells: ['', 'name', 'category', 'proposed', 'conf', 'human_needed', 'risk', 'APPROVED_VALUE', '']
        data_cells = [c.strip() for c in lines[2].split("|")]
        approved_cell_data = data_cells[7]

        # Approved Value column has path/user masked
        assert "[LOCAL_ROOT]/" in approved_cell_data
        assert "john" not in approved_cell_data
        # Approved Value column must NOT contain the raw Windows path
        assert "C:\\Users\\john" not in approved_cell_data
        # And must contain the actual override filename (eeg.fif) after the masked root
        assert "eeg.fif" in approved_cell_data

        # proposed_value column (index 3) is ALSO run through mask_text per user ruling
        # (consistent with Approved Value and edit prompt display)
        proposed_cell_data = data_cells[3]
        assert "[LOCAL_ROOT]/" in proposed_cell_data
        assert "john" not in proposed_cell_data
        assert "C:\\Users\\john" not in proposed_cell_data
        assert "eeg.fif" in proposed_cell_data

        # Row 2: human_approved_value is None → renders "—"
        freq_cells = [c.strip() for c in lines[3].split("|")]
        approved_cell_freq = freq_cells[7]
        assert approved_cell_freq == "—"

    @pytest.mark.asyncio
    async def test_option_a_override_persists_across_scientific_rerun(self):
        """Data-loss Option (a): advisory override snapshotted before scientific-axis Supervisor re-run, then re-applied to new manifest matching by entry name (only if still flagged)."""
        import chainlit_app
        from fc_pipeline.schemas.manifest import ParameterManifestEntry

        session_store = {
            "original_query": None,
            "accumulated_query": None,
            "awaiting_clarification": False,
            "awaiting_transformer_clarification": False,
            "counter": 0,
            "data_path": "test_path.fif",
            "is_real_data": False,
            "gate_1_approved": False,
            "base_run_id": None,
            "gate_1_revision": 0,
        }

        mock_user_session = MagicMock()
        mock_user_session.get.side_effect = lambda k, default=None: session_store.get(k, default)
        mock_user_session.set.side_effect = lambda k, v: session_store.update({k: v})

        sent_messages = []
        class MockClMessage:
            def __init__(self, content="", elements=None):
                self.content = content
                sent_messages.append(self)
            async def send(self): pass
            async def update(self): pass

        # V1 manifest: advisory row has user-set override already; scientific axis row also flagged
        manifest_v1 = [
            ParameterManifestEntry(
                name="trial_adequacy",
                category="advisory",
                proposed_value="medium",
                confidence=0.60,
                needs_human_input=True,
                risk_tier="elevated",
                human_approved_value="high (accept borderline)",  # Already edited in this turn
            ),
            ParameterManifestEntry(
                name="freq_band",
                category="scientific_axis",
                proposed_value="alpha (8-13 Hz)",
                confidence=0.75,
                needs_human_input=True,
                risk_tier="elevated",
            ),
        ]

        # V2 manifest (produced by Supervisor re-run): same advisory row STILL FLAGGED, scientific resolved
        manifest_v2_captured = {"ref": None}
        def build_manifest_v2():
            return [
                ParameterManifestEntry(
                    name="trial_adequacy",
                    category="advisory",
                    proposed_value="medium (20 trials)",
                    confidence=0.61,
                    needs_human_input=True,  # Still flagged → override should be re-applied
                    risk_tier="elevated",
                ),
                ParameterManifestEntry(
                    name="freq_band",
                    category="scientific_axis",
                    proposed_value="beta (13-30 Hz)",  # Now resolved after re-validation
                    confidence=1.0,
                    needs_human_input=False,
                    risk_tier="low",
                ),
            ]

        action_call_count = 0
        class MockAskActionMessage:
            def __init__(self, content="", actions=None, timeout=None): pass
            async def send(self):
                nonlocal action_call_count
                action_call_count += 1
                if action_call_count == 1:
                    # On the first (v1) gate: user clicks edit on scientific_axis row
                    return {"name": "edit_freq_band", "payload": {"entry_name": "freq_band"}}
                # After re-run lands at gate again (v2): approve
                return {"name": "approve"}

        ask_user_call_count = 0
        class MockAskUserMessage:
            def __init__(self, content="", timeout=None): pass
            async def send(self):
                nonlocal ask_user_call_count
                ask_user_call_count += 1
                return {"output": "beta band (13-30 Hz)"}

        def mock_make_async(fn):
            async def _wrapper(*args, **kwargs): return fn(*args, **kwargs)
            return _wrapper

        supervisor_invocations = []
        def mock_run_pipeline(effective_query, run_id, tags=None, **kwargs):
            supervisor_invocations.append((effective_query, run_id, tags))
            if len(supervisor_invocations) == 1:
                return {"_routed_node": "gate_1_review", "parameter_manifest": manifest_v1}, ["supervisor"]
            # V2: build a FRESH manifest to truly simulate a Supervisor regeneration
            v2 = build_manifest_v2()
            manifest_v2_captured["ref"] = v2
            return {"_routed_node": "gate_1_review", "parameter_manifest": v2}, ["supervisor"]

        mock_transform = MagicMock(return_value=QueryTransformerResult("eeg_analysis", "compute", "none", "none"))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chainlit_app.cl, "user_session", mock_user_session)
            mp.setattr(chainlit_app.cl, "Message", MockClMessage)
            mp.setattr(chainlit_app.cl, "AskActionMessage", MockAskActionMessage)
            mp.setattr(chainlit_app.cl, "AskUserMessage", MockAskUserMessage)
            mp.setattr(chainlit_app.cl, "make_async", mock_make_async)
            mp.setattr(chainlit_app, "transform_query", mock_transform)
            mp.setattr(chainlit_app, "_run_pipeline_sync", mock_run_pipeline)

            msg = MagicMock()
            msg.content = "compute"
            await chainlit_app.on_message(msg)

        # Two supervisor runs: initial + scientific-axis re-validation
        assert len(supervisor_invocations) == 2
        # Gate 1 approved (second pass)
        assert session_store["gate_1_approved"] is True

        # KEY ASSERTION: The V2 advisory entry trial_adequacy — which was FRESHLY
        # BUILT with human_approved_value=None — has had the prior override
        # re-applied by _reapply_overrides because it was still flagged needs_human_input=True.
        v2_entries = {e.name: e for e in manifest_v2_captured["ref"]}
        assert v2_entries["trial_adequacy"].human_approved_value == "high (accept borderline)"

        # Scientific axis entry correctly NOT re-applied (no override existed for it anyway)
        assert v2_entries["freq_band"].human_approved_value is None
        # And the scientific axis resolved properly
        assert v2_entries["freq_band"].needs_human_input is False
        assert "beta" in v2_entries["freq_band"].proposed_value.lower()

        # Also verify _snapshot_overrides and _reapply_overrides utility functions directly
        manifest_a = [
            ParameterManifestEntry(name="a", category="advisory", proposed_value="1", needs_human_input=True, human_approved_value="X"),
            ParameterManifestEntry(name="b", category="advisory", proposed_value="2", needs_human_input=True, human_approved_value="Y"),
            ParameterManifestEntry(name="c", category="scientific_axis", proposed_value="3", needs_human_input=False),
        ]
        snap = chainlit_app._snapshot_overrides(manifest_a)
        assert snap == {"a": "X", "b": "Y"}

        # New manifest: 'a' still flagged → override reapplied; 'b' no longer flagged → NOT reapplied
        manifest_b = [
            ParameterManifestEntry(name="a", category="advisory", proposed_value="1", needs_human_input=True),
            ParameterManifestEntry(name="b", category="advisory", proposed_value="2", needs_human_input=False),
            ParameterManifestEntry(name="c", category="scientific_axis", proposed_value="3", needs_human_input=False),
            ParameterManifestEntry(name="d", category="informational", proposed_value="4", needs_human_input=True),  # New, no snapshot
        ]
        chainlit_app._reapply_overrides(manifest_b, snap)
        b_entries = {e.name: e for e in manifest_b}
        assert b_entries["a"].human_approved_value == "X"   # Reapplied (still flagged)
        assert b_entries["b"].human_approved_value is None  # NOT reapplied (no longer flagged)
        assert b_entries["c"].human_approved_value is None  # Never had override
        assert b_entries["d"].human_approved_value is None  # Brand new entry


