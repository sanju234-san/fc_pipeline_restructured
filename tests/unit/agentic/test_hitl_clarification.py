"""Regression tests for the Supervisor halt -> clarification_pause -> Chainlit HITL -> resume lifecycle.

Root-cause regressions covered here:
  * an unresolved clarification must ALWAYS reach the user with clickable controls
    (bounded options when the dataset allows, ``Type manually`` unconditionally);
  * one authoritative ``clarification`` payload is threaded Supervisor -> GraphState ->
    interrupt() -> Chainlit;
  * every answer is validated, normalised and persisted deterministically, and the SAME
    checkpoint resumes with the Supervisor skipping the resolved axis;
  * chained clarifications (band -> channels -> condition) do not crash on resume;
  * Gate 1 approval synchronises the approved manifest into the checkpoint;
  * post-run follow-ups never restart Supervisor / HITL / Gate 1 / Data Prep.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.types import Command

from fc_pipeline.agentic.supervisor.agent import supervisor_node
from fc_pipeline.agentic.supervisor.query_transformer import QueryTransformerResult
from fc_pipeline.agentic.supervisor.hitl_resolution import apply_clarification_reply
from fc_pipeline.pipeline.graph import compile_pipeline_app
from fc_pipeline.pipeline import nodes as pipeline_nodes
from fc_pipeline.schemas.clarification import build_clarification, get_clarification

import chainlit_app as app


# --------------------------------------------------------------------------- helpers
class MockChatModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _call(name, args):
    return f"[TOOL_CALLS]{name}[ARGS]{json.dumps(args)}"


EEGMMIDB_LIKE = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.", "C5.", "C3.", "C1.",
    "Cz.", "C2.", "C4.", "C6.", "Cp3.", "Cpz.", "Cp4.", "Fp1.", "Fp2.", "F3.",
    "F4.", "Pz.", "O1.", "O2.",
]


@pytest.fixture(scope="module")
def mmidb_path(tmp_path_factory):
    """24-channel EDF-style labels ("C3.") with EEGMMIDB event labels T0/T1/T2."""
    sfreq = 160.0
    info = mne.create_info(EEGMMIDB_LIKE, sfreq, "eeg")
    raw = mne.io.RawArray(np.random.RandomState(0).randn(len(EEGMMIDB_LIKE), int(sfreq * 60)) * 1e-6, info, verbose=False)
    raw.set_annotations(mne.Annotations([0, 10, 20], [5, 5, 5], ["T0", "T1", "T2"]))
    path = tmp_path_factory.mktemp("mmidb") / "mmidb_raw.fif"
    raw.save(str(path), overwrite=True, verbose=False)
    return str(path)


def _initial_state(path, run_id, request="compute PLI"):
    return {
        "pipeline_error": None, "run_id": run_id, "raw_data_path": path,
        "user_request": request, "latest_user_message": request,
        "plan": None, "parameter_manifest": None,
        "preflight_confirmed": False, "gate_1_approved": False,
        "clarification": None, "clarification_question": None, "clarification_kind": None,
        "clarification_options": [], "condition_candidates": [],
        "clarification_response": None, "clarification_resume_kind": None,
        "resolved_frequency_band_info": None, "resolved_channel_info": None,
        "resolved_condition_value": None,
        "dataset_sfreq": None, "dataset_duration_seconds": None,
        "dataset_available_channels": None, "dataset_reference": None,
        "informational_response": None, "informational_artifacts": None,
        "decision_context": None, "input_rail_cleared": False,
    }


def _channels_unresolved_llm(path):
    """band resolved (theta), channels tool called but unresolved, condition unresolved."""
    return MockChatModel(responses=[
        _call("get_dataset_info", {"data_path": path}),
        _call("get_dataset_conditions", {"data_path": path}),
        _call("resolve_frequency_band", {"band_name_or_range": "theta", "sfreq": 160.0}),
        _call("resolve_channel_selection", {"requested_channels_or_region": "", "available_channels": []}),
        "Need channels.",
    ])


def _band_unresolved_llm(path):
    """channels resolved, band tool called with junk, condition unresolved."""
    return MockChatModel(responses=[
        _call("get_dataset_info", {"data_path": path}),
        _call("get_dataset_conditions", {"data_path": path}),
        _call("resolve_frequency_band", {"band_name_or_range": "gibberish", "sfreq": 160.0}),
        _call("resolve_channel_selection", {
            "requested_channels_or_region": "C3, C4", "available_channels": EEGMMIDB_LIKE}),
        "Need band.",
    ])


class _Graph:
    """Runs the REAL compiled graph with the LLM / Query Transformer / rails patched."""

    def __init__(self, path, llm, run_id, request="compute PLI"):
        self.path, self.run_id = path, run_id
        self.app = compile_pipeline_app()
        self.cfg = {"configurable": {"thread_id": run_id}}
        self.llm = llm
        self.qt = QueryTransformerResult("eeg_analysis", request, "none", "none")
        self.request = request

    def _patches(self):
        return (
            patch.object(pipeline_nodes, "apply_input_rail", return_value=None),
            patch.object(pipeline_nodes, "transform_query", return_value=self.qt),
            patch.object(pipeline_nodes, "get_supervisor_llm", return_value=self.llm),
        )

    def _drive(self, stream_input):
        p1, p2, p3 = self._patches()
        nodes = []
        with p1, p2, p3:
            for mode, payload in self.app.stream(stream_input, config=self.cfg, stream_mode=["updates"]):
                nodes.extend(payload.keys())
        return nodes

    def start(self):
        return self._drive(_initial_state(self.path, self.run_id, self.request))

    def reply(self, text):
        return self._drive(Command(resume={"reply": text, "value": text, "action": "clarification"}))

    @property
    def values(self):
        return self.app.get_state(self.cfg).values

    @property
    def next(self):
        return self.app.get_state(self.cfg).next

    @property
    def interrupt_payload(self):
        for t in self.app.get_state(self.cfg).tasks:
            for i in t.interrupts:
                return i.value
        return None


# --------------------------------------------------------------------------- 1. the exact bug
class TestExactBugRegression:
    def test_supervisor_channel_halt_carries_one_authoritative_payload(self, mmidb_path):
        """band_ok=True, channels_ok=False, condition_ok=False -> non-empty channel options."""
        state = _initial_state(mmidb_path, "hitl_payload")
        result = supervisor_node(state, llm=_channels_unresolved_llm(mmidb_path), run_id="hitl_payload")

        assert result["clarification_question"] == "Please choose at least two EEG channels or type a valid brain region."
        clar = result["clarification"]
        assert clar["axis"] == "channels" and clar["kind"] == "channel_selection"
        assert clar["allow_manual"] is True
        assert clar["question"] == result["clarification_question"]
        assert clar["options"], "bounded channel options must be produced from the loaded dataset"
        assert clar["options"] == result["clarification_options"]  # legacy fields are derived, not duplicated
        # every option resolves to >=2 REAL dataset channels (raw labels kept for MNE)
        for opt in clar["options"]:
            update, err = apply_clarification_reply({**state, "dataset_available_channels": EEGMMIDB_LIKE},
                                                    "channel_selection", opt["value"])
            assert err is None
            assert len(update["resolved_channel_info"]["resolved_channels"]) >= 2
            assert set(update["resolved_channel_info"]["resolved_channels"]) <= set(EEGMMIDB_LIKE)
        assert result["resolved_frequency_band_info"]["name"] == "theta"  # band_ok survives the halt

    def test_options_come_from_dataset_even_if_llm_skipped_get_dataset_info(self, mmidb_path):
        """Cause A: options were empty when the LLM never called get_dataset_info."""
        llm = MockChatModel(responses=[
            _call("resolve_frequency_band", {"band_name_or_range": "theta", "sfreq": 160.0}),
            _call("resolve_channel_selection", {"requested_channels_or_region": "", "available_channels": []}),
            "Need channels.",
        ])
        result = supervisor_node(_initial_state(mmidb_path, "hitl_skip_info"), llm=llm, run_id="hitl_skip_info")
        assert result["clarification"]["kind"] == "channel_selection"
        assert result["clarification"]["options"]
        assert result["dataset_available_channels"] == EEGMMIDB_LIKE

    def test_halt_reaches_interrupt_payload_with_options_and_manual(self, mmidb_path):
        g = _Graph(mmidb_path, _channels_unresolved_llm(mmidb_path), "hitl_interrupt")
        nodes = g.start()
        assert nodes[-1] == "__interrupt__" and g.next == ("clarification_pause",)
        payload = g.interrupt_payload
        assert payload["type"] == "clarification"
        assert payload["clarification"]["options"] and payload["allow_manual"] is True
        assert payload["options"] == payload["clarification"]["options"]


# --------------------------------------------------------------------------- 2. MOST IMPORTANT
class TestChainlitAlwaysRendersHitlControls:
    QUESTION = "Please choose at least two EEG channels or type a valid brain region."

    @pytest.mark.parametrize("state", [
        # authoritative payload with options
        {"clarification": build_clarification("channel_selection", QUESTION,
                                              [{"name": "c", "value": "C3., C4.", "label": "C3 + C4"}])},
        # legacy flat fields only, options list EMPTY, no dataset channels (worst case)
        {"clarification_question": QUESTION, "clarification_kind": "channel_selection",
         "clarification_options": [], "band_ok": True},
        # legacy fields, options None
        {"clarification_question": QUESTION, "clarification_kind": "channel_selection",
         "clarification_options": None},
        # no kind, no options at all
        {"clarification_question": QUESTION},
        # authoritative payload with junk options
        {"clarification": {"question": QUESTION, "kind": "channel_selection", "options": [{}, None, {"value": ""}]}},
        # nothing at all
        {},
    ])
    def test_never_only_a_blank_text_box(self, state):
        actions = app._build_clarification_actions(state)
        assert actions, "an unresolved clarification must never produce an empty HITL UI"
        assert actions[-1].name == "manual_clarification"
        assert "Type manually" in actions[-1].label

    def test_bounded_channel_actions_plus_type_manually(self, mmidb_path):
        result = supervisor_node(_initial_state(mmidb_path, "hitl_ui"),
                                 llm=_channels_unresolved_llm(mmidb_path), run_id="hitl_ui")
        actions = app._build_clarification_actions(result)
        assert len(actions) >= 2
        assert [a.name for a in actions].count("manual_clarification") == 1
        assert all(a.payload["value"] for a in actions)

    def test_missing_options_fall_back_to_dataset_grounded_choices(self):
        state = {"clarification_question": "pick", "clarification_kind": "channel_selection",
                 "clarification_options": [], "dataset_available_channels": EEGMMIDB_LIKE}
        actions = app._build_clarification_actions(state)
        assert len(actions) > 1 and actions[-1].name == "manual_clarification"

    def test_frequency_choices_respect_sampling_rate(self):
        state = {"clarification": build_clarification("frequency_band", "band?"),
                 "dataset_sfreq": 60.0, "dataset_duration_seconds": 600.0}
        labels = [a.label for a in app._build_clarification_actions(state)]
        assert any("Theta" in l for l in labels)
        assert not any("Gamma" in l for l in labels)  # 45 Hz >= Nyquist(30 Hz)


# --------------------------------------------------------------------------- 3. persistence + same-checkpoint resume
class TestPersistenceAndResume:
    def test_channel_selection_persists_and_is_not_asked_again(self, mmidb_path):
        g = _Graph(mmidb_path, _channels_unresolved_llm(mmidb_path), "hitl_resume_ch")
        g.start()
        first = g.interrupt_payload["clarification"]
        assert first["kind"] == "channel_selection"

        g.reply(first["options"][0]["value"])

        v = g.values
        assert v["resolved_channel_info"]["resolved_channels"]  # persisted (validated vs dataset)
        assert v["resolved_frequency_band_info"]["name"] == "theta"  # earlier axis survives
        nxt = g.interrupt_payload["clarification"]
        assert nxt["kind"] == "condition"  # skips channels, asks ONLY the next unresolved axis
        assert {o["value"] for o in nxt["options"]} == {"T0", "T1", "T2"}
        assert g.next == ("clarification_pause",)

    def test_condition_T0_is_a_valid_selection_and_reaches_gate_1(self, mmidb_path):
        g = _Graph(mmidb_path, _channels_unresolved_llm(mmidb_path), "hitl_full_chain")
        g.start()
        g.reply("motor")
        assert g.interrupt_payload["clarification"]["kind"] == "condition"
        g.reply("T0")

        v = g.values
        assert v["resolved_condition_value"] == "T0"
        assert g.next == ("gate_1_review",)  # Gate 1 is NOT bypassed
        plan = v["plan"]
        assert plan.condition == "T0" and plan.freq_band.name == "theta"
        assert plan.channels == ["C3.", "C4.", "Cz."]  # raw dataset labels retained for MNE
        assert v["clarification"] is None and v["clarification_question"] is None
        assert v["preflight_confirmed"] is False and v["gate_1_approved"] is False

    def test_frequency_selection_persists_band_ok_and_is_not_asked_again(self, mmidb_path):
        g = _Graph(mmidb_path, _band_unresolved_llm(mmidb_path), "hitl_band")
        g.start()
        band = g.interrupt_payload["clarification"]
        assert band["kind"] == "frequency_band"
        assert [o["value"] for o in band["options"]] == ["delta", "theta", "alpha", "beta", "gamma"]

        g.reply("theta")

        assert g.values["resolved_frequency_band_info"]["name"] == "theta"
        assert g.interrupt_payload["clarification"]["kind"] == "condition"  # NOT frequency again

    def test_invalid_channel_option_keeps_axis_unresolved_and_reasks_same_checkpoint(self, mmidb_path):
        g = _Graph(mmidb_path, _channels_unresolved_llm(mmidb_path), "hitl_invalid")
        g.start()
        options = g.interrupt_payload["clarification"]["options"]

        g.reply("Zz9, Qq8")

        assert g.values["resolved_channel_info"] is None
        again = g.interrupt_payload["clarification"]
        assert again["kind"] == "channel_selection" and "Zz9" in again["question"]
        assert again["options"] == options and again["allow_manual"] is True
        assert g.next == ("clarification_pause",)
        g.reply(options[0]["value"])  # same checkpoint still resumable afterwards
        assert g.values["resolved_channel_info"]["resolved_channels"]

    def test_single_channel_is_rejected(self, mmidb_path):
        update, err = apply_clarification_reply(
            {"dataset_available_channels": EEGMMIDB_LIKE}, "channel_selection", "C3")
        assert update == {} and "two" in err

    def test_unknown_condition_is_rejected_and_real_labels_accepted(self, mmidb_path):
        state = {"raw_data_path": mmidb_path}
        assert apply_clarification_reply(state, "condition", "rest")[1] is not None
        assert apply_clarification_reply(state, "condition", "t2")[0] == {"resolved_condition_value": "T2"}

    def test_query_contradiction_clarification_has_authoritative_payload(self):
        contradiction = QueryTransformerResult("eeg_analysis", "x", "Condition axis", "Did you switch?")
        with patch.object(pipeline_nodes, "apply_input_rail", return_value=None), \
                patch.object(pipeline_nodes, "transform_query", return_value=contradiction):
            update = pipeline_nodes.query_transformer_node_adapter(
                {"user_request": "x", "latest_user_message": "x"})
        clar = get_clarification(update)
        assert clar["kind"] == "query_contradiction" and clar["allow_manual"] is True
        assert clar["options"] and update["clarification"] == clar


# --------------------------------------------------------------------------- 4. Chainlit layer end to end
class _Session(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

    def set(self, k, v):
        self[k] = v


class _Msg:
    sent = []

    def __init__(self, content="", **kw):
        self.content = content
        _Msg.sent.append(self)

    async def send(self):
        return self

    async def update(self):
        pass


def _ask_returning(result, log=None):
    class _Ask:
        def __init__(self, content="", actions=None, **kw):
            self.actions = actions
            if log is not None:
                log.append([a.name for a in (actions or [])])

        async def send(self):
            return result
    return _Ask


class TestChainlitFlow:
    def _session(self, path, graph):
        return _Session(graph=graph, data_path=path, is_real_data=True)

    def test_resume_after_first_clarification_reaches_second_clarification(self, mmidb_path):
        """UnboundLocalError regression: _resume_pipeline_sync crashed on a chained interrupt."""
        graph = compile_pipeline_app()
        sess = self._session(mmidb_path, graph)
        qt = QueryTransformerResult("eeg_analysis", "compute PLI theta", "none", "none")
        with patch.object(app.cl, "user_session", sess), \
                patch.object(pipeline_nodes, "apply_input_rail", return_value=None), \
                patch.object(pipeline_nodes, "transform_query", return_value=qt), \
                patch.object(pipeline_nodes, "get_supervisor_llm", return_value=_channels_unresolved_llm(mmidb_path)):
            out, nodes = app._run_pipeline_sync("compute PLI theta", "hitl_ui_resume")
            assert out["_routed_node"] == "clarification_pause"
            assert out["clarification"]["kind"] == "channel_selection"

            state, err = app._resume_pipeline_sync("hitl_ui_resume", {"reply": "motor", "value": "motor", "action": "clarification"})

        assert err is None, err
        assert state["_routed_node"] == "clarification_pause"
        assert state["clarification"]["kind"] == "condition"
        assert state["resolved_channel_info"]["resolved_channels"] == ["C3.", "C4.", "Cz."]

    def test_present_clarification_click_resumes_same_run(self, mmidb_path):
        sess = _Session()
        calls, shown = [], []
        clar = build_clarification("channel_selection", "Pick channels",
                                   [{"name": "region_motor", "value": "motor", "label": "Motor"}])
        picked = {"name": "region_motor", "payload": {"value": "motor"}, "label": "Motor"}

        async def fake_resume(run_id, reply, accumulated, msg=None):
            calls.append((run_id, reply, accumulated))

        async def go():
            with patch.object(app.cl, "user_session", sess), patch.object(app.cl, "Message", _Msg), \
                    patch.object(app.cl, "AskActionMessage", _ask_returning(picked, shown)), \
                    patch.object(app, "_resume_clarification", fake_resume):
                await app._present_clarification({"clarification": clar}, clar, "compute PLI", "run_x", _Msg())

        asyncio.run(go())
        assert shown == [["region_motor", "manual_clarification"]]
        assert calls == [("run_x", "motor", "compute PLI")]
        assert sess["pending_clarification_run_id"] == "run_x"

    def test_type_manually_asks_free_text_then_resumes(self):
        sess = _Session()
        calls = []
        clar = build_clarification("frequency_band", "Which band?")
        manual = {"name": "manual_clarification", "payload": {"value": "manual"}}

        class _AskUser:
            def __init__(self, content="", **kw):
                pass

            async def send(self):
                return {"output": " 8-12 Hz "}

        async def fake_resume(run_id, reply, accumulated, msg=None):
            calls.append(reply)

        async def go():
            with patch.object(app.cl, "user_session", sess), patch.object(app.cl, "Message", _Msg), \
                    patch.object(app.cl, "AskActionMessage", _ask_returning(manual)), \
                    patch.object(app.cl, "AskUserMessage", _AskUser), \
                    patch.object(app, "_resume_clarification", fake_resume):
                await app._present_clarification({"clarification": clar}, clar, "q", "run_y", _Msg())

        asyncio.run(go())
        assert calls == ["8-12 Hz"]

    def test_action_ui_failure_or_timeout_never_dead_ends(self):
        sess = _Session()
        clar = build_clarification("channel_selection", "Pick channels")
        _Msg.sent = []

        class _Boom:
            def __init__(self, *a, **k):
                pass

            async def send(self):
                raise RuntimeError("ui exploded")

        async def go():
            with patch.object(app.cl, "user_session", sess), patch.object(app.cl, "Message", _Msg), \
                    patch.object(app.cl, "AskActionMessage", _Boom):
                await app._present_clarification({"clarification": clar}, clar, "q", "run_z", _Msg())

        asyncio.run(go())
        assert sess["pending_clarification_run_id"] == "run_z"  # typed reply will resume the SAME checkpoint
        assert any("Type it in the chat box" in m.content for m in _Msg.sent)

    def test_handle_pipeline_output_routes_clarification_pause_to_hitl(self):
        sess = _Session()
        clar = build_clarification("channel_selection", "Pick channels")
        seen = []

        async def fake_present(output_state, clarification, acc, run_id, msg):
            seen.append((clarification["kind"], run_id))

        async def go():
            with patch.object(app.cl, "user_session", sess), patch.object(app, "_present_clarification", fake_present):
                await app._handle_pipeline_output(
                    {"_routed_node": "clarification_pause", "clarification": clar,
                     "clarification_question": "Pick channels"},
                    ["supervisor", "clarification_pause"], "q", "run_w", _Msg())

        asyncio.run(go())
        assert seen == [("channel_selection", "run_w")]


# --------------------------------------------------------------------------- 5. Gate 1 manifest synchronisation
class TestGate1ManifestSync:
    def test_approved_manifest_is_written_into_checkpoint_before_data_prep(self, mmidb_path):
        graph = compile_pipeline_app()
        sess = _Session(graph=graph, data_path=mmidb_path, is_real_data=True)
        qt = QueryTransformerResult("eeg_analysis", "compute PLI theta C3 C4 T1", "none", "none")
        llm = MockChatModel(responses=[
            _call("get_dataset_info", {"data_path": mmidb_path}),
            _call("get_dataset_conditions", {"data_path": mmidb_path}),
            _call("resolve_frequency_band", {"band_name_or_range": "theta", "sfreq": 160.0}),
            _call("resolve_channel_selection", {"requested_channels_or_region": "C3, C4",
                                                "available_channels": EEGMMIDB_LIKE}),
            "done",
        ])
        captured = {}

        def fake_run_data_prep(inp):
            captured["input"] = inp
            return SimpleNamespace(success=True, bad_channels_dropped=[], channel_plot_paths={},
                                   preprocessed_data_path="x.fif", error=None, summary=None)

        with patch.object(app.cl, "user_session", sess), \
                patch.object(pipeline_nodes, "apply_input_rail", return_value=None), \
                patch.object(pipeline_nodes, "transform_query", return_value=qt), \
                patch.object(pipeline_nodes, "get_supervisor_llm", return_value=llm), \
                patch("fc_pipeline.deterministic.data_prep.executor.run_data_prep", fake_run_data_prep):
            out, _ = app._run_pipeline_sync("compute PLI theta C3 C4 T1", "hitl_gate1")
            assert out["_routed_node"] == "gate_1_review" and out["plan"] is not None

            manifest = list(out["parameter_manifest"])
            row = next(m for m in manifest if m.name == "reference")
            row.human_approved_value = "average"  # human edit made in the Chainlit review UI
            state, err = app._resume_pipeline_sync("hitl_gate1", "approve", approved_manifest=manifest)

        assert err is None, err
        inp = captured["input"]
        assert inp.gate_1_approved is True and inp.preflight_confirmed is True
        assert next(m for m in inp.parameter_manifest if m.name == "reference").human_approved_value == "average"


# --------------------------------------------------------------------------- 6. post-run follow-up routing
class TestPostRunRouting:
    def _run(self, text, completed=True):
        sess = _Session(completed_run_active=completed, completed_run_state={"run_id": "r"},
                        original_query="q", accumulated_query="q", counter=0, graph=object())
        calls = {"pipeline": 0, "followup": 0}

        def boom(*a, **k):
            calls["pipeline"] += 1
            raise AssertionError("pipeline must not be restarted by a post-run follow-up")

        async def followup(t):
            calls["followup"] += 1
            return True

        async def go():
            with patch.object(app.cl, "user_session", sess), patch.object(app.cl, "Message", _Msg), \
                    patch.object(app, "_run_pipeline_sync", boom), \
                    patch.object(app, "_handle_completed_run_followup", followup):
                await app.on_message(SimpleNamespace(content=text))

        asyncio.run(go())
        return calls

    def test_plot_before_and_after_does_not_restart_pipeline(self):
        calls = self._run("plot the dataset before and after Data Prep")
        assert calls == {"pipeline": 0, "followup": 1}

    def test_only_explicit_new_query_leaves_completed_mode(self):
        calls = self._run("new query: compute PLI alpha", completed=True)
        assert calls["followup"] == 0 and calls["pipeline"] == 1
