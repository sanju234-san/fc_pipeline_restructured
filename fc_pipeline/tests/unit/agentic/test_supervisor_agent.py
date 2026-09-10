"""Tests for fc_pipeline.agentic.supervisor.agent (ReAct loop, manifest, metrics, fallback parser)."""

import json
from typing import List
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand
from fc_pipeline.schemas.state import GraphState
from fc_pipeline.agentic.supervisor.agent import (
    canonicalize_metrics_inline,
    compile_and_confirm_manifest,
    supervisor_node,
)
from fc_pipeline.agentic.supervisor.tool_call_parser import extract_mistral_style_tool_calls


# ============================================================================
# MockChatModel: FakeListChatModel subclass implementing bind_tools
# ============================================================================


class MockChatModel(FakeListChatModel):
    """FakeListChatModel subclass that implements bind_tools for ReAct testing.

    LangChain's base FakeListChatModel does not implement bind_tools and raises
    NotImplementedError. This mock overrides bind_tools to return self so that
    agent ReAct loop tests run deterministically without a live LLM endpoint.
    """

    def bind_tools(self, tools, **kwargs):
        return self


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================


FIXTURE_CHANNELS = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]


def _make_plan(
    metrics: List[MetricEnum] = None,
    fmin: float = 8.0,
    fmax: float = 12.0,
    channels: List[str] = None,
    condition: str = "rest",
) -> AnalysisPlan:
    """Helper to assemble a valid AnalysisPlan for testing."""
    if metrics is None:
        metrics = [MetricEnum.PLI, MetricEnum.WPLI, MetricEnum.IMAGINARY_COHERENCE,
                   MetricEnum.PLV, MetricEnum.COHERENCE]
    if channels is None:
        channels = ["F3", "F4"]
    return AnalysisPlan(
        metrics=metrics,
        freq_band=FrequencyBand(name="alpha", fmin=fmin, fmax=fmax),
        channels=channels,
        condition=condition,
    )


def _tool_call(name: str, args: dict) -> str:
    """Helper to format a Mistral [TOOL_CALLS]name[ARGS]{...} string."""
    return f"[TOOL_CALLS]{name}[ARGS]{json.dumps(args)}"


# ============================================================================
# canonicalize_metrics_inline (8 tests)
# ============================================================================


class TestMetricCanonicalization:
    """Tests for deterministic inline metric canonicalization against METRIC_LOOKUP."""

    def test_unspecified_defaults_all_five(self):
        """[Regression] Unspecified query defaults to all 5 foundational metrics."""
        metrics, err = canonicalize_metrics_inline("analyze alpha on F3, F4")
        assert err is None
        assert len(metrics) == 5
        assert MetricEnum.PLI in metrics
        assert MetricEnum.WPLI in metrics
        assert MetricEnum.IMAGINARY_COHERENCE in metrics
        assert MetricEnum.PLV in metrics
        assert MetricEnum.COHERENCE in metrics

    def test_alias_pli(self):
        """[New] 'pli', 'phase lag index', 'phase-lag index' resolve to MetricEnum.PLI."""
        for phrase in ["compute pli", "calculate phase lag index", "run phase-lag index"]:
            metrics, err = canonicalize_metrics_inline(phrase)
            assert err is None
            assert metrics == [MetricEnum.PLI]

    def test_alias_wpli(self):
        """[New] 'wpli' resolves to MetricEnum.WPLI."""
        metrics, err = canonicalize_metrics_inline("compute wpli")
        assert err is None
        assert metrics == [MetricEnum.WPLI]

    def test_alias_imcoh(self):
        """[New] 'imcoh', 'icoh' resolve uniquely to MetricEnum.IMAGINARY_COHERENCE."""
        for phrase in ["compute imcoh", "calculate icoh"]:
            metrics, err = canonicalize_metrics_inline(phrase)
            assert err is None
            assert metrics == [MetricEnum.IMAGINARY_COHERENCE]

    def test_alias_plv(self):
        """[New] 'plv', 'phase locking value' resolve to MetricEnum.PLV."""
        for phrase in ["compute plv", "calculate phase locking value", "phase lock"]:
            metrics, err = canonicalize_metrics_inline(phrase)
            assert err is None
            assert metrics == [MetricEnum.PLV]

    def test_alias_coh(self):
        """[New] 'coh', 'coherence', 'spectral coherence' resolve to MetricEnum.COHERENCE."""
        for phrase in ["compute coh", "spectral coherence across channels"]:
            metrics, err = canonicalize_metrics_inline(phrase)
            assert err is None
            assert metrics == [MetricEnum.COHERENCE]

    def test_multiple_metrics_in_request(self):
        """[Regression] Request naming multiple metrics returns exactly those metrics."""
        metrics, err = canonicalize_metrics_inline("compute PLI and Coherence for alpha")
        assert err is None
        assert set(metrics) == {MetricEnum.PLI, MetricEnum.COHERENCE}

    def test_case_insensitivity_and_word_boundaries(self):
        """[New] Case insensitive; substrings don't trigger false positives."""
        metrics, err = canonicalize_metrics_inline("COMPUTE PLI ON CHANNELS")
        assert metrics == [MetricEnum.PLI]


# ============================================================================
# compile_and_confirm_manifest & Invariants (8 tests)
# ============================================================================


class TestManifestCompilationAndInvariants:
    """Tests for compile_and_confirm_manifest structural invariants and advisory check."""

    def test_valid_plan_compiles_manifest(self):
        """[Regression] Valid plan with all 5 metrics produces complete Gate 1 manifest."""
        plan = _make_plan()
        confidences = {"frequency_band": 1.0, "channels": 1.0, "condition": 1.0}
        manifest, err = compile_and_confirm_manifest(plan, confidences)
        assert err is None
        assert len(manifest) >= 4  # freq_band, channels, condition, metrics, thresholds
        names = [entry.name for entry in manifest]
        assert "freq_band" in names
        assert "channels" in names
        assert "condition" in names
        assert "metrics" in names
        # Advisory must be absent when both Phase-Robust and Zero-Lag metrics are present
        assert "cross_metric_synthesis" not in names

    def test_cross_metric_synthesis_advisory_fires_when_unbalanced(self):
        """[New] Advisory entry added when metrics lack both phase-robust and zero-lag."""
        plan = _make_plan(metrics=[MetricEnum.PLI, MetricEnum.WPLI])
        manifest, err = compile_and_confirm_manifest(plan, {})
        assert err is None
        advisory_entries = [e for e in manifest if e.name == "cross_metric_synthesis"]
        assert len(advisory_entries) == 1
        advisory = advisory_entries[0]
        assert advisory.category == "advisory"
        assert advisory.risk_tier == "elevated"
        assert advisory.needs_human_input is True
        assert "DISABLED" in advisory.proposed_value

    def test_cross_metric_synthesis_advisory_absent_when_balanced(self):
        """[New] Advisory entry NOT added when both phase-robust and zero-lag are present."""
        plan = _make_plan(metrics=[MetricEnum.PLI, MetricEnum.COHERENCE])
        manifest, err = compile_and_confirm_manifest(plan, {})
        assert err is None
        advisory_entries = [e for e in manifest if e.name == "cross_metric_synthesis"]
        assert len(advisory_entries) == 0

    def test_invariant_fewer_than_two_channels(self):
        """[New] Manifest compilation rejects plan with < 2 channels."""
        plan = _make_plan(channels=["F3"])
        manifest, err = compile_and_confirm_manifest(plan, {})
        assert manifest == []
        assert "at least 2 distinct channels" in err

    def test_invariant_fmin_greater_equal_fmax(self):
        """[New] Manifest compilation rejects plan where fmin >= fmax."""
        plan = _make_plan(fmin=12.0, fmax=8.0)
        manifest, err = compile_and_confirm_manifest(plan, {})
        assert manifest == []
        assert "strictly less than fmax" in err

    def test_invariant_empty_metrics(self):
        """[New] Manifest compilation rejects plan with empty metric list."""
        plan = _make_plan(metrics=[])
        manifest, err = compile_and_confirm_manifest(plan, {})
        assert manifest == []
        assert "Metric list cannot be empty" in err

    def test_invariant_missing_condition(self):
        """[New] Manifest compilation rejects plan with empty condition."""
        plan = _make_plan(condition="")
        manifest, err = compile_and_confirm_manifest(plan, {})
        assert manifest == []
        assert "Experimental condition must be specified" in err

    def test_low_confidence_trips_human_flag(self):
        """[Regression] Tool confidence < 0.80 sets needs_human_input=True in manifest."""
        plan = _make_plan()
        confidences = {"frequency_band": 0.70, "channels": 1.0, "condition": 1.0}
        manifest, err = compile_and_confirm_manifest(plan, confidences)
        assert err is None
        fb_entry = next(e for e in manifest if e.name == "freq_band")
        assert fb_entry.needs_human_input is True


# ============================================================================
# extract_mistral_style_tool_calls (3 tests)
# ============================================================================


class TestMistralFallbackParser:
    """Tests for raw Mistral tokenizer tool call extraction."""

    def test_standard_mistral_tool_call(self):
        """[Regression] Extracts single tool call from [TOOL_CALLS]name[ARGS]{...} tokens."""
        raw = '[TOOL_CALLS]resolve_frequency_band[ARGS]{"band_name_or_range": "alpha", "sfreq": 250.0}'
        calls = extract_mistral_style_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0]["name"] == "resolve_frequency_band"
        assert calls[0]["args"]["band_name_or_range"] == "alpha"
        assert calls[0]["args"]["sfreq"] == 250.0

    def test_multiple_mistral_tool_calls(self):
        """[New] Extracts multiple tool calls from consecutive [TOOL_CALLS] tokens."""
        raw = (
            '[TOOL_CALLS]get_dataset_info[ARGS]{"data_path": "test.fif"}'
            '[TOOL_CALLS]get_dataset_conditions[ARGS]{"data_path": "test.fif"}'
        )
        calls = extract_mistral_style_tool_calls(raw)
        assert len(calls) == 2
        assert calls[0]["name"] == "get_dataset_info"
        assert calls[1]["name"] == "get_dataset_conditions"

    def test_no_tool_calls_plain_text(self):
        """[New] Plain text without [TOOL_CALLS] returns empty list."""
        assert extract_mistral_style_tool_calls("Please clarify the channels.") == []


# ============================================================================
# supervisor_node ReAct Execution Loop (7 tests with MockChatModel)
# ============================================================================


class TestSupervisorReActLoop:
    """Tests for the Supervisor ReAct agent loop using MockChatModel."""

    def _make_state(self, data_path: str, query: str) -> GraphState:
        return {
            "pipeline_error": None,
            "raw_data_path": str(data_path),
            "user_request": query,
            "clarification_question": None,
            "plan": None,
            "parameter_manifest": None,
            "preflight_confirmed": False,
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

    def test_react_loop_successful_resolution(self, synthetic_eeg_path):
        """[Regression] Full 4-tool ReAct sequence resolves axes and compiles manifest."""
        path = str(synthetic_eeg_path)

        fake_llm = MockChatModel(responses=[
            _tool_call("get_dataset_info", {"data_path": path}),
            _tool_call("get_dataset_conditions", {"data_path": path}),
            _tool_call("resolve_frequency_band",
                       {"band_name_or_range": "alpha", "sfreq": 250.0}),
            _tool_call("resolve_channel_selection",
                       {"requested_channels_or_region": "F3, F4",
                        "available_channels": FIXTURE_CHANNELS}),
            "All parameters resolved. Analysis plan is ready for preflight review.",
        ])

        state = self._make_state(path, "compute PLI for alpha on F3, F4 during rest")
        result = supervisor_node(state, llm=fake_llm, run_id="test_success", max_iterations=8)

        assert result["clarification_question"] is None
        assert result["plan"] is not None
        assert result["plan"].condition == "rest"
        assert result["plan"].freq_band.name == "alpha"
        assert result["plan"].channels == ["F3", "F4"]
        assert MetricEnum.PLI in result["plan"].metrics
        assert result["parameter_manifest"] is not None
        assert result["preflight_confirmed"] is False

    def test_react_loop_fallback_mistral_parsing(self, synthetic_eeg_path):
        """[New] Mistral [TOOL_CALLS] in content parsed by fallback extractor."""
        path = str(synthetic_eeg_path)

        fake_llm = MockChatModel(responses=[
            _tool_call("get_dataset_info", {"data_path": path}),
            _tool_call("get_dataset_conditions", {"data_path": path}),
            _tool_call("resolve_frequency_band",
                       {"band_name_or_range": "alpha", "sfreq": 250.0}),
            _tool_call("resolve_channel_selection",
                       {"requested_channels_or_region": "F3, F4",
                        "available_channels": FIXTURE_CHANNELS}),
            "Plan assembled.",
        ])

        state = self._make_state(path, "compute PLI for alpha on F3, F4 during rest")
        result = supervisor_node(state, llm=fake_llm, run_id="test_fallback", max_iterations=8)

        assert result["plan"] is not None
        assert result["parameter_manifest"] is not None

    def test_missing_band_halts(self, synthetic_eeg_path):
        """[New] Skipping resolve_frequency_band halts with clarification."""
        path = str(synthetic_eeg_path)

        fake_llm = MockChatModel(responses=[
            _tool_call("get_dataset_info", {"data_path": path}),
            _tool_call("get_dataset_conditions", {"data_path": path}),
            _tool_call("resolve_channel_selection",
                       {"requested_channels_or_region": "F3, F4",
                        "available_channels": FIXTURE_CHANNELS}),
            "I need to know the frequency band for this analysis.",
        ])

        state = self._make_state(path, "compute PLI on F3, F4 during rest")
        result = supervisor_node(state, llm=fake_llm, run_id="test_no_band", max_iterations=8)

        assert result["plan"] is None
        assert result["parameter_manifest"] is None
        assert result["clarification_question"] is not None

    def test_missing_channels_halts(self, synthetic_eeg_path):
        """[New] Skipping resolve_channel_selection halts with clarification."""
        path = str(synthetic_eeg_path)

        fake_llm = MockChatModel(responses=[
            _tool_call("get_dataset_info", {"data_path": path}),
            _tool_call("get_dataset_conditions", {"data_path": path}),
            _tool_call("resolve_frequency_band",
                       {"band_name_or_range": "alpha", "sfreq": 250.0}),
            "Which channels or brain region should I use?",
        ])

        state = self._make_state(path, "compute PLI for alpha during rest")
        result = supervisor_node(state, llm=fake_llm, run_id="test_no_channels", max_iterations=8)

        assert result["plan"] is None
        assert result["parameter_manifest"] is None
        assert result["clarification_question"] is not None

    def test_missing_condition_halts(self, synthetic_eeg_path):
        """[New] Missing condition in user_request halts with clarification."""
        path = str(synthetic_eeg_path)

        fake_llm = MockChatModel(responses=[
            _tool_call("get_dataset_info", {"data_path": path}),
            _tool_call("get_dataset_conditions", {"data_path": path}),
            _tool_call("resolve_frequency_band",
                       {"band_name_or_range": "alpha", "sfreq": 250.0}),
            _tool_call("resolve_channel_selection",
                       {"requested_channels_or_region": "F3, F4",
                        "available_channels": FIXTURE_CHANNELS}),
            "What experimental condition should I use?",
        ])

        state = self._make_state(path, "compute PLI for alpha on F3, F4")
        result = supervisor_node(state, llm=fake_llm, run_id="test_no_condition", max_iterations=8)

        assert result["plan"] is None
        assert result["parameter_manifest"] is None
        assert result["clarification_question"] is not None

    def test_max_iterations_exhaustion(self, synthetic_eeg_path):
        """[New] Loop terminates at max_iterations without hanging."""
        path = str(synthetic_eeg_path)

        fake_llm = MockChatModel(responses=[
            _tool_call("get_dataset_info", {"data_path": path}),
        ])

        state = self._make_state(path, "analyze this EEG during rest")
        result = supervisor_node(state, llm=fake_llm, run_id="test_max_iter", max_iterations=3)

        assert result["plan"] is None
        assert result["parameter_manifest"] is None
        assert result["clarification_question"] is not None

    def test_direct_clarification_no_tools(self, synthetic_eeg_path):
        """[Regression] LLM immediately asks for clarification without tool calls."""
        path = str(synthetic_eeg_path)

        fake_llm = MockChatModel(responses=[
            "I need more information. What frequency band, channels, and condition?",
        ])

        state = self._make_state(path, "analyze this EEG")
        result = supervisor_node(state, llm=fake_llm, run_id="test_direct_clarif", max_iterations=8)

        assert result["plan"] is None
        assert result["parameter_manifest"] is None
        assert result["clarification_question"] is not None
        assert "information" in result["clarification_question"].lower()
