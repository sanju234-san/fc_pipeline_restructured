"""Tests for scientific completeness checks (trial adequacy, bad channel, frequency cycle)."""

import pytest
from typing import List
from unittest.mock import MagicMock, patch

from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.agentic.supervisor.agent import compile_and_confirm_manifest
from fc_pipeline.agentic.supervisor.tools.frequency_band import resolve_frequency_band
from fc_pipeline.config.thresholds import N_CYCLES_MIN

# Helper from test_supervisor_agent.py
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

class TestScientificCompletenessChecks:
    """Tests for the scientific completeness checks."""

    def test_trial_adequacy_warning_triggers(self):
        """Verify that a warning is added for conditions with fewer than 3 trials."""
        plan = _make_plan(condition="visual stimulus with two trials")
        manifest, err = compile_and_confirm_manifest(plan, {}, trial_count=2)
        assert err is None

        adequacy_warnings = [
            entry for entry in manifest
            if entry.name == "trial_adequacy"
        ]
        assert len(adequacy_warnings) == 1
        warning = adequacy_warnings[0]
        assert warning.risk_tier == "elevated"
        assert "Only 2 trial(s) available" in warning.proposed_value
        assert warning.needs_human_input is True

    def test_trial_adequacy_warning_not_triggered_for_enough_trials(self):
        """Verify no warning is added for conditions with 3 or more trials."""
        plan = _make_plan(condition="visual stimulus with ten trials")
        manifest, err = compile_and_confirm_manifest(plan, {}, trial_count=10)
        assert err is None

        adequacy_warnings = [
            entry for entry in manifest
            if entry.name == "trial_adequacy"
        ]
        assert len(adequacy_warnings) == 0

    def test_bad_channel_screening_note_appears(self):
        """Verify the static bad-channel screening note is always added to the manifest."""
        plan = _make_plan()
        manifest, err = compile_and_confirm_manifest(plan, {})
        assert err is None

        screening_notes = [
            entry for entry in manifest if entry.name == "bad_channel_screening"
        ]
        assert len(screening_notes) == 1
        note = screening_notes[0]
        assert note.category == "informational"
        assert note.risk_tier == "low"
        assert "Not yet performed" in note.proposed_value
        assert note.needs_human_input is False

    def test_frequency_band_cycle_check_triggers(self):
        """Verify that the frequency-band cycle check triggers for short durations."""
        # fmin=8, N_CYCLES_MIN=3, so min_duration = 3/8 = 0.375s
        # duration_seconds = 0.3 should trigger the warning
        observation = resolve_frequency_band.invoke(
            {"band_name_or_range": "alpha", "sfreq": 250.0, "duration_seconds": 0.3}
        )
        assert "error" in observation
        assert "Epoch duration (0.30s) is too short" in observation["error"]
        assert observation["confidence"] == 0.0

    def test_frequency_band_cycle_check_not_triggered_for_enough_duration(self):
        """Verify that the frequency-band cycle check does not trigger for sufficient durations."""
        # fmin=8, N_CYCLES_MIN=3, so min_duration = 3/8 = 0.375s
        # duration_seconds = 0.5 should not trigger the warning
        observation = resolve_frequency_band.invoke(
            {"band_name_or_range": "alpha", "sfreq": 250.0, "duration_seconds": 0.5}
        )
        assert observation["error"] is None
        assert observation["confidence"] == 1.0
        assert observation["fmin"] == 8
        assert observation["fmax"] == 12