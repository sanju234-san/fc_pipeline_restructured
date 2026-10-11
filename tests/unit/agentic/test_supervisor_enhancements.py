"""Supervisor / plan-card enhancements: no metrics question (S1), single-condition
awareness (S2), readable names in the plan card (S4)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fc_pipeline.agentic.supervisor.agent import (
    _single_condition_question,
    _strip_metric_prompts,
    _unresolved_axes_question,
)
from fc_pipeline.agentic.supervisor.prompts import SUPERVISOR_SYSTEM_PROMPT
from fc_pipeline.pipeline.run_context import answer_run_question, display_channel, metric_label
from fc_pipeline.schemas.enums import MetricEnum

LLM_QUESTION = """Could you please specify the following details for the analysis?

1. **Condition** - which experimental condition(s) in the recording should be used?
2. **Frequency band** - the band of interest (e.g., alpha, 8-12 Hz).
3. **Channels** - the EEG electrodes or brain region (at least two channels).
4. **Metrics** - any particular connectivity metrics you want, or should I compute the full default set?

If you have particular connectivity metrics in mind, let me know; otherwise I'll compute the full set (PLI, wPLI, ImCoh, PLV, Coherence)."""


# ---- S1 -------------------------------------------------------------------


def test_metrics_are_never_part_of_a_clarification_question():
    out = _strip_metric_prompts(LLM_QUESTION)
    assert "metric" not in out.lower()
    assert "Condition" in out and "Frequency band" in out and "Channels" in out


def test_clarification_that_is_only_about_metrics_becomes_a_deterministic_axes_question():
    assert _strip_metric_prompts("Which metrics do you want?") == ""
    q = _unresolved_axes_question(band_ok=False, channels_ok=True, condition_ok=False)
    assert "Condition" in q and "Frequency band" in q
    assert "Channels" not in q and "metric" not in q.lower()


def test_supervisor_prompt_forbids_asking_about_metrics():
    assert "NEVER ASK ABOUT METRICS" in SUPERVISOR_SYSTEM_PROMPT


# ---- S2 -------------------------------------------------------------------


def test_single_condition_question_confirms_and_warns():
    q = _single_condition_question("T0")
    assert "only one condition label" in q and "**T0**" in q
    assert "not possible" in q


# ---- S4 -------------------------------------------------------------------


@pytest.mark.parametrize("raw,shown", [("C3..", "C3"), ("Fp1.", "Fp1"), ("Cz", "Cz"), (" F7. ", "F7"), (".", ".")])
def test_display_channel_strips_trailing_dots_only_for_display(raw, shown):
    assert display_channel(raw) == shown


def test_metric_enum_is_shown_as_a_readable_name():
    assert [metric_label(m) for m in MetricEnum] == ["PLI", "wPLI", "|ImCoh|", "PLV", "Coherence"]


def test_plan_card_shows_readable_metrics_and_channels():
    import chainlit_app

    plan = SimpleNamespace(
        freq_band=SimpleNamespace(name="theta", fmin=4.0, fmax=8.0),
        condition="T0",
        channels=["C3..", "C4..", "Cz.."],
        metrics=list(MetricEnum),
    )
    md = chainlit_app.format_plan_markdown(plan)
    assert "MetricEnum" not in md
    assert "`C3`" in md and "`Cz`" in md and ".." not in md
    assert "PLI" in md and "wPLI" in md and "Coherence" in md


def test_manifest_table_shows_readable_channel_and_metric_values():
    import chainlit_app
    from fc_pipeline.schemas.manifest import ParameterManifestEntry

    def row(name, value):
        return ParameterManifestEntry(
            name=name, category="scientific_axis", proposed_value=value,
            confidence=1.0, needs_human_input=False, risk_tier="low", human_approved_value=None,
        )

    md = chainlit_app.format_manifest_markdown([row("channels", "C3.., C4.., Cz.."), row("metrics", "pli, wpli, imcoh, plv, coh")])
    assert "C3, C4, Cz" in md
    # the pipes of |ImCoh| are escaped so they cannot break the table
    assert "PLI, wPLI, \\|ImCoh\\|, PLV, Coherence" in md


def test_followup_channel_answer_uses_readable_names_but_keeps_real_ones_for_matching():
    ctx = {
        "plan": {"channels": ["Fp1.", "Fp2.", "Cz.."], "freq_band": {"name": "theta", "fmin": 4, "fmax": 8}},
        "data_prep_summary": {"original_channel_count": 64, "dropped_channels": ["Fp2."]},
        "bad_channels_dropped": ["Fp2."],
    }
    text = answer_run_question("which channels were used", ctx)
    assert "Fp1, Fp2, Cz" in text
    assert "Retained:** Fp1, Cz" in text
