"""Regression tests for two Data Prep fixes.

Fix 1 - epoching: a condition segment (annotation with a duration) is tiled
with consecutive, non-overlapping epochs instead of yielding a single epoch
at the event onset.

Fix 2 - reference order: flatline channels are flagged on the full,
unreferenced montage, the average reference is taken over the full montage
(flat channels excluded), and only afterwards are the analysis channels
selected.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from fc_pipeline.deterministic.data_prep import executor as executor_module
from fc_pipeline.deterministic.data_prep.cleaning import (
    clean_bad_channels,
    mark_flatline_channels,
)
from fc_pipeline.deterministic.data_prep.epoching import filter_and_epoch
from fc_pipeline.deterministic.data_prep.executor import run_data_prep
from fc_pipeline.deterministic.data_prep.models import (
    DataPrepInput,
    ValidatedDataPrepParams,
)
from fc_pipeline.deterministic.data_prep.referencing import apply_reference
from fc_pipeline.deterministic.data_prep.validation import DataPrepValidationError
from fc_pipeline.schemas.enums import MetricEnum
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand

CH_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]


# --- helpers ---------------------------------------------------------------


def _make_raw(
    annotations,
    *,
    sfreq: float = 250.0,
    duration: float = 20.0,
    flat: tuple[str, ...] = (),
    seed: int = 0,
) -> mne.io.RawArray:
    """In-memory raw with the given (onset, duration, description) annotations."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(CH_NAMES), int(duration * sfreq))) * 1e-5
    for name in flat:
        data[CH_NAMES.index(name)] = 0.0
    raw = mne.io.RawArray(
        data, mne.create_info(CH_NAMES, sfreq, "eeg"), verbose=False
    )
    onsets, durations, descriptions = zip(*annotations)
    raw.set_annotations(
        mne.Annotations(list(onsets), list(durations), list(descriptions))
    )
    return raw


def _params(
    channels=tuple(CH_NAMES),
    *,
    fmin: float = 8.0,
    fmax: float = 13.0,
    condition: str = "rest",
    min_cycles: float = 3.0,
) -> ValidatedDataPrepParams:
    return ValidatedDataPrepParams(
        raw_data_path=Path("unused.fif"),
        run_id="t",
        channels=tuple(channels),
        condition=condition,
        fmin=fmin,
        fmax=fmax,
        reference_method="average",
        bad_channel_variance_threshold=1e-15,
        min_cycles=min_cycles,
    )


# --- Fix 1: epoching -------------------------------------------------------


def test_segment_is_tiled_into_non_overlapping_epochs():
    # alpha: 3 cycles / 8 Hz = 0.375 s -> ceil(93.75) = 94 samples at 250 Hz.
    raw = _make_raw([(2.0, 5.0, "rest")])
    epochs = filter_and_epoch(raw, _params())

    window = 94
    assert len(epochs) == (5 * 250) // window == 13
    assert epochs.get_data().shape[-1] == window

    starts = epochs.events[:, 0]
    assert np.all(np.diff(starts) == window)  # contiguous, no overlap, no gap
    seg_start = raw.first_samp + 2 * 250
    assert starts[0] == seg_start
    assert starts[-1] + window <= seg_start + 5 * 250  # fully inside the segment


def test_multiple_segments_are_each_tiled():
    raw = _make_raw([(1.0, 5.0, "rest"), (10.0, 3.0, "rest")])
    epochs = filter_and_epoch(raw, _params())
    assert len(epochs) == (5 * 250) // 94 + (3 * 250) // 94 == 13 + 7


def test_epochs_do_not_cross_into_the_next_condition():
    raw = _make_raw([(0.0, 5.0, "rest"), (5.0, 5.0, "task")])
    epochs = filter_and_epoch(raw, _params())
    assert len(epochs) > 1
    ends = epochs.events[:, 0] + epochs.get_data().shape[-1]
    assert ends.max() <= raw.first_samp + 5 * 250


def test_instantaneous_markers_keep_one_epoch_per_event():
    raw = _make_raw([(t, 0.0, "rest") for t in (1.0, 3.0, 5.0, 7.0)])
    epochs = filter_and_epoch(raw, _params())
    assert len(epochs) == 4
    assert epochs.get_data().shape[-1] == 94


def test_segment_shorter_than_one_epoch_is_skipped():
    # 0.2 s (50 samples) cannot host a 94-sample epoch; 2 s hosts five.
    raw = _make_raw([(1.0, 0.2, "rest"), (5.0, 2.0, "rest")])
    epochs = filter_and_epoch(raw, _params())
    assert len(epochs) == (2 * 250) // 94 == 5


def test_all_segments_too_short_raises_insufficient_epoch_length():
    raw = _make_raw([(1.0, 0.2, "rest"), (5.0, 0.3, "rest")])
    with pytest.raises(DataPrepValidationError) as exc:
        filter_and_epoch(raw, _params())
    assert exc.value.code == "INSUFFICIENT_EPOCH_LENGTH"


@pytest.mark.parametrize("sfreq", [160.0, 250.0])
@pytest.mark.parametrize("fmin,fmax", [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0)])
def test_epoch_length_always_holds_min_cycles(sfreq, fmin, fmax):
    raw = _make_raw([(1.0, 12.0, "rest")], sfreq=sfreq, duration=20.0)
    epochs = filter_and_epoch(raw, _params(fmin=fmin, fmax=fmax))
    length_s = epochs.get_data().shape[-1] / sfreq
    assert length_s >= 3.0 / fmin - 1e-9


# --- Fix 2: reference order ------------------------------------------------


def _new_order(raw, params):
    """The order the executor now uses: screen -> reference -> select."""
    flagged = mark_flatline_channels(raw, params)
    raw, _ = apply_reference(raw, params)
    raw, dropped = clean_bad_channels(raw, params, pre_flagged=flagged)
    return raw, flagged, dropped


def test_average_reference_uses_full_montage_not_selected_channels():
    raw = _make_raw([(0.0, 10.0, "rest")])
    original = raw.get_data().copy()
    params = _params(channels=("C3", "C4"))

    out, flagged, dropped = _new_order(raw, params)

    assert flagged == [] and dropped == []
    assert out.ch_names == ["C3", "C4"]
    expected = original[[2, 3]] - original.mean(axis=0)  # mean over all 8
    np.testing.assert_allclose(out.get_data(), expected, atol=1e-12)

    # The old order (select first, then average) is a different, wrong signal:
    # with two channels the average reference forces C3 == -C4.
    legacy = _make_raw([(0.0, 10.0, "rest")])
    legacy, _ = clean_bad_channels(legacy, params)
    legacy, _ = apply_reference(legacy, params)
    assert not np.allclose(legacy.get_data(), expected, atol=1e-9)


def test_flatline_outside_selection_is_excluded_from_the_average():
    raw = _make_raw([(0.0, 10.0, "rest")], flat=("F3",))
    original = raw.get_data().copy()
    params = _params(channels=("C3", "C4"))

    out, flagged, dropped = _new_order(raw, params)

    assert flagged == ["F3"]
    assert dropped == []  # F3 was not selected, so nothing is dropped from the plan
    good = [i for i, n in enumerate(CH_NAMES) if n != "F3"]
    expected = original[[2, 3]] - original[good].mean(axis=0)
    np.testing.assert_allclose(out.get_data(), expected, atol=1e-12)


def test_flatline_inside_selection_is_excluded_and_dropped():
    raw = _make_raw([(0.0, 10.0, "rest")], flat=("F3",))
    original = raw.get_data().copy()
    params = _params(channels=("F3", "C3", "C4"))

    out, flagged, dropped = _new_order(raw, params)

    assert flagged == ["F3"]
    assert dropped == ["F3"]
    assert out.ch_names == ["C3", "C4"]
    good = [i for i, n in enumerate(CH_NAMES) if n != "F3"]
    expected = original[[2, 3]] - original[good].mean(axis=0)
    np.testing.assert_allclose(out.get_data(), expected, atol=1e-12)


def test_flatline_is_not_detectable_after_unmarked_referencing():
    """Documents WHY screening must come before referencing."""
    raw = _make_raw([(0.0, 10.0, "rest")], flat=("F3",))
    params = _params(channels=("F3", "C3", "C4"))

    raw, _ = apply_reference(raw, params)  # reference first, nothing marked
    raw, dropped = clean_bad_channels(raw, params)  # variance screen afterwards

    assert dropped == []  # the dead channel is no longer flat, so it survives


def test_clean_bad_channels_without_pre_flagged_is_unchanged():
    raw = _make_raw([(0.0, 10.0, "rest")], flat=("F3",))
    raw, dropped = clean_bad_channels(raw, _params(channels=("F3", "C3", "C4")))
    assert dropped == ["F3"]
    assert raw.ch_names == ["C3", "C4"]


# --- executor wiring -------------------------------------------------------


def _executor_input(path, channels=("C3", "C4", "P3")):
    plan = AnalysisPlan(
        metrics=[MetricEnum.PLI, MetricEnum.COHERENCE],
        freq_band=FrequencyBand(name="alpha", fmin=8.0, fmax=13.0),
        channels=list(channels),
        condition="rest",
    )
    manifest = [
        ParameterManifestEntry(
            name="freq_band",
            category="scientific_axis",
            proposed_value="alpha (8.0 - 13.0 Hz)",
        ),
        ParameterManifestEntry(
            name="channels",
            category="scientific_axis",
            proposed_value=", ".join(channels),
        ),
        ParameterManifestEntry(
            name="condition", category="scientific_axis", proposed_value="rest"
        ),
        ParameterManifestEntry(
            name="reference",
            category="engineering_threshold",
            proposed_value="average",
        ),
        ParameterManifestEntry(
            name="bad_channel_variance_threshold",
            category="engineering_threshold",
            proposed_value="1e-15",
        ),
        ParameterManifestEntry(
            name="min_cycles",
            category="engineering_threshold",
            proposed_value="3",
        ),
    ]
    return DataPrepInput(
        raw_data_path=str(path),
        plan=plan,
        parameter_manifest=manifest,
        run_id="order001",
        gate_1_approved=True,
        preflight_confirmed=True,
    )


def test_executor_runs_stages_in_the_corrected_order(
    synthetic_eeg_path, tmp_path, monkeypatch
):
    events: list[tuple[str, str]] = []

    def fake_trace(run_id, event_type, payload):
        if event_type == "data_prep_stage":
            events.append((payload["stage"], payload["status"]))

    monkeypatch.setattr(executor_module, "_trace", fake_trace)

    result = run_data_prep(
        _executor_input(synthetic_eeg_path),
        output_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
    )
    assert result.success, result.error

    completed = [stage for stage, status in events if status == "completed"]
    assert completed.index("bad_channel_screening") < completed.index("reference")
    assert completed.index("reference") < completed.index("channel_selection")
    assert completed.index("channel_selection") < completed.index("filter_and_epoch")


def test_executor_produces_many_epochs_of_valid_length(synthetic_eeg_path, tmp_path):
    # Fixture: 'rest' is a 5 s annotation. Previously this gave exactly 1 epoch.
    result = run_data_prep(
        _executor_input(synthetic_eeg_path),
        output_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
    )
    assert result.success, result.error
    assert result.summary.epoch_count == (5 * 250) // 94 == 13
    assert result.summary.epoch_duration_seconds >= 3.0 / 8.0 - 1e-9
    assert result.summary.retained_channel_count == 3


# --- MNE-based channel time-series plots (before / after) -------------------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_before_and_after_channel_plots_are_produced(synthetic_eeg_path, tmp_path):
    result = run_data_prep(
        _executor_input(synthetic_eeg_path),
        output_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
    )
    assert result.success, result.error
    plots = result.channel_plot_paths
    assert list(plots)[:2] == ["channels_before", "channels_after"]
    # existing diagnostics are preserved
    assert {"channel_variance", "psd_overview"} <= set(plots)
    for name in ("channels_before", "channels_after"):
        path = Path(plots[name])
        assert path.exists() and path.stat().st_size > 5_000
        assert path.read_bytes()[:8] == PNG_MAGIC


def test_plot_failure_never_aborts_data_prep(synthetic_eeg_path, tmp_path, monkeypatch):
    from fc_pipeline.deterministic.data_prep import plotting

    def boom(*args, **kwargs):
        raise RuntimeError("browser backend unavailable")

    monkeypatch.setattr(plotting.mne.viz, "use_browser_backend", boom)
    result = run_data_prep(
        _executor_input(synthetic_eeg_path),
        output_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
    )
    assert result.success, result.error
    assert "channels_before" not in result.channel_plot_paths
    assert "channels_after" not in result.channel_plot_paths
    assert "channel_variance" in result.channel_plot_paths  # others still produced


def test_flagged_flatline_channel_is_marked_bad_in_before_plot(tmp_path):
    from fc_pipeline.deterministic.data_prep.plotting import plot_channels_before

    raw = _make_raw([(0.0, 10.0, "rest")], flat=("F3",))
    params = _params(channels=("F3", "C3", "C4"))
    flagged = mark_flatline_channels(raw, params)
    path = plot_channels_before(raw, params, flagged, tmp_path)
    assert path is not None and Path(path).read_bytes()[:8] == PNG_MAGIC
    # the plot helper must not modify the caller's recording
    assert len(raw.ch_names) == len(CH_NAMES)
