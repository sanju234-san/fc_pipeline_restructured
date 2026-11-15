"""Tests verifying parallel multi-dataset orchestration with non-blocking Gate 1 pause."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from fc_pipeline.mcp_server.orchestrator import (
    DatasetJob,
    run_datasets_in_parallel,
    run_single_dataset_via_mcp,
)
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.plan import AnalysisPlan, FrequencyBand


def _make_sample_fif(path: Path, *, cond_label: str = "T0", seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    sfreq = 160.0
    n_times = int(sfreq * 12)
    ch_names = ["Fp1", "Fp2", "F3", "F4"]
    data = rng.normal(0.0, 20e-6, size=(len(ch_names), n_times))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(onset=[1.0, 4.0, 7.0], duration=[2.0, 2.0, 2.0], description=[cond_label] * 3)
    )
    raw.save(str(path), overwrite=True, verbose=False)
    return path


def _plan_and_manifest(cond_label: str) -> tuple[dict, list[dict]]:
    plan = AnalysisPlan(
        metrics=["pli", "plv"],
        freq_band=FrequencyBand(name="alpha", fmin=8.0, fmax=12.0),
        channels=["Fp1", "Fp2", "F3", "F4"],
        condition=cond_label,
    )
    manifest = [
        ParameterManifestEntry(
            name="freq_band",
            category="frequency_band",
            proposed_value="alpha (8.0-12.0 Hz)",
            confidence=1.0,
            needs_human_input=False,
            risk_tier="low",
            human_approved_value="alpha (8.0-12.0 Hz)",
            rationale="Alpha band",
        ),
        ParameterManifestEntry(
            name="channels",
            category="channels",
            proposed_value="Fp1, Fp2, F3, F4",
            confidence=1.0,
            needs_human_input=False,
            risk_tier="low",
            human_approved_value="Fp1, Fp2, F3, F4",
            rationale="Frontal channels",
        ),
        ParameterManifestEntry(
            name="condition",
            category="condition",
            proposed_value=cond_label,
            confidence=1.0,
            needs_human_input=False,
            risk_tier="low",
            human_approved_value=cond_label,
            rationale="Condition",
        ),
        ParameterManifestEntry(
            name="reference",
            category="preprocessing",
            proposed_value="average",
            confidence=1.0,
            needs_human_input=False,
            risk_tier="elevated",
            human_approved_value="average",
            rationale="CAR",
        ),
        ParameterManifestEntry(
            name="bad_channel_variance_threshold",
            category="preprocessing",
            proposed_value="1e-15",
            confidence=1.0,
            needs_human_input=False,
            risk_tier="elevated",
            human_approved_value="1e-15",
            rationale="Flatline variance threshold",
        ),
        ParameterManifestEntry(
            name="min_cycles",
            category="preprocessing",
            proposed_value="3",
            confidence=1.0,
            needs_human_input=False,
            risk_tier="elevated",
            human_approved_value="3",
            rationale="Minimum cycles",
        ),
        ParameterManifestEntry(
            name="tau_phase",
            category="coupling_threshold",
            proposed_value="0.20",
            confidence=1.0,
            needs_human_input=False,
            risk_tier="elevated",
            human_approved_value="0.20",
            rationale="Default",
        ),
        ParameterManifestEntry(
            name="tau_zerolag",
            category="coupling_threshold",
            proposed_value="0.35",
            confidence=1.0,
            needs_human_input=False,
            risk_tier="elevated",
            human_approved_value="0.35",
            rationale="Default",
        ),
    ]
    return plan.model_dump(mode="json"), [m.model_dump(mode="json") for m in manifest]


@pytest.mark.asyncio
async def test_parallel_datasets_non_blocking_gate_1_pause(tmp_path: Path):
    raw_a = _make_sample_fif(tmp_path / "a_raw.fif", cond_label="T0", seed=1)
    raw_b = _make_sample_fif(tmp_path / "b_raw.fif", cond_label="T1", seed=2)
    raw_c = _make_sample_fif(tmp_path / "c_raw.fif", cond_label="T0", seed=3)
    out_dir = str(tmp_path / "outputs")

    plan_a, man_a = _plan_and_manifest("T0")
    plan_b, man_b = _plan_and_manifest("T1")
    plan_c, man_c = _plan_and_manifest("T0")

    jobs = [
        DatasetJob(run_id="batch_a", raw_data_path=str(raw_a), plan=plan_a, parameter_manifest=man_a, gate_1_approved=True),
        DatasetJob(run_id="batch_b", raw_data_path=str(raw_b), plan=plan_b, parameter_manifest=man_b, gate_1_approved=False),
        DatasetJob(run_id="batch_c", raw_data_path=str(raw_c), plan=plan_c, parameter_manifest=man_c, gate_1_approved=True),
    ]

    results = await run_datasets_in_parallel(jobs, output_dir=out_dir, use_llm=False)

    assert results["batch_a"]["status"] == "completed"
    assert results["batch_c"]["status"] == "completed"
    assert results["batch_b"]["status"] == "paused_gate_1"
    assert Path(results["batch_a"]["final_report_path"]).name == "report_batch_a.md"
    assert Path(results["batch_c"]["final_report_path"]).name == "report_batch_c.md"

    # Now approve Dataset B and resume it independently
    jobs[1].gate_1_approved = True
    res_b = await run_single_dataset_via_mcp(jobs[1], output_dir=out_dir, use_llm=False)
    assert res_b["status"] == "completed"
    assert Path(res_b["final_report_path"]).name == "report_batch_b.md"
