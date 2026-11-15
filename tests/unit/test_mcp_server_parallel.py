"""Unit & concurrency tests for the internal FastMCP server and parallel workers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import mne
import numpy as np
import pytest

from fc_pipeline.mcp_server.client import call_mcp_tool, list_mcp_tools
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
    ann = mne.Annotations(
        onset=[1.0, 4.0, 7.0],
        duration=[2.0, 2.0, 2.0],
        description=[cond_label, cond_label, cond_label],
    )
    raw.set_annotations(ann)
    raw.save(str(path), overwrite=True, verbose=False)
    return path


def _make_plan_and_manifest(cond_label: str) -> tuple[dict, list[dict]]:
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
async def test_mcp_tool_inventory_and_no_db_tools():
    tools = set(await list_mcp_tools())
    expected = {
        "get_dataset_info",
        "get_dataset_conditions",
        "resolve_frequency_band",
        "resolve_channel_selection",
        "generate_dataset_overview_plot",
        "record_gate_1_approval",
        "run_data_prep",
        "run_connectivity",
        "run_evaluator_checks",
        "run_synthesis",
        "read_offloaded_context",
        "search_run_context",
    }
    assert expected.issubset(tools)
    assert not any("sql" in t or "db" in t or "delete" in t for t in tools)


@pytest.mark.asyncio
async def test_mcp_supervisor_discovery_tools_unwrapped():
    band = await call_mcp_tool(
        "resolve_frequency_band",
        {"band_name_or_range": "alpha", "sfreq": 160.0, "duration_seconds": 10.0},
    )
    assert band["name"] == "alpha"
    assert band["fmin"] == 8.0 and band["fmax"] == 12.0

    ch = await call_mcp_tool(
        "resolve_channel_selection",
        {"requested_channels_or_region": "Fp1, Fp2", "available_channels": ["Fp1", "Fp2", "F3"]},
    )
    assert ch["resolved_channels"] == ["Fp1", "Fp2"]


@pytest.mark.asyncio
async def test_mcp_gate_1_blocks_unapproved_data_prep(tmp_path: Path):
    raw_file = _make_sample_fif(tmp_path / "unapproved_raw.fif")
    plan_dict, manifest_list = _make_plan_and_manifest("T0")
    res = await call_mcp_tool(
        "run_data_prep",
        {
            "run_id": "unapproved_run_99",
            "raw_data_path": str(raw_file),
            "plan": plan_dict,
            "parameter_manifest": manifest_list,
            "output_dir": str(tmp_path / "out"),
        },
    )
    assert res["success"] is False
    assert "GATE_1_NOT_APPROVED" in res["error"]


@pytest.mark.asyncio
async def test_mcp_parallel_multi_dataset_execution(tmp_path: Path):
    raw_a = _make_sample_fif(tmp_path / "ds_a_raw.fif", cond_label="T0", seed=1)
    raw_b = _make_sample_fif(tmp_path / "ds_b_raw.fif", cond_label="T1", seed=2)
    out_dir = tmp_path / "outputs"

    async def _run_one(run_id: str, raw_path: Path, cond: str) -> dict:
        plan_dict, manifest_list = _make_plan_and_manifest(cond)
        gate_res = await call_mcp_tool(
            "record_gate_1_approval",
            {"run_id": run_id, "plan": plan_dict, "parameter_manifest": manifest_list},
        )
        assert gate_res["ok"] is True

        dp = await call_mcp_tool(
            "run_data_prep",
            {
                "run_id": run_id,
                "raw_data_path": str(raw_path),
                "output_dir": str(out_dir),
                "plot_dir": str(out_dir / "plots"),
            },
        )
        assert dp["success"] is True, dp.get("error")

        conn = await call_mcp_tool(
            "run_connectivity",
            {
                "run_id": run_id,
                "preprocessed_data_path": dp["preprocessed_data_path"],
                "plan": plan_dict,
                "parameter_manifest": manifest_list,
                "output_dir": str(out_dir),
            },
        )
        assert conn["success"] is True, conn.get("error")

        state_for_eval = {
            "run_id": run_id,
            "raw_data_path": str(raw_path),
            "preprocessed_data_path": dp["preprocessed_data_path"],
            "plan": plan_dict,
            "parameter_manifest": manifest_list,
            "bad_channels_dropped": dp["bad_channels_dropped"],
            "channel_plot_paths": dp["channel_plot_paths"],
            "metric_csv_paths": conn["metric_csv_paths"],
            "heatmap_image_paths": conn["heatmap_image_paths"],
            "network_image_paths": conn["network_image_paths"],
            "numerical_summaries": conn["numerical_summaries"],
            "evidence_summary": conn["evidence_summary"],
        }
        ev = await call_mcp_tool(
            "run_evaluator_checks",
            {"run_id": run_id, "state_dict": state_for_eval, "use_llm": False, "output_dir": str(out_dir)},
        )
        assert ev["success"] is True, ev.get("error")

        state_for_syn = {
            **state_for_eval,
            "evaluation_verdict": ev["evaluation_verdict"],
            "evaluation_summary_csv_path": ev["evaluation_summary_csv_path"],
        }
        syn = await call_mcp_tool(
            "run_synthesis",
            {"run_id": run_id, "state_dict": state_for_syn, "use_llm": False, "output_dir": str(out_dir)},
        )
        assert syn["success"] is True, syn.get("error")
        return syn

    res_a, res_b = await asyncio.gather(
        _run_one("par_run_a", raw_a, "T0"),
        _run_one("par_run_b", raw_b, "T1"),
    )

    assert Path(res_a["final_report_path"]).name == "report_par_run_a.md"
    assert Path(res_b["final_report_path"]).name == "report_par_run_b.md"
    assert (out_dir / "run_context_par_run_a.json").is_file()
    assert (out_dir / "run_context_par_run_b.json").is_file()

