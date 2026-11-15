"""Parallel multi-dataset orchestrator over the internal FastMCP server.

Each dataset runs in its own isolated asyncio task keyed by ``run_id``.
If one dataset is paused at Gate 1 (``gate_1_approved=False``), it returns
``status='paused_gate_1'`` without blocking other approved datasets from
completing Data Prep -> Connectivity -> Evaluator -> Synthesis in parallel.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fc_pipeline.mcp_server.client import call_mcp_tool


@dataclass
class DatasetJob:
    """Single dataset execution descriptor for parallel MCP orchestration."""

    run_id: str
    raw_data_path: str
    user_request: str = ""
    plan: Optional[Dict[str, Any]] = None
    parameter_manifest: List[Dict[str, Any]] = field(default_factory=list)
    gate_1_approved: bool = False


async def run_single_dataset_via_mcp(
    job: DatasetJob,
    *,
    output_dir: Optional[str] = None,
    use_llm: bool = False,
    embed_base64_images: bool = False,
) -> Dict[str, Any]:
    """Execute one dataset through the fixed 5-stage MCP flow without blocking others."""
    run_id = job.run_id
    if not job.gate_1_approved or not job.plan:
        return {
            "run_id": run_id,
            "status": "paused_gate_1",
            "gate_1_approved": False,
            "final_report_path": None,
            "error": None,
        }

    gate_res = await call_mcp_tool(
        "record_gate_1_approval",
        {
            "run_id": run_id,
            "plan": job.plan,
            "parameter_manifest": job.parameter_manifest,
        },
    )
    if not gate_res.get("ok"):
        return {
            "run_id": run_id,
            "status": "failed",
            "error": gate_res.get("error") or "Gate 1 approval recording failed.",
        }

    dp_args: Dict[str, Any] = {
        "run_id": run_id,
        "raw_data_path": job.raw_data_path,
        "plan": job.plan,
        "parameter_manifest": job.parameter_manifest,
    }
    if output_dir:
        dp_args["output_dir"] = output_dir
        dp_args["plot_dir"] = f"{output_dir}/plots"
    dp = await call_mcp_tool("run_data_prep", dp_args)
    if not dp.get("success"):
        return {"run_id": run_id, "status": "failed", "stage": "data_prep", "error": dp.get("error")}

    conn = await call_mcp_tool(
        "run_connectivity",
        {
            "run_id": run_id,
            "preprocessed_data_path": dp["preprocessed_data_path"],
            "plan": job.plan,
            "parameter_manifest": job.parameter_manifest,
            **({"output_dir": output_dir} if output_dir else {}),
        },
    )
    if not conn.get("success"):
        return {"run_id": run_id, "status": "failed", "stage": "connectivity", "error": conn.get("error")}

    eval_state: Dict[str, Any] = {
        "run_id": run_id,
        "raw_data_path": job.raw_data_path,
        "user_request": job.user_request,
        "preprocessed_data_path": dp["preprocessed_data_path"],
        "plan": job.plan,
        "parameter_manifest": job.parameter_manifest,
        "bad_channels_dropped": dp.get("bad_channels_dropped") or [],
        "channel_plot_paths": dp.get("channel_plot_paths") or {},
        "metric_csv_paths": conn.get("metric_csv_paths") or {},
        "heatmap_image_paths": conn.get("heatmap_image_paths") or {},
        "network_image_paths": conn.get("network_image_paths") or {},
        "numerical_summaries": conn.get("numerical_summaries") or {},
        "evidence_summary": conn.get("evidence_summary") or [],
    }
    ev = await call_mcp_tool(
        "run_evaluator_checks",
        {
            "run_id": run_id,
            "state_dict": eval_state,
            "use_llm": use_llm,
            **({"output_dir": output_dir} if output_dir else {}),
        },
    )
    if not ev.get("success"):
        return {"run_id": run_id, "status": "failed", "stage": "evaluator", "error": ev.get("error")}

    syn_state = {
        **eval_state,
        "evaluation_verdict": ev.get("evaluation_verdict"),
        "evaluation_summary_csv_path": ev.get("evaluation_summary_csv_path"),
    }
    syn = await call_mcp_tool(
        "run_synthesis",
        {
            "run_id": run_id,
            "state_dict": syn_state,
            "use_llm": use_llm,
            "embed_base64_images": embed_base64_images,
            **({"output_dir": output_dir} if output_dir else {}),
        },
    )
    if not syn.get("success"):
        return {"run_id": run_id, "status": "failed", "stage": "synthesis", "error": syn.get("error")}

    return {
        "run_id": run_id,
        "status": "completed",
        "gate_1_approved": True,
        "preprocessed_data_path": dp["preprocessed_data_path"],
        "evaluation_summary_csv_path": ev.get("evaluation_summary_csv_path"),
        "final_report_path": syn.get("final_report_path"),
        "run_context_path": syn.get("run_context_path"),
        "error": None,
    }


async def run_datasets_in_parallel(
    jobs: List[DatasetJob],
    *,
    output_dir: Optional[str] = None,
    use_llm: bool = False,
    embed_base64_images: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run multiple datasets concurrently via MCP; returns {run_id: result_dict}."""
    results = await asyncio.gather(
        *(
            run_single_dataset_via_mcp(
                job,
                output_dir=output_dir,
                use_llm=use_llm,
                embed_base64_images=embed_base64_images,
            )
            for job in jobs
        )
    )
    return {res["run_id"]: res for res in results}
