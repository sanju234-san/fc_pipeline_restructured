"""Internal FastMCP server wrapping ToolboxRegistry & parallel stage workers.

Exposes:
  - 5 fine-grained Supervisor discovery tools (unwrapped from ToolboxRegistry)
  - 1 Gate 1 approval ledger tool (record_gate_1_approval)
  - 4 stage-level execution tools (run_data_prep, run_connectivity,
    run_evaluator_checks, run_synthesis) via ProcessPoolExecutor('spawn')
  - 2 read-only follow-up context tools (read_offloaded_context, search_run_context)

Zero database read/write tools are registered.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import Context, FastMCP

from fc_pipeline.mcp_server.workers import (
    worker_connectivity,
    worker_data_prep,
    worker_evaluator_checks,
    worker_synthesis,
)
from fc_pipeline.toolbox.registry import toolbox_registry

mcp = FastMCP("fc_pipeline_internal")

_DEFAULT_MAX_WORKERS = max(1, min(int(os.getenv("MCP_MAX_WORKERS", "2")), os.cpu_count() or 2))
_POOL: Optional[ProcessPoolExecutor] = None
_APPROVED_RUNS: Dict[str, Dict[str, Any]] = {}
_RUN_LOCKS: Dict[str, asyncio.Lock] = {}
_PLOT_LOCK = asyncio.Lock()
_LLM_SEMAPHORE = asyncio.Semaphore(2)


def _get_pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        ctx = multiprocessing.get_context("spawn")
        _POOL = ProcessPoolExecutor(max_workers=_DEFAULT_MAX_WORKERS, mp_context=ctx)
    return _POOL


def _get_run_lock(run_id: str) -> asyncio.Lock:
    lock = _RUN_LOCKS.get(run_id)
    if lock is None:
        lock = asyncio.Lock()
        _RUN_LOCKS[run_id] = lock
    return lock


def _unwrap_registry_tool(name: str):
    tool_obj = toolbox_registry.get_tool(name)
    return getattr(tool_obj, "func", tool_obj)


async def _run_with_heartbeats(
    ctx: Optional[Context],
    fn,
    /,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a picklable worker in ProcessPoolExecutor while emitting progress heartbeats."""
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_get_pool(), _call_kw, fn, kwargs)
    step = 0
    while not fut.done():
        if ctx is not None:
            try:
                await ctx.report_progress(progress=step, total=None)
            except Exception:
                pass
        step += 1
        done, _ = await asyncio.wait({fut}, timeout=2.0)
        if done:
            break
    return await fut


def _call_kw(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# 1. Five fine-grained Supervisor discovery tools (from ToolboxRegistry)
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_dataset_info(data_path: str) -> Dict[str, Any]:
    """Inspect EEG recording header via MNE (preload=False, PHI scrubbed)."""
    fn = _unwrap_registry_tool("get_dataset_info")
    return await asyncio.to_thread(fn, data_path=data_path)


@mcp.tool()
async def get_dataset_conditions(data_path: str) -> Dict[str, Any]:
    """Extract condition/event labels and counts from EEG annotations."""
    fn = _unwrap_registry_tool("get_dataset_conditions")
    return await asyncio.to_thread(fn, data_path=data_path)


@mcp.tool()
async def resolve_frequency_band(
    band_name_or_range: str,
    sfreq: float,
    duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve frequency band name or numeric Hz range against Nyquist & min cycles."""
    fn = _unwrap_registry_tool("resolve_frequency_band")
    return fn(
        band_name_or_range=band_name_or_range,
        sfreq=sfreq,
        duration_seconds=duration_seconds,
    )


@mcp.tool()
async def resolve_channel_selection(
    requested_channels_or_region: str,
    available_channels: List[str],
) -> Dict[str, Any]:
    """Match requested channels or 10-20 anatomical region against dataset channels."""
    fn = _unwrap_registry_tool("resolve_channel_selection")
    return fn(
        requested_channels_or_region=requested_channels_or_region,
        available_channels=available_channels,
    )


@mcp.tool()
async def generate_dataset_overview_plot(data_path: str, run_id: str = "default") -> Dict[str, Any]:
    """Generate diagnostic pre-analysis overview plot (time snippet + Welch PSD)."""
    fn = _unwrap_registry_tool("generate_dataset_overview_plot")
    async with _PLOT_LOCK:
        return await asyncio.to_thread(fn, data_path=data_path, run_id=run_id)


# ---------------------------------------------------------------------------
# 2. Gate 1 Approval Ledger (per run_id)
# ---------------------------------------------------------------------------


@mcp.tool()
async def record_gate_1_approval(
    run_id: str,
    plan: Dict[str, Any],
    parameter_manifest: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Record explicit human Gate 1 approval for run_id before Data Prep can run."""
    if not run_id or not str(run_id).strip():
        return {"ok": False, "error": "INVALID_RUN_ID: run_id is required."}
    if not isinstance(plan, dict) or not plan:
        return {"ok": False, "error": "INVALID_PLAN: non-empty plan dict is required."}
    async with _get_run_lock(run_id):
        _APPROVED_RUNS[run_id] = {
            "run_id": run_id,
            "plan": plan,
            "parameter_manifest": list(parameter_manifest or []),
        }
    return {"ok": True, "run_id": run_id, "gate_1_approved": True}


# ---------------------------------------------------------------------------
# 3. Four stage-level execution tools (ProcessPoolExecutor 'spawn')
# ---------------------------------------------------------------------------


@mcp.tool()
async def run_data_prep(
    run_id: str,
    raw_data_path: str,
    plan: Optional[Dict[str, Any]] = None,
    parameter_manifest: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
    plot_dir: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Run Node 2 (Data Prep) in an isolated worker; refuses unapproved run_ids."""
    approved = _APPROVED_RUNS.get(run_id)
    if approved is None:
        return {
            "run_id": run_id,
            "success": False,
            "error": (
                f"GATE_1_NOT_APPROVED: run_id '{run_id}' has no recorded Gate 1 approval "
                "in the MCP server ledger."
            ),
        }
    async with _get_run_lock(run_id):
        return await _run_with_heartbeats(
            ctx,
            worker_data_prep,
            run_id=run_id,
            raw_data_path=raw_data_path,
            plan_dict=plan or approved["plan"],
            manifest_list=parameter_manifest if parameter_manifest is not None else approved["parameter_manifest"],
            gate_1_approved=True,
            preflight_confirmed=True,
            output_dir=output_dir,
            plot_dir=plot_dir,
        )


@mcp.tool()
async def run_connectivity(
    run_id: str,
    preprocessed_data_path: str,
    plan: Dict[str, Any],
    parameter_manifest: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Run Node 3 (Connectivity Analysis) in an isolated worker process."""
    async with _get_run_lock(run_id):
        return await _run_with_heartbeats(
            ctx,
            worker_connectivity,
            run_id=run_id,
            preprocessed_data_path=preprocessed_data_path,
            plan_dict=plan,
            manifest_list=parameter_manifest or [],
            output_dir=output_dir,
        )


@mcp.tool()
async def run_evaluator_checks(
    run_id: str,
    state_dict: Dict[str, Any],
    use_llm: bool = False,
    output_dir: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Run Node 4 (Multimodal Evaluator) in an isolated worker with bounded LLM concurrency."""
    async with _get_run_lock(run_id):
        if use_llm:
            async with _LLM_SEMAPHORE:
                return await _run_with_heartbeats(
                    ctx,
                    worker_evaluator_checks,
                    run_id=run_id,
                    state_dict=state_dict,
                    use_llm=True,
                    output_dir=output_dir,
                )
        return await _run_with_heartbeats(
            ctx,
            worker_evaluator_checks,
            run_id=run_id,
            state_dict=state_dict,
            use_llm=False,
            output_dir=output_dir,
        )


@mcp.tool()
async def run_synthesis(
    run_id: str,
    state_dict: Dict[str, Any],
    use_llm: bool = False,
    output_dir: Optional[str] = None,
    embed_base64_images: bool = False,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Run Node 5 (Final Synthesis + Gate 2 Verification) in an isolated worker."""
    async with _get_run_lock(run_id):
        if use_llm:
            async with _LLM_SEMAPHORE:
                return await _run_with_heartbeats(
                    ctx,
                    worker_synthesis,
                    run_id=run_id,
                    state_dict=state_dict,
                    use_llm=True,
                    output_dir=output_dir,
                    embed_base64_images=embed_base64_images,
                )
        return await _run_with_heartbeats(
            ctx,
            worker_synthesis,
            run_id=run_id,
            state_dict=state_dict,
            use_llm=False,
            output_dir=output_dir,
            embed_base64_images=embed_base64_images,
        )


# ---------------------------------------------------------------------------
# 4. Two read-only Post-Run Follow-Up context tools (no DB access)
# ---------------------------------------------------------------------------


def _load_offload_registry(run_id: str, output_dir: str = "outputs") -> Dict[str, str]:
    from fc_pipeline.agentic.followup.context_window import PostRunContextWindowManager
    from fc_pipeline.pipeline.run_context import load_run_context, run_context_path

    ctx_path = run_context_path(None, run_id, fallback_dir=output_dir)
    if not ctx_path.is_file():
        return {}
    run_ctx = load_run_context(ctx_path)
    if not run_ctx:
        return {}
    mgr = PostRunContextWindowManager(run_ctx)
    sel = mgr.select_for_turn("summary")
    return {
        off.full_text_ref: mgr._offload_registry.get(off.full_text_ref, "")
        for off in sel.offloaded_context
    }


@mcp.tool()
async def read_offloaded_context(
    run_id: str,
    key: str,
    output_dir: str = "outputs",
) -> str:
    """Read an offloaded run-context section from run_context_<run_id>.json (read-only)."""
    reg = _load_offload_registry(run_id, output_dir=output_dir)
    if not reg:
        return f"No run context found for run_id '{run_id}'."
    body = reg.get(str(key).strip())
    if body is None:
        return f"Key '{key}' not found. Available keys: {', '.join(sorted(reg.keys()))}"
    return body


@mcp.tool()
async def search_run_context(
    run_id: str,
    pattern: str,
    output_dir: str = "outputs",
) -> str:
    """Search offloaded run-context lines for run_id matching pattern (read-only)."""
    reg = _load_offload_registry(run_id, output_dir=output_dir)
    if not reg:
        return f"No run context found for run_id '{run_id}'."
    needle = str(pattern).lower()
    hits: List[str] = []
    for k, body in reg.items():
        for line_no, line in enumerate(body.splitlines(), start=1):
            if needle in line.lower():
                hits.append(f"- {k}:{line_no}: {line.strip()[:200]}")
    return "\n".join(hits[:40]) if hits else f"No matches for '{pattern}' in run_id '{run_id}'."

