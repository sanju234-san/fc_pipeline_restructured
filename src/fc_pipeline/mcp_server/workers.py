"""Top-level picklable worker functions for ProcessPoolExecutor ('spawn').

All workers accept plain JSON-serializable dicts/strings, execute the existing
pipeline stages without refactoring them, persist run_context_<run_id>.json
atomically, and clean up Matplotlib figures + Python GC memory on exit.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional


def _cleanup_worker_memory() -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    gc.collect()


def _rehydrate_plan(plan_data: Any) -> Any:
    from fc_pipeline.schemas.plan import AnalysisPlan

    if plan_data is None or isinstance(plan_data, AnalysisPlan):
        return plan_data
    return AnalysisPlan.model_validate(plan_data)


def _rehydrate_manifest(manifest_data: Optional[List[Any]]) -> List[Any]:
    from fc_pipeline.schemas.manifest import ParameterManifestEntry

    out: List[Any] = []
    for item in manifest_data or []:
        if isinstance(item, ParameterManifestEntry):
            out.append(item)
        else:
            out.append(ParameterManifestEntry.model_validate(item))
    return out


def _persist_worker_context(
    state_dict: Dict[str, Any],
    *,
    fallback_dir: Optional[str] = None,
) -> Optional[str]:
    from fc_pipeline.pipeline.run_context import (
        build_run_context_from_state,
        load_run_context,
        run_context_path,
        save_run_context,
    )

    run_id = str(state_dict.get("run_id") or "unknown")
    prep_path = state_dict.get("preprocessed_data_path")
    ctx_path = run_context_path(
        str(prep_path) if prep_path else None,
        run_id,
        fallback_dir=fallback_dir or "outputs",
    )
    existing = load_run_context(ctx_path) or {}
    merged = {**existing, **state_dict}
    ctx = build_run_context_from_state(merged)
    save_run_context(ctx, ctx_path)
    return str(ctx_path)


def worker_data_prep(
    *,
    run_id: str,
    raw_data_path: str,
    plan_dict: Dict[str, Any],
    manifest_list: List[Dict[str, Any]],
    gate_1_approved: bool,
    preflight_confirmed: bool,
    output_dir: Optional[str] = None,
    plot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Isolated worker for Node 2 (Data Preparation)."""
    try:
        from fc_pipeline.deterministic.data_prep.executor import run_data_prep
        from fc_pipeline.deterministic.data_prep.models import DataPrepInput

        plan = _rehydrate_plan(plan_dict)
        manifest = _rehydrate_manifest(manifest_list)
        dp_input = DataPrepInput(
            raw_data_path=raw_data_path,
            plan=plan,
            parameter_manifest=manifest,
            run_id=run_id,
            gate_1_approved=gate_1_approved,
            preflight_confirmed=preflight_confirmed,
        )
        out_p = Path(output_dir) if output_dir else None
        plt_p = Path(plot_dir) if plot_dir else (out_p / "plots" if out_p else None)
        res = run_data_prep(dp_input, output_dir=out_p, plot_dir=plt_p)
        payload = {
            "run_id": run_id,
            "success": res.success,
            "preprocessed_data_path": res.preprocessed_data_path,
            "bad_channels_dropped": res.bad_channels_dropped or [],
            "channel_plot_paths": res.channel_plot_paths or {},
            "data_prep_summary": res.summary.model_dump(mode="json") if res.summary else None,
            "error": res.error,
        }
        if res.success:
            ctx_file = _persist_worker_context(
                {
                    "run_id": run_id,
                    "raw_data_path": raw_data_path,
                    "plan": plan,
                    "parameter_manifest": manifest,
                    "preprocessed_data_path": res.preprocessed_data_path,
                    "bad_channels_dropped": res.bad_channels_dropped or [],
                    "channel_plot_paths": res.channel_plot_paths or {},
                    "data_prep_summary": res.summary,
                },
                fallback_dir=output_dir,
            )
            payload["run_context_path"] = ctx_file
        return payload
    finally:
        _cleanup_worker_memory()


def worker_connectivity(
    *,
    run_id: str,
    preprocessed_data_path: str,
    plan_dict: Dict[str, Any],
    manifest_list: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Isolated worker for Node 3 (Connectivity Analysis)."""
    try:
        from fc_pipeline.deterministic.connectivity.executor import run_connectivity_analysis
        from fc_pipeline.deterministic.connectivity.models import ConnectivityInput

        plan = _rehydrate_plan(plan_dict)
        manifest = _rehydrate_manifest(manifest_list)
        conn_in = ConnectivityInput(
            preprocessed_data_path=preprocessed_data_path,
            plan=plan,
            parameter_manifest=manifest,
            run_id=run_id,
        )
        out_p = Path(output_dir) if output_dir else None
        res = run_connectivity_analysis(conn_in, output_dir=out_p)
        evidence_json = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in (res.evidence_summary or [])
        ]
        payload = {
            "run_id": run_id,
            "success": res.success,
            "metric_csv_paths": res.metric_csv_paths,
            "heatmap_image_paths": res.heatmap_image_paths,
            "network_image_paths": res.network_image_paths,
            "numerical_summaries": res.numerical_summaries,
            "evidence_summary": evidence_json,
            "consolidated_report_path": res.consolidated_report_path,
            "error": res.error,
        }
        if res.success:
            ctx_file = _persist_worker_context(
                {
                    "run_id": run_id,
                    "plan": plan,
                    "parameter_manifest": manifest,
                    "preprocessed_data_path": preprocessed_data_path,
                    "metric_csv_paths": res.metric_csv_paths,
                    "heatmap_image_paths": res.heatmap_image_paths,
                    "network_image_paths": res.network_image_paths,
                    "numerical_summaries": res.numerical_summaries,
                    "evidence_summary": res.evidence_summary,
                    "consolidated_report_path": res.consolidated_report_path,
                },
                fallback_dir=output_dir,
            )
            payload["run_context_path"] = ctx_file
        return payload
    finally:
        _cleanup_worker_memory()


def worker_evaluator_checks(
    *,
    run_id: str,
    state_dict: Dict[str, Any],
    use_llm: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Isolated worker for Node 4 (Multimodal Evaluator)."""
    try:
        from fc_pipeline.agentic.evaluator.agent import run_evaluator_agent

        llm = None
        if use_llm:
            try:
                from fc_pipeline.agentic.supervisor.llm_provider import get_supervisor_llm

                llm = get_supervisor_llm()
            except Exception:
                llm = None

        work_state = dict(state_dict)
        work_state["run_id"] = run_id
        if work_state.get("plan") is not None:
            work_state["plan"] = _rehydrate_plan(work_state["plan"])
        if work_state.get("parameter_manifest") is not None:
            work_state["parameter_manifest"] = _rehydrate_manifest(work_state["parameter_manifest"])

        out_p = Path(output_dir) if output_dir else None
        update = run_evaluator_agent(work_state, llm=llm, output_dir=out_p)
        verdict = update.get("evaluation_verdict")
        verdict_json = verdict.model_dump(mode="json") if hasattr(verdict, "model_dump") else verdict
        ctx_file = _persist_worker_context(
            {**work_state, **update},
            fallback_dir=output_dir,
        )
        return {
            "run_id": run_id,
            "success": not bool(update.get("pipeline_error")),
            "evaluation_verdict": verdict_json,
            "evaluation_summary_csv_path": update.get("evaluation_summary_csv_path"),
            "error": update.get("pipeline_error"),
            "run_context_path": ctx_file,
        }
    finally:
        _cleanup_worker_memory()


def worker_synthesis(
    *,
    run_id: str,
    state_dict: Dict[str, Any],
    use_llm: bool = False,
    output_dir: Optional[str] = None,
    embed_base64_images: bool = False,
) -> Dict[str, Any]:
    """Isolated worker for Node 5 (Final Synthesis + Gate 2 Pre-Publication Verification)."""
    try:
        from fc_pipeline.agentic.synthesis.agent import run_synthesis_agent
        from fc_pipeline.schemas.evaluation import EvaluationVerdict
        from fc_pipeline.schemas.evidence import PairEvidenceSynthesis

        llm = None
        if use_llm:
            try:
                from fc_pipeline.agentic.supervisor.llm_provider import get_supervisor_llm

                llm = get_supervisor_llm()
            except Exception:
                llm = None

        work_state = dict(state_dict)
        work_state["run_id"] = run_id
        if work_state.get("plan") is not None:
            work_state["plan"] = _rehydrate_plan(work_state["plan"])
        if work_state.get("parameter_manifest") is not None:
            work_state["parameter_manifest"] = _rehydrate_manifest(work_state["parameter_manifest"])
        if isinstance(work_state.get("evaluation_verdict"), dict):
            work_state["evaluation_verdict"] = EvaluationVerdict.model_validate(
                work_state["evaluation_verdict"]
            )
        if work_state.get("evidence_summary"):
            work_state["evidence_summary"] = [
                item
                if isinstance(item, PairEvidenceSynthesis)
                else PairEvidenceSynthesis.model_validate(item)
                for item in work_state["evidence_summary"]
            ]

        out_p = Path(output_dir) if output_dir else None
        update = run_synthesis_agent(
            work_state,
            llm=llm,
            output_dir=out_p,
            embed_base64_images=embed_base64_images,
        )
        ctx_file = _persist_worker_context(
            {**work_state, **update},
            fallback_dir=output_dir,
        )
        return {
            "run_id": run_id,
            "success": bool(update.get("final_report_path")) and not bool(update.get("pipeline_error")),
            "final_report_path": update.get("final_report_path"),
            "error": update.get("pipeline_error"),
            "run_context_path": ctx_file,
        }
    finally:
        _cleanup_worker_memory()

