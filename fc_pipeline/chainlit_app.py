from __future__ import annotations

import datetime
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chainlit as cl

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import numpy as np
import mne

import mlflow
import mlflow.langchain

mlflow.langchain.autolog()

from fc_pipeline.agentic.supervisor.llm_provider import get_supervisor_llm
from fc_pipeline.agentic.supervisor.tools.dataset_info import get_dataset_info
from fc_pipeline.agentic.supervisor.tools.dataset_conditions import get_dataset_conditions
from fc_pipeline.config.thresholds import SUPERVISOR_CONFIDENCE_THRESHOLD, TAU_PHASE, TAU_ZEROLAG
from fc_pipeline.schemas.state import GraphState
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.pipeline.graph import build_pipeline_graph
from fc_pipeline.observability import mlflow_tracker


def mask_text(text: str) -> str:
    if not text:
        return text
    user = os.getenv("USERNAME") or os.getenv("USER") or ""
    if user:
        text = re.sub(re.escape(user), "[MASKED_USER]", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]:\\Users\\[^\\]+\\", "[LOCAL_ROOT]/", text)
    text = re.sub(r"/(?:home|Users)/[^/]+/", "[LOCAL_ROOT]/", text)
    _IP_RE = r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?(?:/v1)?"
    _URL_RE = r"https?://[^\s]+"
    text = re.sub(
        f"({_IP_RE})|({_URL_RE})",
        lambda m: (
            "http://[CONFIGURED - INTERNAL IP/HOST MASKED]"
            if m.group(1)
            else "http://[MASKED_ENDPOINT]"
        ),
        text,
    )
    text = re.sub(r"gsk_[a-zA-Z0-9]{20,}", "[MASKED - API KEY]", text)
    return text


def mask_endpoint(url: Optional[str]) -> str:
    if not url or not url.strip():
        return "[NOT CONFIGURED]"
    cleaned = url.strip().strip('"').strip("'")
    if "groq.com" in cleaned.lower():
        return "https://api.groq.com/openai/v1"
    return "http://[CONFIGURED - INTERNAL IP/HOST MASKED]"


def mask_model(model_name: Optional[str]) -> str:
    if not model_name or not model_name.strip():
        return "[NOT CONFIGURED]"
    cleaned = model_name.strip().strip('"').strip("'")
    return f"[CONFIGURED: {cleaned}]"


def create_synthetic_eeg_fixture(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    ch_names = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    sfreq = 250.0
    n_samples = int(10.0 * sfreq)
    np.random.seed(42)
    data = np.random.randn(len(ch_names), n_samples) * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    annotations = mne.Annotations(
        onset=[0.0, 5.0],
        duration=[5.0, 5.0],
        description=["rest", "task"],
    )
    raw.set_annotations(annotations)
    raw.save(output_path, overwrite=True, verbose=False)
    return output_path


def resolve_data_source() -> tuple[Path, bool]:
    env_path = os.getenv("PLAYGROUND_REAL_DATA_PATH")
    if env_path and env_path.strip():
        p = Path(env_path.strip().strip('"').strip("'")).resolve()
        if p.exists():
            return p, True
    synthetic_path = Path("outputs") / "synthetic_test_raw.fif"
    create_synthetic_eeg_fixture(synthetic_path)
    return synthetic_path, False


def format_manifest_markdown(manifest: Optional[List[ParameterManifestEntry]]) -> str:
    if not manifest:
        return "_No parameter manifest generated (guardrail halted with clarification question)._\n"
    lines = [
        "| Parameter | Category | Proposed Value | Confidence | Needs Human Input | Risk Tier |",
        "|:---|:---|:---|:---:|:---:|:---:|",
    ]
    for entry in manifest:
        conf_str = f"{entry.confidence:.2f}" if entry.confidence is not None else "—"
        human_str = "⚠️ **YES**" if entry.needs_human_input else "No"
        risk_str = (
            f"**{entry.risk_tier.upper()}**" if entry.risk_tier == "elevated" else entry.risk_tier
        )
        val_str = str(entry.proposed_value).replace("|", "\\|")
        lines.append(
            f"| `{entry.name}` | {entry.category} | {val_str} | {conf_str} | {human_str} | {risk_str} |"
        )
    return "\n".join(lines) + "\n"


def format_plan_markdown(plan: Any) -> str:
    if plan is None:
        return "_No plan generated._\n"
    metric_names = [str(m) for m in plan.metrics]
    lines = [
        "### Analysis Plan",
        "",
        f"- **Frequency Band**: `{plan.freq_band.name}` ({plan.freq_band.fmin}–{plan.freq_band.fmax} Hz)",
        f"- **Condition**: `{plan.condition}`",
        f"- **Channels**: {', '.join(f'`{c}`' for c in plan.channels)}",
        f"- **Metrics**: {', '.join(f'`{m}`' for m in metric_names)}",
        "",
    ]
    return "\n".join(lines)


def summarize_node_result(node_name: str, state: Dict[str, Any]) -> str:
    """Return 2-4 sentence plain-English summary of node outcome.

    Generic: inspects state keys/values, not node name. Safe for any node
    (Supervisor, DataPrep, Connectivity, Evaluator, Synthesis, ...).
    """
    routed = mask_text(str(state.get("_routed_node") or state.get("routed_node") or node_name))
    manifest = state.get("parameter_manifest") or []
    plan = state.get("plan")

    errors = [
        (k, mask_text(str(v)[:200]))
        for k, v in state.items()
        if k.endswith("_error") and v is not None
    ]
    artifact_keys = [
        k for k in state
        if (k.endswith("_path") or k.endswith("_paths")) and state.get(k) is not None
        and k not in ("raw_data_path", "data_path")
    ]
    other_non_null_output = [
        k for k in (
            "bad_channels_dropped", "numerical_summaries",
            "evidence_summary", "evaluation_verdict", "final_report_path",
        )
        if state.get(k) is not None
    ]

    manifest_flagged = 0
    manifest_elevated = 0
    if isinstance(manifest, list):
        for m in manifest:
            try:
                if getattr(m, "needs_human_input", False):
                    manifest_flagged += 1
                if getattr(m, "risk_tier", "low") == "elevated":
                    manifest_elevated += 1
            except Exception:
                pass

    sentences: List[str] = []
    sentences.append(f"Node `{mask_text(node_name)}` completed and routed to `{routed}`.")

    if errors:
        err_names = ", ".join(f"`{k}`" for k, _ in errors[:3])
        sentences.append(f"Errors were emitted in {err_names}; execution halted unexpectedly.")
    else:
        if plan is not None:
            n_ch = len(getattr(plan, "channels", []) or [])
            n_met = len(getattr(plan, "metrics", []) or [])
            sentences.append(
                f"An analysis plan was resolved covering {n_ch} channel(s) and {n_met} metric(s)."
            )
        if manifest:
            sent_parts = [f"A parameter manifest with {len(manifest)} entries was compiled"]
            if manifest_flagged:
                sent_parts.append(f"; {manifest_flagged} flagged for human input")
            if manifest_elevated:
                sent_parts.append(f" ({manifest_elevated} at elevated risk)")
            sent_parts.append(".")
            sentences.append("".join(sent_parts))

    clarif = state.get("clarification_question")
    if clarif:
        sentences.append(
            "A clarification question was emitted; the pipeline is paused for a user reply before resuming."
        )

    produced = []
    if artifact_keys:
        produced.append(f"{len(artifact_keys)} artifact file(s)")
    if other_non_null_output:
        produced.append(f"{len(other_non_null_output)} structured output(s)")
    if produced:
        sentences.append("Produced " + " and ".join(produced) + ".")

    if len(sentences) < 2:
        if plan is None and not manifest and not clarif and not artifact_keys:
            sentences.append(
                "No structured artifacts were written on this pass; review routed node for the expected downstream behaviour."
            )

    return " ".join(sentences[:4])


@cl.on_chat_start
async def on_chat_start():
    data_path, is_real_data = resolve_data_source()

    try:
        _ = get_supervisor_llm()
        llm_ok = True
        endpoint_msg = mask_endpoint(os.getenv("SUPERVISOR_LLM_ENDPOINT"))
        model_msg = mask_model(os.getenv("SUPERVISOR_LLM_MODEL"))
    except Exception as e:
        llm_ok = False
        endpoint_msg = f"ERROR: {e}"
        model_msg = "[NOT CONFIGURED]"

    try:
        info_dict = get_dataset_info.invoke({"data_path": str(data_path)})
        cond_dict = get_dataset_conditions.invoke({"data_path": str(data_path)})
        n_ch = info_dict.get("n_channels", 0)
        sfreq = info_dict.get("sfreq", 0.0)
        dur = info_dict.get("duration_seconds", 0.0)
        sample_ch = info_dict.get("available_channels", [])[:8]
        ch_suffix = f" … (+{n_ch - 8} more)" if n_ch > 8 else ""
        cond_count = cond_dict.get("total_trials", 0)
        cond_labels = list(cond_dict.get("conditions", {}).keys())
    except Exception as e:
        n_ch, sfreq, dur = 0, 0.0, 0.0
        sample_ch, ch_suffix, cond_count, cond_labels = [], "", 0, []

    session_state = {
        "original_query": None,
        "accumulated_query": None,
        "awaiting_clarification": False,
        "data_path": str(data_path),
        "is_real_data": is_real_data,
        "graph": build_pipeline_graph().compile(),
        "counter": 0,
    }
    for k, v in session_state.items():
        cl.user_session.set(k, v)

    data_type = "REAL DATASET" if is_real_data else "SYNTHETIC 10-20 FIXTURE"
    welcome = "\n".join([
        "# EEG FC Pipeline — Supervisor Chat",
        "",
        "## Environment",
        f"- **LLM Model**: {model_msg}",
        f"- **Endpoint**: {endpoint_msg}",
        f"- **Data Source**: `{data_type}`",
        f"- **Channels**: {n_ch} ({sfreq} Hz) — {', '.join(sample_ch)}{ch_suffix}",
        f"- **Duration**: {dur:.1f}s | **Conditions**: {cond_labels} (trials: {cond_count})",
        f"- **Data Path**: `{mask_text(str(data_path))}`",
        "",
        "Send a query to begin. Try:",
        "- `compute PLI and coherence for alpha band on F3, F4 during rest`",
        "- `analyze this EEG` → triggers zero-guessing guardrail (clarification)",
    ])
    if not llm_ok:
        welcome += "\n\n⚠️ **LLM provider is not configured. Please set SUPERVISOR_LLM_ENDPOINT and SUPERVISOR_LLM_MODEL in .env.**"
    await cl.Message(content=welcome).send()


def _build_graph_state(user_request: str, run_id: str) -> GraphState:
    data_path = cl.user_session.get("data_path")
    return {
        "pipeline_error": None,
        "run_id": run_id,
        "raw_data_path": data_path,
        "user_request": user_request,
        "plan": None,
        "parameter_manifest": None,
        "preflight_confirmed": False,
        "clarification_question": None,
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


def _run_pipeline_sync(user_request: str, run_id: str) -> tuple[Dict[str, Any], List[str]]:
    pipeline_app = cl.user_session.get("graph")
    initial_state = _build_graph_state(user_request, run_id)

    is_real_data = cl.user_session.get("is_real_data", False)
    data_path = cl.user_session.get("data_path")

    try:
        active_run = mlflow_tracker.start_run(run_id)
        if active_run and hasattr(active_run, "info"):
            mlflow_tracker.log_params({
                "model_name": os.getenv("SUPERVISOR_LLM_MODEL", ""),
                "endpoint": os.getenv("SUPERVISOR_LLM_ENDPOINT", ""),
                "confidence_threshold": SUPERVISOR_CONFIDENCE_THRESHOLD,
                "tau_phase": TAU_PHASE,
                "tau_zerolag": TAU_ZEROLAG,
                "query": user_request,
                "data_path": str(data_path),
                "is_real_data": is_real_data,
            })
    except Exception:
        pass

    output_state: Dict[str, Any] = dict(initial_state)
    executed_nodes: List[str] = []
    exec_error: Optional[Exception] = None

    try:
        for mode, payload in pipeline_app.stream(
            initial_state,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                executed_nodes.extend(payload.keys())
            elif mode == "values":
                output_state = payload
    except Exception as e:
        exec_error = e
        output_state["pipeline_error"] = str(e)

    plan = output_state.get("plan")
    manifest = output_state.get("parameter_manifest")
    clarif = output_state.get("clarification_question")
    pipe_err = output_state.get("pipeline_error")

    if len(executed_nodes) > 1:
        routed_node = executed_nodes[-1]
    elif executed_nodes:
        routed_node = f"HALTED IN {executed_nodes[0]}"
    elif exec_error:
        routed_node = "N/A (INFRA ERROR)"
    else:
        routed_node = "N/A (NOT STARTED)"

    try:
        mlflow_tracker.log_manifest(manifest)
        mlflow_tracker.log_result(
            routed_node=routed_node,
            plan=plan,
            clarification_question=clarif,
            pipeline_error=pipe_err,
        )
    except Exception:
        pass
    finally:
        try:
            mlflow_tracker.end_run()
        except Exception:
            pass

    output_state["_routed_node"] = routed_node
    output_state["_executed_nodes"] = executed_nodes
    return output_state, executed_nodes


@cl.on_message
async def on_message(message: cl.Message):
    counter = cl.user_session.get("counter", 0) + 1
    cl.user_session.set("counter", counter)

    original_query = cl.user_session.get("original_query")
    accumulated_query = cl.user_session.get("accumulated_query")
    awaiting_clarification = cl.user_session.get("awaiting_clarification", False)

    user_text = message.content.strip()
    user_text_lc = user_text.lower().lstrip()

    is_explicit_new = (
        user_text_lc.startswith("new query:")
        or user_text_lc.startswith("new topic:")
        or user_text_lc.startswith("reset")
        or user_text_lc == "/reset"
    )

    if original_query is None or is_explicit_new:
        if is_explicit_new:
            stripped = re.sub(
                r"^\s*(new\s+(query|topic)\s*:|/reset|reset)\s*",
                "",
                user_text,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            user_text = stripped or user_text
        original_query = user_text
        accumulated_query = user_text
        cl.user_session.set("original_query", original_query)
        cl.user_session.set("accumulated_query", accumulated_query)
    elif awaiting_clarification:
        accumulated_query = f"{accumulated_query}\n[User clarification reply]: {user_text}"
        cl.user_session.set("accumulated_query", accumulated_query)
        cl.user_session.set("awaiting_clarification", False)
    else:
        accumulated_query = f"{accumulated_query}\n[User follow-up refinement]: {user_text}"
        cl.user_session.set("accumulated_query", accumulated_query)

    run_id = f"chainlit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter:03d}"

    msg = cl.Message(content="🔄 Invoking Supervisor Agent…")
    await msg.send()

    try:
        output_state, executed_nodes = await cl.make_async(_run_pipeline_sync)(
            accumulated_query, run_id
        )
    except Exception as e:
        err = f"**Pipeline execution failed**: `{type(e).__name__}: {e}`"
        msg.content = mask_text(err)
        await msg.update()
        return

    routed_node = output_state.get("_routed_node", "")
    plan = output_state.get("plan")
    manifest = output_state.get("parameter_manifest")
    clarif = output_state.get("clarification_question")
    pipe_err = output_state.get("pipeline_error")

    summary_text = mask_text(summarize_node_result("supervisor", output_state))

    if "supervisor_error" in routed_node or pipe_err:
        cl.user_session.set("awaiting_clarification", False)
        body = ["## 🛑 Supervisor Error", "", "### Node Result Summary", summary_text, ""]
        if pipe_err:
            body.append(f"```\n{mask_text(pipe_err)}\n```")
            body.append("")
        body.append(f"_Routed to: `{routed_node}`_")
        body.append(f"_Executed nodes: `{executed_nodes}`_")
        body.append("")
        body.append(
            "_Tip: next message will be treated as a follow-up refinement. "
            "Type `new query: …` to start fresh._"
        )
        msg.content = "\n".join(body)
        await msg.update()
        return

    if "clarification_pause" in routed_node and clarif:
        cl.user_session.set("awaiting_clarification", True)
        body = [
            "## ❓ Clarification Required",
            "",
            "### Node Result Summary",
            summary_text,
            "",
            f"> {mask_text(clarif)}",
            "",
            "Reply with your answer and it will be appended to your request.",
            "",
            "---",
            f"_Current accumulated request:_",
            "```",
            mask_text(accumulated_query),
            "```",
        ]
        msg.content = "\n".join(body)
        await msg.update()
        return

    if "gate_1_review" in routed_node:
        cl.user_session.set("awaiting_clarification", False)
        body = [
            "## ✅ Gate 1 — Plan & Parameter Manifest",
            "",
            f"_Scenario ID_: `{run_id}`",
            "",
            "### Node Result Summary",
            summary_text,
            "",
            format_plan_markdown(plan),
            "### Parameter Manifest",
            "",
            format_manifest_markdown(manifest),
            "",
            "---",
            f"_Routed node_: `{routed_node}`",
            "",
            "_Tip: next message will be treated as a follow-up refinement. "
            "Type `new query: …` to start fresh._",
            "",
            "_Data Prep → Connectivity → Evaluator → Synthesis nodes are not yet implemented._",
        ]
        msg.content = "\n".join(body)
        await msg.update()

        overview_plot = Path("outputs") / "plots" / f"{run_id}_overview.png"
        if overview_plot.exists():
            image = cl.Image(
                path=str(overview_plot),
                name="Dataset Overview",
                display="inline",
            )
            await cl.Message(
                content="### Dataset Overview Plot",
                elements=[image],
            ).send()
        return

    fallback = [
        "## ⚠️ Unknown Outcome",
        "",
        "### Node Result Summary",
        summary_text,
        "",
        f"_Routed node_: `{routed_node}`",
        f"_Executed nodes_: `{executed_nodes}`",
        "",
        f"```\nplan={plan is not None}\nclarification={clarif is not None}\nmanifest={manifest is not None}\npipeline_error={pipe_err}\n```",
        "",
        "_Tip: next message will be treated as a follow-up refinement. Type `new query: …` to start fresh._",
    ]
    cl.user_session.set("awaiting_clarification", False)
    msg.content = "\n".join(fallback)
    await msg.update()
