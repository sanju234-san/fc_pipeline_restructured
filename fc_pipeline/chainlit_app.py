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
from fc_pipeline.agentic.supervisor.query_transformer import transform_query


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


# Categories where direct edits MUST route through Supervisor re-validation
# (the 3 mandatory scientific axes + metric selection feeding structural invariants).
SCIENTIFIC_EDIT_CATEGORIES = {"scientific_axis", "metric_selection"}


def format_manifest_markdown(manifest: Optional[List[ParameterManifestEntry]]) -> str:
    if not manifest:
        return "_No parameter manifest generated (guardrail halted with clarification question)._\n"
    lines = [
        "| Parameter | Category | Proposed Value | Confidence | Needs Human Input | Risk Tier | Approved Value |",
        "|:---|:---|:---|:---:|:---:|:---:|:---|",
    ]
    for entry in manifest:
        conf_str = f"{entry.confidence:.2f}" if entry.confidence is not None else "—"
        human_str = "⚠️ **YES**" if entry.needs_human_input else "No"
        risk_str = (
            f"**{entry.risk_tier.upper()}**" if entry.risk_tier == "elevated" else entry.risk_tier
        )
        val_str = str(entry.proposed_value).replace("|", "\\|")
        approved_str = mask_text(entry.human_approved_value) if entry.human_approved_value else "—"
        lines.append(
            f"| `{entry.name}` | {entry.category} | {val_str} | {conf_str} | {human_str} | {risk_str} | {approved_str} |"
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
        "awaiting_transformer_clarification": False,
        "base_run_id": None,
        "gate_1_revision": 0,
        "gate_1_approved": False,
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


def _run_pipeline_sync(
    user_request: str,
    run_id: str,
    tags: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    pipeline_app = cl.user_session.get("graph")
    initial_state = _build_graph_state(user_request, run_id)

    is_real_data = cl.user_session.get("is_real_data", False)
    data_path = cl.user_session.get("data_path")

    try:
        active_run = mlflow_tracker.start_run(run_id, tags=tags)
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
    awaiting_transformer_clarification = cl.user_session.get("awaiting_transformer_clarification", False)

    user_text = message.content.strip()
    user_text_lc = user_text.lower().lstrip()

    is_explicit_new = (
        user_text_lc.startswith("new query:")
        or user_text_lc.startswith("new topic:")
        or user_text_lc.startswith("reset")
        or user_text_lc == "/reset"
    )

    # --- Handle transformer-clarification confirmation flow ---
    if awaiting_transformer_clarification:
        # User is responding to a contradiction surfaced by the query transformer.
        # Treat their reply as a clarification that gets appended, then re-run
        # the transformer on the updated accumulated string.
        accumulated_query = cl.user_session.get("accumulated_query", "")
        accumulated_query = f"{accumulated_query}\n[User clarification reply]: {user_text}"
        cl.user_session.set("accumulated_query", accumulated_query)
        cl.user_session.set("awaiting_transformer_clarification", False)
        # Fall through to the transformer call below (do not return here)

    elif original_query is None or is_explicit_new:
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
        cl.user_session.set("base_run_id", None)
        cl.user_session.set("gate_1_revision", 0)
        cl.user_session.set("gate_1_approved", False)
    elif awaiting_clarification:
        accumulated_query = f"{accumulated_query}\n[User clarification reply]: {user_text}"
        cl.user_session.set("accumulated_query", accumulated_query)
        cl.user_session.set("awaiting_clarification", False)
    else:
        accumulated_query = f"{accumulated_query}\n[User follow-up refinement]: {user_text}"
        cl.user_session.set("accumulated_query", accumulated_query)

    run_id = f"chainlit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter:03d}"

    # --- Query Transformer pre-processing step ---
    transformer_msg = cl.Message(content="🔍 Analyzing your request…")
    await transformer_msg.send()

    try:
        qt_result = await cl.make_async(transform_query)(
            accumulated_query, user_text
        )
    except Exception as e:
        # Transformer failure — degrade gracefully, pass raw query to Supervisor
        qt_result = None

    if qt_result is not None:
        # --- Out-of-scope ---
        if qt_result.is_out_of_scope:
            reason = qt_result.condensed
            transformer_msg.content = (
                f"This pipeline handles EEG functional connectivity analysis "
                f"— I can't help with {mask_text(reason)}, but I'm glad to help "
                f"with frequency bands, channels, conditions, or connectivity metrics."
            )
            await transformer_msg.update()
            return

        # --- Contradiction detected (blocked or with correction language) ---
        if qt_result.has_contradiction:
            clarification_text = qt_result.clarification
            if qt_result.has_clarification and clarification_text.lower().strip() != "none":
                cl.user_session.set("awaiting_transformer_clarification", True)
                body = [
                    "## ⚠️ Possible Contradiction Detected",
                    "",
                    f"> **Contradiction**: {mask_text(qt_result.contradiction)}",
                    "",
                    mask_text(clarification_text),
                    "",
                    "---",
                    "_Reply with your clarification, or type `new query: …` to start fresh._",
                ]
                transformer_msg.content = "\n".join(body)
                await transformer_msg.update()
                return

        # --- Clean passthrough or fallback ---
        condensed = qt_result.condensed
        transformer_msg.content = f"Understood your request as: *'{mask_text(condensed)}'*"
        await transformer_msg.update()
        # Use the condensed query for the Supervisor
        effective_query = condensed
    else:
        # Transformer failed entirely — pass raw accumulated query
        transformer_msg.content = "🔄 Proceeding with your request…"
        await transformer_msg.update()
        effective_query = accumulated_query

    # Set base_run_id if starting a fresh sequence
    if cl.user_session.get("base_run_id") is None:
        cl.user_session.set("base_run_id", run_id)

    # --- Invoke Supervisor ReAct loop ---
    msg = cl.Message(content="🔄 Invoking Supervisor Agent…")
    await msg.send()

    try:
        output_state, executed_nodes = await cl.make_async(_run_pipeline_sync)(
            effective_query,
            run_id,
            tags={"scenario_id": run_id, "parent_scenario_id": run_id, "revision": 0},
        )
    except Exception as e:
        err = f"**Pipeline execution failed**: `{type(e).__name__}: {e}`"
        msg.content = mask_text(err)
        await msg.update()
        return

    await _handle_pipeline_output(
        output_state,
        executed_nodes,
        accumulated_query,
        run_id,
        msg,
    )


def _reset_conversation_state():
    """Reset conversational accumulation and gate state so next message starts clean."""
    cl.user_session.set("original_query", None)
    cl.user_session.set("accumulated_query", None)
    cl.user_session.set("base_run_id", None)
    cl.user_session.set("gate_1_revision", 0)


def _snapshot_overrides(
    manifest: Optional[List[ParameterManifestEntry]],
) -> Dict[str, str]:
    """Capture {entry.name: human_approved_value} for entries with overrides.

    Used to preserve advisory edits before a scientific-axis edit triggers a
    Supervisor re-run that produces a brand-new manifest.
    """
    overrides: Dict[str, str] = {}
    if manifest:
        for entry in manifest:
            if entry.human_approved_value is not None:
                overrides[entry.name] = entry.human_approved_value
    return overrides


def _reapply_overrides(
    manifest: Optional[List[ParameterManifestEntry]],
    overrides: Dict[str, str],
) -> None:
    """Re-apply previously snapshotted human_approved_value overrides onto a
    fresh manifest produced by a Supervisor re-run.

    Only re-applies if the entry still exists and is still flagged
    (needs_human_input=True).  Entries that disappeared or are no longer
    flagged are silently dropped — the Supervisor re-run may have resolved
    the issue that prompted the original flag.
    """
    if not manifest or not overrides:
        return
    for entry in manifest:
        if entry.name in overrides and entry.needs_human_input:
            entry.human_approved_value = overrides[entry.name]


def _build_gate_1_card(
    run_id: str,
    summary_text: str,
    plan: Any,
    manifest: Optional[List[ParameterManifestEntry]],
    routed_node: str,
) -> str:
    """Build the Gate 1 markdown card body (reusable across initial render and edit re-renders)."""
    flagged = [
        e for e in (manifest or [])
        if getattr(e, "needs_human_input", False) or getattr(e, "risk_tier", "low") == "elevated"
    ]
    attention_lines: List[str] = []
    if flagged:
        attention_lines.append("> ⚠️ **Attention Required (Elevated Risk / Human Input Needed)**:")
        for fe in flagged:
            cat = f" ({fe.category})" if getattr(fe, "category", None) else ""
            risk = f" [Risk: **{fe.risk_tier.upper()}**]" if getattr(fe, "risk_tier", None) else ""
            attention_lines.append(f"> - **`{fe.name}`**{cat}: {mask_text(str(fe.proposed_value))}{risk}")
        attention_lines.append("")

    body = [
        "## 📋 Gate 1 — Plan & Parameter Preflight Review",
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
    ]
    if attention_lines:
        body.extend(attention_lines)

    body.extend([
        "---",
        f"_Routed node_: `{routed_node}`",
    ])
    return "\n".join(body)


def _build_gate_1_actions(
    manifest: Optional[List[ParameterManifestEntry]],
) -> List[cl.Action]:
    """Build the combined list of per-row Edit buttons + Approve/Request Changes/Reject."""
    actions: List[cl.Action] = []

    # Per-row edit buttons for needs_human_input=True entries
    editable = [
        e for e in (manifest or [])
        if getattr(e, "needs_human_input", False)
    ]
    for entry in editable:
        actions.append(
            cl.Action(
                name=f"edit_{entry.name}",
                payload={"value": f"edit_{entry.name}", "entry_name": entry.name},
                label=f"✏️ Edit: {entry.name}",
                description=f"Edit the value for {entry.name} ({entry.category})",
            )
        )

    # The 3 standard gate actions
    actions.extend([
        cl.Action(
            name="approve",
            payload={"value": "approve"},
            label="✅ Approve",
            description="Approve manifest as-is and proceed to Node 2 (Data Preparation)",
        ),
        cl.Action(
            name="request_changes",
            payload={"value": "request_changes"},
            label="🔄 Request Changes",
            description="Specify parameter adjustments and re-run Supervisor",
        ),
        cl.Action(
            name="reject",
            payload={"value": "reject"},
            label="🛑 Reject",
            description="Abort this pipeline run",
        ),
    ])
    return actions


async def _handle_pipeline_output(
    output_state: Dict[str, Any],
    executed_nodes: List[str],
    accumulated_query: str,
    run_id: str,
    msg: cl.Message,
):
    """Render pipeline outputs and manage Gate 1 HITL action controls."""
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

        # Store the live manifest in session so edits persist across the
        # edit→re-render loop within a single Gate 1 turn.
        cl.user_session.set("gate_1_manifest", manifest)

        # --- Gate 1 HITL loop (edit → re-render → action) ---
        # Uses a while loop so multiple sequential edits stay in the same
        # turn without recursive calls into _handle_pipeline_output.
        while True:
            # (Re-)build and render the Gate 1 card
            msg.content = _build_gate_1_card(run_id, summary_text, plan, manifest, routed_node)
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

            # Build combined action list (edit buttons + Approve/Request Changes/Reject)
            actions = _build_gate_1_actions(manifest)

            action_res = await cl.AskActionMessage(
                content="**Decision required**: Review the proposed parameters. Edit flagged entries or choose an action:",
                actions=actions,
                timeout=300,
            ).send()

            if action_res is None:
                # Strictly aborted on timeout or session drop — never auto-approve
                _reset_conversation_state()
                cl.user_session.set("gate_1_approved", False)
                await cl.Message(
                    content="⚠️ **Gate 1 timed out after 5 minutes with no response.** Pipeline execution aborted. Type a new query or `new query: …` to start fresh."
                ).send()
                return

            chosen_action = (
                action_res.get("name")
                if isinstance(action_res, dict)
                else getattr(action_res, "name", None)
            )

            # --- Approve ---
            if chosen_action == "approve":
                # Re-log the final manifest (with any human_approved_value edits) to MLflow
                try:
                    mlflow_tracker.log_manifest(manifest)
                except Exception:
                    pass
                _reset_conversation_state()
                cl.user_session.set("gate_1_approved", True)
                await cl.Message(
                    content=(
                        "### ✅ Gate 1 Approved\n\n"
                        "Manifest validated and locked. Ready to proceed to **Node 2 (Data Preparation)**.\n\n"
                        "*(Note: Nodes 2–5 are currently scaffolded stubs.)*"
                    )
                ).send()
                return

            # --- Reject ---
            elif chosen_action == "reject":
                _reset_conversation_state()
                cl.user_session.set("gate_1_approved", False)
                await cl.Message(
                    content=(
                        "### 🛑 Gate 1 Rejected\n\n"
                        "Pipeline run aborted without executing downstream nodes. "
                        "Type a new query or `new query: …` to start fresh."
                    )
                ).send()
                return

            # --- Per-row Edit ---
            elif chosen_action and chosen_action.startswith("edit_"):
                entry_name = (
                    action_res.get("payload", {}).get("entry_name")
                    if isinstance(action_res, dict)
                    else getattr(action_res, "payload", {}).get("entry_name")
                ) or chosen_action[len("edit_"):]

                # Find the target entry in the manifest
                target_entry = None
                for e in (manifest or []):
                    if e.name == entry_name:
                        target_entry = e
                        break

                if target_entry is None:
                    await cl.Message(
                        content=f"⚠️ Entry `{mask_text(entry_name)}` not found in manifest."
                    ).send()
                    continue  # Re-render and re-prompt

                # --- Scientific-axis guard ---
                if target_entry.category in SCIENTIFIC_EDIT_CATEGORIES:
                    # Scientific axes MUST go through Supervisor re-validation.
                    # Snapshot existing overrides so they survive the re-run.
                    prior_overrides = _snapshot_overrides(manifest)

                    edit_reply = await cl.AskUserMessage(
                        content=(
                            f"**`{mask_text(entry_name)}`** is a `{target_entry.category}` parameter — "
                            f"changes require Supervisor re-validation.\n\n"
                            f"Enter the new value for `{mask_text(entry_name)}` "
                            f"(current: *{mask_text(str(target_entry.proposed_value))}*):"
                        ),
                        timeout=300,
                    ).send()

                    if edit_reply is None:
                        _reset_conversation_state()
                        cl.user_session.set("gate_1_approved", False)
                        await cl.Message(
                            content="⚠️ **Edit timed out after 5 minutes with no response.** Pipeline execution aborted. Type a new query or `new query: …` to start fresh."
                        ).send()
                        return

                    edit_value = (
                        edit_reply.get("output", "").strip()
                        if isinstance(edit_reply, dict)
                        else getattr(edit_reply, "output", "").strip()
                    )
                    if not edit_value:
                        await cl.Message(content="⚠️ No value provided — edit cancelled.").send()
                        continue  # Re-render

                    # Route through Request Changes path (skip transformer)
                    current_accumulated = cl.user_session.get("accumulated_query", "")
                    new_accumulated = f"{current_accumulated}\n[User Gate 1 Change Request]: Change {entry_name} to {edit_value}"
                    cl.user_session.set("accumulated_query", new_accumulated)

                    base_run_id = cl.user_session.get("base_run_id") or run_id
                    cl.user_session.set("base_run_id", base_run_id)
                    rev_counter = cl.user_session.get("gate_1_revision", 0) + 1
                    cl.user_session.set("gate_1_revision", rev_counter)
                    next_run_id = f"{base_run_id}_rev{rev_counter}"

                    re_msg = cl.Message(
                        content=f"🔄 `{mask_text(entry_name)}` requires Supervisor re-validation — re-invoking with: *'{mask_text(edit_value)}'*…"
                    )
                    await re_msg.send()

                    try:
                        new_state, new_nodes = await cl.make_async(_run_pipeline_sync)(
                            new_accumulated,
                            next_run_id,
                            tags={
                                "scenario_id": next_run_id,
                                "parent_scenario_id": base_run_id,
                                "revision": rev_counter,
                            },
                        )
                    except Exception as e:
                        err = f"**Pipeline execution failed during re-run**: `{type(e).__name__}: {e}`"
                        await cl.Message(content=mask_text(err)).send()
                        return

                    # Re-apply prior advisory overrides onto the fresh manifest
                    new_manifest = new_state.get("parameter_manifest")
                    if prior_overrides and new_manifest:
                        _reapply_overrides(new_manifest, prior_overrides)

                    # Re-enter _handle_pipeline_output for the new result
                    # (may route to gate_1_review again, or clarification_pause, etc.)
                    await _handle_pipeline_output(
                        new_state,
                        new_nodes,
                        new_accumulated,
                        next_run_id,
                        re_msg,
                    )
                    return

                # --- Advisory / informational / engineering_threshold: direct edit ---
                edit_reply = await cl.AskUserMessage(
                    content=(
                        f"Enter new value for `{mask_text(entry_name)}` "
                        f"(current: *{mask_text(str(target_entry.proposed_value))}*):"
                    ),
                    timeout=300,
                ).send()

                if edit_reply is None:
                    _reset_conversation_state()
                    cl.user_session.set("gate_1_approved", False)
                    await cl.Message(
                        content="⚠️ **Edit timed out after 5 minutes with no response.** Pipeline execution aborted. Type a new query or `new query: …` to start fresh."
                    ).send()
                    return

                edit_value = (
                    edit_reply.get("output", "").strip()
                    if isinstance(edit_reply, dict)
                    else getattr(edit_reply, "output", "").strip()
                )
                if not edit_value:
                    await cl.Message(content="⚠️ No value provided — edit cancelled.").send()
                    continue  # Re-render

                # Write directly to human_approved_value — no Supervisor round-trip
                target_entry.human_approved_value = edit_value
                await cl.Message(
                    content=f"✅ `{mask_text(entry_name)}` approved value set to: *{mask_text(edit_value)}*"
                ).send()
                continue  # Loop back to re-render Gate 1 card with updated manifest

            # --- Request Changes (full re-run) ---
            elif chosen_action == "request_changes":
                change_reply = await cl.AskUserMessage(
                    content="**What changes would you like to make to the plan?**",
                    timeout=300,
                ).send()

                if change_reply is None:
                    _reset_conversation_state()
                    cl.user_session.set("gate_1_approved", False)
                    await cl.Message(
                        content="⚠️ **Change request timed out after 5 minutes with no response.** Pipeline execution aborted. Type a new query or `new query: …` to start fresh."
                    ).send()
                    return

                feedback_text = (
                    change_reply.get("output", "").strip()
                    if isinstance(change_reply, dict)
                    else getattr(change_reply, "output", "").strip()
                )
                if not feedback_text:
                    _reset_conversation_state()
                    cl.user_session.set("gate_1_approved", False)
                    await cl.Message(
                        content="⚠️ **No changes specified.** Pipeline execution aborted. Type a new query or `new query: …` to start fresh."
                    ).send()
                    return

                # Snapshot existing overrides before Supervisor re-run
                prior_overrides = _snapshot_overrides(manifest)

                # Append change request directly to accumulated_query
                current_accumulated = cl.user_session.get("accumulated_query", "")
                new_accumulated = f"{current_accumulated}\n[User Gate 1 Change Request]: {feedback_text}"
                cl.user_session.set("accumulated_query", new_accumulated)

                # Suffix run_id and track lineage in MLflow
                base_run_id = cl.user_session.get("base_run_id") or run_id
                cl.user_session.set("base_run_id", base_run_id)
                rev_counter = cl.user_session.get("gate_1_revision", 0) + 1
                cl.user_session.set("gate_1_revision", rev_counter)
                next_run_id = f"{base_run_id}_rev{rev_counter}"

                re_msg = cl.Message(
                    content=f"🔄 Applying requested changes: *'{mask_text(feedback_text)}'* — Re-invoking Supervisor Agent…"
                )
                await re_msg.send()

                # Skip query transformer on this path, feed directly to Supervisor
                try:
                    new_state, new_nodes = await cl.make_async(_run_pipeline_sync)(
                        new_accumulated,
                        next_run_id,
                        tags={
                            "scenario_id": next_run_id,
                            "parent_scenario_id": base_run_id,
                            "revision": rev_counter,
                        },
                    )
                except Exception as e:
                    err = f"**Pipeline execution failed during re-run**: `{type(e).__name__}: {e}`"
                    await cl.Message(content=mask_text(err)).send()
                    return

                # Re-apply prior advisory overrides onto the fresh manifest
                new_manifest = new_state.get("parameter_manifest")
                if prior_overrides and new_manifest:
                    _reapply_overrides(new_manifest, prior_overrides)

                # Re-enter to handle new pipeline output
                await _handle_pipeline_output(
                    new_state,
                    new_nodes,
                    new_accumulated,
                    next_run_id,
                    re_msg,
                )
                return

        # End of while True — unreachable but kept for clarity
        return  # pragma: no cover

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
