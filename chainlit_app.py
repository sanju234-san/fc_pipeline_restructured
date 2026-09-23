from __future__ import annotations

import datetime
import logging
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import chainlit as cl

logger = logging.getLogger(__name__)


def _trace_span(name: str, span_type: str = "TOOL", inputs: Optional[Dict[str, Any]] = None):
    """Create a child MLflow span; degrades to a no-op if tracing is unavailable.

    Delegates to the shared helper in observability.mlflow_tracker so the (subtle)
    correct-API handling lives in exactly one place. See that helper for why
    `inputs` cannot be passed straight to mlflow.start_span().
    """
    try:
        from fc_pipeline.observability.mlflow_tracker import trace_span as _shared
        return _shared(name=name, span_type=span_type, inputs=inputs)
    except Exception:
        return nullcontext()


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

from langgraph.types import Command
from fc_pipeline.agentic.supervisor.llm_provider import get_supervisor_llm
from fc_pipeline.agentic.supervisor.tools.dataset_info import get_dataset_info
from fc_pipeline.agentic.supervisor.tools.dataset_conditions import get_dataset_conditions
from fc_pipeline.config.thresholds import SUPERVISOR_CONFIDENCE_THRESHOLD, TAU_PHASE, TAU_ZEROLAG
from fc_pipeline.schemas.state import GraphState
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.pipeline.graph import build_pipeline_graph, compile_pipeline_app
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
        val_str = mask_text(str(entry.proposed_value)).replace("|", "\\|")
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
        "graph": compile_pipeline_app(),
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


def _build_graph_state(
    user_request: str,
    run_id: str,
    latest_user_message: Optional[str] = None,
) -> GraphState:
    data_path = cl.user_session.get("data_path")
    return {
        "pipeline_error": None,
        "run_id": run_id,
        "raw_data_path": data_path,
        "user_request": user_request,
        "latest_user_message": latest_user_message or user_request,
        "plan": None,
        "parameter_manifest": None,
        "preflight_confirmed": False,
        "gate_1_approved": False,
        "clarification_question": None,
        "informational_response": None,
        "informational_artifacts": None,
        "decision_context": None,
        "input_rail_cleared": False,
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
    *,
    latest_user_message: Optional[str] = None,
    resume_action: Optional[str] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Runs the graph. With `resume_action`, resumes the paused checkpoint for
    `run_id` via Command(resume={"action": ...}) instead of starting a new run."""
    pipeline_app = cl.user_session.get("graph")
    latest_msg = latest_user_message or cl.user_session.get("latest_user_message") or user_request
    initial_state = _build_graph_state(
        user_request,
        run_id,
        latest_user_message=latest_msg,
    )
    config = {"configurable": {"thread_id": run_id}}

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

    stream_input = Command(resume={"action": resume_action}) if resume_action else initial_state

    try:
        for mode, payload in pipeline_app.stream(
            stream_input,
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for k in payload.keys():
                    if k == "__interrupt__":
                        executed_nodes.append("gate_1_review")
                    else:
                        executed_nodes.append(k)
            elif mode == "values":
                output_state = payload

        # Check snapshot for interrupt state (e.g. paused at gate_1_review)
        snapshot = pipeline_app.get_state(config)
        if snapshot and snapshot.values:
            output_state = dict(snapshot.values)
            if snapshot.next and "gate_1_review" in snapshot.next:
                if "gate_1_review" not in executed_nodes:
                    executed_nodes.append("gate_1_review")
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
@mlflow.trace(name="chainlit_on_message", span_type="CHAIN")
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
    cl.user_session.set("latest_user_message", user_text)

    # Set base_run_id if starting a fresh sequence
    if cl.user_session.get("base_run_id") is None:
        cl.user_session.set("base_run_id", run_id)

    # --- Invoke Pipeline (Query Transformer → Supervisor ReAct loop) ---
    msg = cl.Message(content="🔄 Running Pipeline (Query Transformer → Supervisor)…")
    await msg.send()

    try:
        output_state, executed_nodes = await cl.make_async(_run_pipeline_sync)(
            accumulated_query,
            run_id,
            tags={"scenario_id": run_id, "parent_scenario_id": run_id, "revision": 0},
        )
    except Exception as e:
        err = f"**Pipeline execution failed**: `{type(e).__name__}: {e}`"
        msg.content = mask_text(err)
        await msg.update()
        return

    # Update condensed_query in session if the pipeline transformed it
    if output_state.get("user_request"):
        cl.user_session.set("condensed_query", output_state.get("user_request"))

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
    cl.user_session.set("condensed_query", None)
    cl.user_session.set("gate_1_approved", False)
    cl.user_session.set("preflight_confirmed", False)


def _prepare_revision_run(current_run_id: str) -> tuple[str, str, int, Dict[str, Any]]:
    """Compute the next revision's run_id + MLflow tags + update session counters.

    Shared by (1) scientific-axis per-row edit re-validation, and
    (2) standalone Request Changes button re-validation. Ensures lineage
    (base_run_id / revision counter / suffixed run_id) cannot drift between
    the two code paths.

    Returns: (base_run_id, next_run_id, rev_counter, mlflow_tags_dict)
    """
    base_run_id = cl.user_session.get("base_run_id") or current_run_id
    cl.user_session.set("base_run_id", base_run_id)
    rev_counter = cl.user_session.get("gate_1_revision", 0) + 1
    cl.user_session.set("gate_1_revision", rev_counter)
    cl.user_session.set("gate_1_approved", False)
    cl.user_session.set("preflight_confirmed", False)
    next_run_id = f"{base_run_id}_rev{rev_counter}"
    tags = {
        "scenario_id": next_run_id,
        "parent_scenario_id": base_run_id,
        "revision": rev_counter,
    }
    return base_run_id, next_run_id, rev_counter, tags


def _build_revision_query(change_feedback: str) -> str:
    """Build the re-validation query for a revision run.

    Skip query transformer per spec (intent already unambiguous). Start from
    session.stored condensed_query (if present) so re-validation uses the
    same condensed intent as the original Supervisor pass, not a potentially
    large accumulated_query string. Fall back to original_query if
    condensed_query is absent (e.g. original transformer failed entirely).

    KNOWN EDGE CASE (condensed_query staleness across chained revisions):
    condensed_query is written once per plain on_message() call, from the
    query transformer's condensed output. Inside the Gate 1 HITL loop, any
    revision (standalone Request Changes OR scientific-axis per-row edit)
    skips the transformer and therefore does NOT re-condense the resulting
    updated intent. If the user chains revision N → reaches Gate 1 again
    → then triggers revision N+1 within the same turn, the condensed_query
    prefix on revision N+1 will still reflect the pre-revision-N condensed
    intent. The revision-N feedback is preserved as the *previous*
    "[User Gate 1 Change Request]" line ONLY IF it was captured into the
    user_request string that the Supervisor re-processed. However: any
    intent change introduced *only* via revision-N feedback and NOT
    re-incorporated back into condensed_query via a plain on_message call
    will appear in revision N+1's query string ONLY if it's part of the
    accumulated state the Supervisor re-emits in its plan. This is the
    explicit contract of the skip-transformer design — revising a revision
    loses the condensation property between hops, but never loses the
    raw feedback text (it's in the change_feedback arg passed here). If
    chained revisions become a common workflow, the correct fix is to
    re-enable the query transformer on revision runs, overriding the
    current "skip transformer per spec" rule for N >= 2 only.
    """
    condensed = cl.user_session.get("condensed_query")
    if not condensed:
        condensed = cl.user_session.get("original_query") or ""
    if condensed:
        return f"{condensed}\n[User Gate 1 Change Request]: {change_feedback}"
    return f"[User Gate 1 Change Request]: {change_feedback}"


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


def _build_rail_card(run_id: str, ctx: Dict[str, Any], flagged_text: Optional[str]) -> str:
    """Markdown card for a NeMo-rail-triggered pause (decision_context kind input_rail/output_rail)."""
    details = ctx.get("details") or {}
    lines = [
        f"## 🛡️ {ctx.get('title') or 'Guardrail Review'}",
        "",
        f"_Scenario ID_: `{run_id}`",
        "",
        f"> {mask_text(str(ctx.get('reason') or ''))}",
        "",
    ]
    if details.get("rail"):
        lines.append(f"_Rail_: `{details['rail']}`")
        lines.append("")
    if flagged_text:
        lines.extend(["**Flagged text:**", "```text", mask_text(flagged_text)[:2000], "```", ""])
    lines.extend([
        "---",
        "_Nothing proceeds until you decide. No response within 5 minutes aborts the run._",
    ])
    return "\n".join(lines)


def _build_rail_actions(ctx: Dict[str, Any]) -> List[cl.Action]:
    """Same Approve / Request Changes / Reject vocabulary as Gate 1 (no new action types)."""
    is_input = ctx.get("kind") == "input_rail"
    actions = [
        cl.Action(
            name="approve",
            payload={"value": "approve"},
            label="✅ Proceed anyway" if is_input else "✅ Show anyway",
            description=(
                "Override the guardrail and continue to the Query Transformer"
                if is_input
                else "Release the flagged response"
            ),
        ),
    ]
    if "request_changes" in (ctx.get("allowed_responses") or []):
        actions.append(
            cl.Action(
                name="request_changes",
                payload={"value": "request_changes"},
                label="✏️ Rephrase query",
                description="Enter a new request instead",
            )
        )
    actions.append(
        cl.Action(
            name="reject",
            payload={"value": "reject"},
            label="🛑 Block" if is_input else "🛑 Withhold",
            description="Stop here without running the pipeline" if is_input else "Do not show this response",
        )
    )
    return actions


async def _handle_rail_decision(
    output_state: Dict[str, Any],
    run_id: str,
    msg: cl.Message,
):
    """Presents a NeMo-rail ask_human pause through the existing Gate 1 surface.

    Uses the same AskActionMessage + Command(resume={"action": ...}) mechanics as
    the manifest review; timeouts abort (never auto-approve).
    """
    ctx = output_state.get("decision_context") or {}
    kind = ctx.get("kind")
    flagged = (ctx.get("details") or {}).get("flagged_text")

    msg.content = _build_rail_card(run_id, ctx, flagged)
    await msg.update()

    action_res = await cl.AskActionMessage(
        content="**Decision required**: the guardrail flagged this. Choose an action:",
        actions=_build_rail_actions(ctx),
        timeout=300,
    ).send()

    if action_res is None:
        _reset_conversation_state()
        await cl.Message(
            content="⚠️ **Guardrail review timed out after 5 minutes with no response.** Nothing was run. Type a new query or `new query: …` to start fresh."
        ).send()
        return

    chosen = (
        action_res.get("name")
        if isinstance(action_res, dict)
        else getattr(action_res, "name", None)
    )
    accumulated = cl.user_session.get("accumulated_query") or ""
    resume_tags = {"scenario_id": run_id, "parent_scenario_id": run_id, "revision": 0}

    with _trace_span(
        name=f"guardrail_decision_{chosen}",
        span_type="TOOL",
        inputs={"run_id": run_id, "kind": kind, "action": chosen},
    ):
        # --- Approve: override the rail and continue ---
        if chosen == "approve":
            try:
                new_state, new_nodes = await cl.make_async(_run_pipeline_sync)(
                    accumulated, run_id, tags=resume_tags, resume_action="approve"
                )
            except Exception as e:
                await cl.Message(
                    content=mask_text(f"**Pipeline execution failed on resume**: `{type(e).__name__}: {e}`")
                ).send()
                return
            re_msg = cl.Message(content="✅ Guardrail overridden — continuing…")
            await re_msg.send()
            if kind == "output_rail":
                # Released response: render through the normal informational branch
                new_state["_routed_node"] = "informational_complete"
                new_nodes = ["informational_complete"]
            await _handle_pipeline_output(new_state, new_nodes, accumulated, run_id, re_msg)
            return

        # --- Request Changes (input rail only): rephrase and start a fresh run ---
        if chosen == "request_changes":
            reply = await cl.AskUserMessage(
                content="**Enter a rephrased request:**",
                timeout=300,
            ).send()
            new_text = (
                (reply.get("output", "") if isinstance(reply, dict) else getattr(reply, "output", "")) or ""
            ).strip() if reply is not None else ""
            if not new_text:
                _reset_conversation_state()
                await cl.Message(
                    content="⚠️ **No rephrased request provided.** Nothing was run. Type a new query or `new query: …` to start fresh."
                ).send()
                return
            _, next_run_id, _, rev_tags = _prepare_revision_run(run_id)
            cl.user_session.set("original_query", new_text)
            cl.user_session.set("accumulated_query", new_text)
            cl.user_session.set("latest_user_message", new_text)
            re_msg = cl.Message(content=f"🔄 Re-running with: *'{mask_text(new_text)}'*…")
            await re_msg.send()
            try:
                new_state, new_nodes = await cl.make_async(_run_pipeline_sync)(
                    new_text, next_run_id, tags=rev_tags, latest_user_message=new_text
                )
            except Exception as e:
                await cl.Message(
                    content=mask_text(f"**Pipeline execution failed during re-run**: `{type(e).__name__}: {e}`")
                ).send()
                return
            await _handle_pipeline_output(new_state, new_nodes, new_text, next_run_id, re_msg)
            return

        # --- Reject ---
        try:
            await cl.make_async(_run_pipeline_sync)(
                accumulated, run_id, tags=resume_tags, resume_action="reject"
            )
        except Exception as e:
            logger.warning("Error resuming graph on guardrail reject: %s", e)
        _reset_conversation_state()
        await cl.Message(
            content=(
                "### 🛑 Blocked by Guardrail Review\n\n"
                + (
                    "The request was not run."
                    if kind == "input_rail"
                    else "The flagged response was withheld."
                )
                + " Type a new query or `new query: …` to start fresh."
            )
        ).send()


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
        if "Contradiction" in clarif:
            cl.user_session.set("awaiting_transformer_clarification", True)
        else:
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

    if "informational_complete" in routed_node:
        # Request fully satisfied by informational/diagnostic tools alone
        # (e.g. a bare overview plot, or "what conditions does this
        # dataset have") — no AnalysisPlan was needed, nothing to gate,
        # nothing further to ask. Distinct from both clarification_pause
        # (genuinely missing info) and gate_1_review (needs plan approval).
        cl.user_session.set("awaiting_clarification", False)

        info_text = output_state.get("informational_response") or "Request completed."
        body = [
            "## ℹ️ Request Completed",
            "",
            "### Node Result Summary",
            summary_text,
            "",
            mask_text(info_text),
            "",
            "---",
            f"_Routed node_: `{routed_node}`",
        ]
        msg.content = "\n".join(body)
        await msg.update()

        # Same deterministic overview-plot lookup convention as the Gate 1
        # branch below — generate_dataset_overview_plot always writes to
        # outputs/plots/{run_id}_overview.png, regardless of exactly what
        # keys its own observation dict happens to return, so we don't need
        # to parse informational_artifacts to find it.
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

    if "gate_1_review" in routed_node:
        cl.user_session.set("awaiting_clarification", False)

        # Generalized gate: a NeMo rail pause carries a decision_context and is
        # presented through the same AskActionMessage/resume mechanics.
        if (output_state.get("decision_context") or {}).get("kind") in ("input_rail", "output_rail"):
            await _handle_rail_decision(output_state, run_id, msg)
            return

        # Overview plot is static for the whole Gate 1 turn — render it once
        # before the edit loop, not once per iteration.
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

        # --- Gate 1 HITL loop (edit → re-render → action) ---
        # Uses a while loop so multiple sequential edits stay in the same
        # turn without recursive calls into _handle_pipeline_output.
        while True:
            # (Re-)build and render the Gate 1 card (reflects edits made on
            # prior iterations via the manifest closure reference).
            msg.content = _build_gate_1_card(run_id, summary_text, plan, manifest, routed_node)
            await msg.update()

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
                with _trace_span(
                    name="gate_1_action_approve",
                    span_type="TOOL",
                    inputs={"run_id": run_id, "manifest_entry_count": len(manifest or [])},
                ) as _s:
                    # Resume LangGraph checkpoint — Data Prep now executes within this stream
                    pipeline_app = cl.user_session.get("graph")
                    config = {"configurable": {"thread_id": run_id}}
                    dp_error = None
                    if pipeline_app:
                        try:
                            list(pipeline_app.stream(Command(resume={"action": "approve"}), config=config))
                        except Exception as e:
                            logger.warning("Error resuming graph on approve: %s", e)
                            dp_error = f"Graph resume failed: {type(e).__name__}: {e}"

                    # Read final state after Data Prep execution
                    final_state: Dict[str, Any] = {}
                    if pipeline_app:
                        try:
                            snapshot = pipeline_app.get_state(config)
                            if snapshot and snapshot.values:
                                final_state = dict(snapshot.values)
                        except Exception:
                            pass

                    # Re-log the final manifest (with any human_approved_value edits) to MLflow
                    try:
                        mlflow_tracker.log_manifest(manifest)
                    except Exception as e:
                        logger.warning(
                            "MLflow log_manifest failed during Gate 1 Approve (post-edit artifact): %s: %s",
                            type(e).__name__, e,
                        )
                    _reset_conversation_state()
                    cl.user_session.set("gate_1_approved", True)
                    cl.user_session.set("preflight_confirmed", True)
                    output_state["gate_1_approved"] = True
                    output_state["preflight_confirmed"] = True

                    # Extract Data Prep results from final state
                    dp_error = dp_error or final_state.get("data_prep_error")
                    dp_dropped = final_state.get("bad_channels_dropped") or []
                    dp_plots = final_state.get("channel_plot_paths") or {}
                    dp_output = final_state.get("preprocessed_data_path")

                    # Build the result message
                    result_lines = [
                        "### ✅ Gate 1 Approved",
                        "",
                        "Manifest validated and locked. `preflight_confirmed=True`, `gate_1_approved=True`.",
                        "",
                    ]

                    if dp_error:
                        result_lines.extend([
                            "### ❌ Data Preparation Failed",
                            "",
                            f"```\n{mask_text(str(dp_error))}\n```",
                            "",
                        ])
                    elif dp_output:
                        result_lines.extend([
                            "### ✅ Data Preparation Complete",
                            "",
                            f"- **Output**: `{mask_text(str(dp_output))}`",
                        ])
                        if dp_dropped:
                            result_lines.append(
                                f"- **Bad channels removed**: {', '.join(f'`{c}`' for c in dp_dropped)}"
                            )
                        else:
                            result_lines.append("- **Bad channels removed**: 0 (all channels healthy ✓)")
                        if dp_plots:
                            result_lines.append(f"- **Diagnostic plots**: {len(dp_plots)} generated")
                            for pname, ppath in dp_plots.items():
                                result_lines.append(f"  - `{pname}`: `{mask_text(str(ppath))}`")
                        result_lines.append("")
                    else:
                        result_lines.extend([
                            "### ⚠️ Data Preparation",
                            "",
                            "Data Preparation executed but produced no output path.",
                            "",
                        ])

                    await cl.Message(content="\n".join(result_lines)).send()

                    # Display diagnostic plot images inline
                    for plot_name, plot_path in dp_plots.items():
                        if Path(plot_path).exists():
                            image = cl.Image(
                                path=plot_path,
                                name=plot_name,
                                display="inline",
                            )
                            await cl.Message(
                                content=f"### {plot_name.replace('_', ' ').title()}",
                                elements=[image],
                            ).send()

                    if _s is not None:
                        _s.set_outputs({
                            "gate_1_approved": True,
                            "preflight_confirmed": True,
                            "data_prep_success": dp_output is not None and dp_error is None,
                        })
                return

            # --- Reject ---
            elif chosen_action == "reject":
                with _trace_span(
                    name="gate_1_action_reject",
                    span_type="TOOL",
                    inputs={"run_id": run_id},
                ) as _s:
                    # Resume LangGraph checkpoint with reject
                    pipeline_app = cl.user_session.get("graph")
                    config = {"configurable": {"thread_id": run_id}}
                    if pipeline_app:
                        try:
                            list(pipeline_app.stream(Command(resume={"action": "reject"}), config=config))
                        except Exception as e:
                            logger.warning("Error resuming graph on reject: %s", e)
                    _reset_conversation_state()
                    cl.user_session.set("gate_1_approved", False)
                    cl.user_session.set("preflight_confirmed", False)
                    output_state["gate_1_approved"] = False
                    output_state["preflight_confirmed"] = False
                    await cl.Message(
                        content=(
                            "### 🛑 Gate 1 Rejected\n\n"
                            "Pipeline run aborted without executing downstream nodes. "
                            "Type a new query or `new query: …` to start fresh."
                        )
                    ).send()
                    if _s is not None:
                        _s.set_outputs({"gate_1_approved": False, "preflight_confirmed": False})
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

                    # Skip query transformer per spec (intent already unambiguous).
                    # Shared helpers keep lineage counters in sync with the
                    # standalone Request Changes path.
                    feedback = f"Change {entry_name} to {edit_value}"
                    revision_query = _build_revision_query(feedback)
                    base_run_id, next_run_id, rev_counter, rev_tags = _prepare_revision_run(run_id)

                    # Also keep accumulated_query updated for any follow-up
                    # plain messages that route through on_message's else-branch
                    current_accumulated = cl.user_session.get("accumulated_query", "")
                    new_accumulated = f"{current_accumulated}\n[User Gate 1 Change Request]: {feedback}"
                    cl.user_session.set("accumulated_query", new_accumulated)

                    re_msg = cl.Message(
                        content=f"🔄 `{mask_text(entry_name)}` requires Supervisor re-validation — re-invoking with: *'{mask_text(edit_value)}'*…"
                    )
                    await re_msg.send()

                    try:
                        with _trace_span(
                            name="revision_supervisor_rerun_scientific_edit",
                            span_type="CHAIN",
                            inputs={
                                "entry_name": entry_name,
                                "edit_value": edit_value[:100],
                                "revision": rev_counter,
                                "next_run_id": next_run_id,
                            },
                        ) as _rev_span:
                            new_state, new_nodes = await cl.make_async(_run_pipeline_sync)(
                                revision_query,
                                next_run_id,
                                tags=rev_tags,
                            )
                            if _rev_span is not None:
                                _rev_span.set_outputs({
                                    "routed_node": new_state.get("_routed_node", ""),
                                    "manifest_entries": len(new_state.get("parameter_manifest") or []),
                                })
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
                with _trace_span(
                    name="gate_1_direct_edit_advisory",
                    span_type="TOOL",
                    inputs={"entry_name": entry_name, "category": target_entry.category},
                ) as _s:
                    target_entry.human_approved_value = edit_value
                    if _s is not None:
                        _s.set_outputs({"approved_value_preview": edit_value[:50]})
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

                # Skip query transformer per spec (intent already unambiguous).
                # Shared helpers keep lineage counters in sync with the
                # scientific-axis per-row edit path.
                revision_query = _build_revision_query(feedback_text)
                base_run_id, next_run_id, rev_counter, rev_tags = _prepare_revision_run(run_id)

                # Also keep accumulated_query updated for any follow-up
                # plain messages that route through on_message's else-branch
                current_accumulated = cl.user_session.get("accumulated_query", "")
                new_accumulated = f"{current_accumulated}\n[User Gate 1 Change Request]: {feedback_text}"
                cl.user_session.set("accumulated_query", new_accumulated)

                re_msg = cl.Message(
                    content=f"🔄 Applying requested changes: *'{mask_text(feedback_text)}'* — Re-invoking Supervisor Agent…"
                )
                await re_msg.send()

                try:
                    with _trace_span(
                        name="revision_supervisor_rerun_request_changes",
                        span_type="CHAIN",
                        inputs={
                            "feedback_text": feedback_text[:200],
                            "revision": rev_counter,
                            "next_run_id": next_run_id,
                        },
                    ) as _rev_span:
                        new_state, new_nodes = await cl.make_async(_run_pipeline_sync)(
                            revision_query,
                            next_run_id,
                            tags=rev_tags,
                        )
                        if _rev_span is not None:
                            _rev_span.set_outputs({
                                "routed_node": new_state.get("_routed_node", ""),
                                "manifest_entries": len(new_state.get("parameter_manifest") or []),
                            })
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
