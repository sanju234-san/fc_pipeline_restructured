from __future__ import annotations

import asyncio
import datetime
import json
import shutil
import zipfile
from urllib.request import urlopen, Request
from urllib.parse import urlsplit
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
from fc_pipeline.agentic.supervisor.tools.dataset_overview_plot import generate_dataset_overview_plot
from fc_pipeline.config.thresholds import SUPERVISOR_CONFIDENCE_THRESHOLD, TAU_PHASE, TAU_ZEROLAG
from fc_pipeline.schemas.state import GraphState
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.schemas.clarification import build_clarification, get_clarification
from fc_pipeline.agentic.supervisor.agent import _channel_options, _valid_frequency_band_options
from fc_pipeline.pipeline.graph import build_pipeline_graph, compile_pipeline_app
from fc_pipeline.observability import mlflow_tracker
from fc_pipeline.agentic.supervisor.query_transformer import transform_query
from fc_pipeline.agentic.supervisor.tracer import supervisor_event_sink


def _strip_local_dir(match) -> str:
    """Replace a local absolute path with [LOCAL_ROOT]/<filename>, keeping the
    filename (existing tests / UI rely on the filename staying visible) but
    dropping every intermediate folder name (which is where usernames and
    project layout actually live)."""
    filename = match.group("file") or ""
    return f"[LOCAL_ROOT]/{filename}"


def mask_text(text: str) -> str:
    if not text:
        return text
    user = os.getenv("USERNAME") or os.getenv("USER") or ""
    if user:
        text = re.sub(re.escape(user), "[MASKED_USER]", text, flags=re.IGNORECASE)
    # URLs/IPs first, so a path-shaped fragment inside a URL doesn't get
    # double-processed by the path regexes below.
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
    # Local absolute paths (Windows drive or POSIX): strip every directory
    # component, not just up through the username — the earlier version only
    # replaced "\Users\<name>\", leaving the rest of the folder tree exposed.
    text = re.sub(
        r"(?:[A-Za-z]:)?(?:\\[^\\\r\n]+)+\\(?P<file>[^\\\r\n]*)",
        _strip_local_dir,
        text,
    )
    text = re.sub(
        r"(?<![\w/])/(?:[^/\s]+/)+(?P<file>[^/\s]*)",
        _strip_local_dir,
        text,
    )
    # Internal run/scenario identifiers are never meant for the chat — strip
    # them whether the model writes "Run ID: ..." / "scenario_id: ..." or
    # just echoes a bare run-id-shaped token.
    text = re.sub(r"(?i)\b(run[\s_-]?id|scenario[\s_-]?id)\s*[:=]\s*\S+", "[RUN_ID MASKED]", text)
    text = re.sub(r"\bchainlit_\d{8}_\d{6}_\d{3}(?:_rev\d+)?\b", "[RUN_ID MASKED]", text)
    text = re.sub(
        r"(?i)(?<![\w/])(?:outputs|logs|artifacts|data|tmp|uploads)/[^\s\"']+",
        lambda m: f"[ARTIFACT]/{Path(m.group(0).rstrip('.,;:)')).name}",
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


SUPPORTED_DATA_EXTENSIONS = {".fif", ".edf"}


def _validate_dataset_path(path: Path) -> tuple[bool, str]:
    """Validate an EEG file, including extensionless/incorrectly labelled downloads."""
    if not path.exists() or not path.is_file():
        return False, "Dataset file does not exist."

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DATA_EXTENSIONS:
        # Probe readers for files whose server omitted or mislabelled the extension.
        detected = None
        for ext, reader in ((".fif", mne.io.read_raw_fif), (".edf", mne.io.read_raw_edf)):
            try:
                raw = reader(str(path), preload=False, verbose=False)
                raw.close()
                detected = ext
                break
            except Exception:
                continue
        if detected is None:
            return False, "URL/upload is not a readable .fif or .edf EEG dataset."
        canonical = path.with_suffix(detected)
        if canonical != path:
            path.rename(canonical)
            path = canonical

    info = get_dataset_info.invoke({"data_path": str(path)})
    if info.get("error"):
        return False, mask_text(str(info["error"]))
    if not info.get("available_channels") or float(info.get("sfreq", 0)) <= 0:
        return False, "Dataset validation failed: no usable EEG channels or sampling frequency were found."
    return True, ""


def _extract_dataset_zip(zip_path: Path) -> Path:
    """Safely extract the first supported EEG file from a user ZIP."""
    root = Path("outputs") / "uploaded_datasets" / zip_path.stem
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and Path(m.filename).suffix.lower() in SUPPORTED_DATA_EXTENSIONS]
        if not members:
            raise ValueError("ZIP contains no supported .fif or .edf EEG file.")
        member = members[0]
        target = (root / Path(member.filename).name).resolve()
        if root.resolve() not in target.parents:
            raise ValueError("Unsafe ZIP entry path rejected.")
        with zf.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def _dataset_from_upload(file_obj: Any) -> Path:
    raw_path = Path(getattr(file_obj, "path", ""))
    if not raw_path.exists():
        raise ValueError("Uploaded file could not be located on the server.")
    if raw_path.suffix.lower() == ".zip":
        return _extract_dataset_zip(raw_path)
    return raw_path


async def _prompt_for_analysis_query() -> str:
    """Collect the user's analysis request immediately after dataset selection.

    The request is optional: an empty response means ``analyze this EEG``.
    Keeping this at startup means the Supervisor receives the user's actual
    intent on its first run instead of starting from a generic placeholder.
    """
    reply = await cl.AskUserMessage(
        content=(
            "### What would you like to analyze?\n\n"
            "Describe the EEG analysis you want. For example:\n"
            "- `compute PLI and coherence for alpha band on C3 and C4 during rest`\n"
            "- `compare theta connectivity between rest and task`\n"
            "- `analyze this EEG`\n\n"
            "You can leave this blank to let the Supervisor inspect the dataset and "
            "ask the required HITL questions."
        ),
        timeout=300,
    ).send()
    query = (
        reply.get("output", "")
        if isinstance(reply, dict)
        else getattr(reply, "output", "")
    ).strip()
    return query or "analyze this EEG"


async def _prompt_for_dataset() -> tuple[Path, bool, str]:
    """Require the user to provide an EEG dataset and analysis request.

    The startup contract is intentionally strict: no synthetic/demo fallback.
    The user must upload .fif/.edf/.zip or provide a URL that resolves to one,
    then may describe the analysis they want before the Supervisor starts.
    """
    while True:
        choice = await cl.AskActionMessage(
            content=(
                "### Dataset required\n\n"
                "Provide the EEG dataset before the analysis pipeline starts."
            ),
            actions=[
                cl.Action(
                    name="upload",
                    payload={"value": "upload"},
                    label="📁 Upload .fif / .edf / .zip",
                    description="Upload an EEG file or ZIP containing one.",
                ),
                cl.Action(
                    name="url",
                    payload={"value": "url"},
                    label="🔗 Provide dataset URL",
                    description="Paste a URL that resolves to a .fif, .edf, or .zip file.",
                ),
            ],
            timeout=300,
        ).send()
        if choice is None:
            raise RuntimeError("Dataset input timed out. Please restart the chat and provide a dataset.")

        name = choice.get("name") if isinstance(choice, dict) else getattr(choice, "name", None)
        if name == "upload":
            files = await cl.AskFileMessage(
                content="Upload one `.fif`, `.edf`, or `.zip` EEG dataset.",
                accept={
                    "application/octet-stream": [".fif", ".edf"],
                    "application/zip": [".zip"],
                },
                max_size_mb=1000,
                max_files=1,
                timeout=300,
            ).send()
            if not files:
                await cl.Message(content="⚠️ No dataset was uploaded. Please choose an input method again.").send()
                continue
            try:
                path = await cl.make_async(_dataset_from_upload)(files[0])
                ok, reason = await cl.make_async(_validate_dataset_path)(path)
                if not ok:
                    raise ValueError(reason)
                query = await _prompt_for_analysis_query()
                return path.resolve(), True, query
            except Exception as exc:
                await cl.Message(content=f"❌ Dataset rejected: {mask_text(str(exc))}").send()
                continue

        if name == "url":
            url_reply = await cl.AskUserMessage(
                content=(
                    "Paste the dataset URL. It may point directly to `.fif`, `.edf`, or `.zip`; "
                    "the URL does not have to end with the file extension."
                ),
                timeout=300,
            ).send()
            url = (
                url_reply.get("output", "")
                if isinstance(url_reply, dict)
                else getattr(url_reply, "output", "")
            ).strip()
            if not url:
                await cl.Message(content="⚠️ No URL was provided. Please try again.").send()
                continue
            try:
                await cl.Message(content="⏳ Downloading dataset from URL...").send()
                path = await cl.make_async(_download_dataset_url)(url)
                ok, reason = await cl.make_async(_validate_dataset_path)(path)
                if not ok:
                    raise ValueError(reason)
                query = await _prompt_for_analysis_query()
                return path.resolve(), True, query
            except Exception as exc:
                await cl.Message(
                    content=(
                        f"❌ **Dataset URL could not be downloaded/validated:** {mask_text(str(exc))}\n\n"
                        "Please try another URL."
                    )
                ).send()
                continue

        await cl.Message(content="⚠️ Please choose **Upload** or **Provide dataset URL**.").send()


def _download_dataset_url(url: str) -> Path:
    """Download a URL and identify its actual EEG/ZIP type.

    Handles query strings/tracking parameters, redirects, missing file
    extensions, and servers that return application/octet-stream.
    """
    download_dir = Path("outputs") / "uploaded_datasets"
    download_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    parsed = urlsplit(url)
    url_suffix = Path(parsed.path).suffix.lower()

    req = Request(url, headers={"User-Agent": "EEG-FC-Pipeline/1.0"})
    with urlopen(req, timeout=120) as src:
        content_type = (src.headers.get("Content-Type") or "").lower()
        disposition = src.headers.get("Content-Disposition") or ""
        final_url = src.geturl() or url
        final_suffix = Path(urlsplit(final_url).path).suffix.lower()
        filename_hint = ""
        match = re.search(r'filename[^;=]*=\s*["\']?([^"\';]+)', disposition, flags=re.I)
        if match:
            filename_hint = match.group(1).strip()

        suffix = Path(filename_hint).suffix.lower() or final_suffix or url_suffix
        if suffix not in SUPPORTED_DATA_EXTENSIONS and suffix != ".zip":
            if "zip" in content_type:
                suffix = ".zip"
            else:
                suffix = ".bin"
        destination = download_dir / f"remote_dataset_{stamp}{suffix}"
        with open(destination, "wb") as dst:
            shutil.copyfileobj(src, dst)

    # ZIP magic is more reliable than Content-Type when a server mislabels it.
    with open(destination, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] == b"PK" and destination.suffix.lower() != ".zip":
        zip_path = destination.with_suffix(".zip")
        destination.rename(zip_path)
        destination = zip_path

    if destination.suffix.lower() == ".zip":
        return _extract_dataset_zip(destination)

    # If the URL had no useful extension, probe both supported MNE readers and
    # rename to the canonical extension so downstream MNE loading is reliable.
    if destination.suffix.lower() not in SUPPORTED_DATA_EXTENSIONS:
        for ext, reader in ((".fif", mne.io.read_raw_fif), (".edf", mne.io.read_raw_edf)):
            try:
                raw = reader(str(destination), preload=False, verbose=False)
                raw.close()
                canonical = destination.with_suffix(ext)
                destination.rename(canonical)
                return canonical
            except Exception:
                continue
        raise ValueError("URL did not return a readable .fif, .edf, or ZIP EEG dataset.")

    return destination


# Categories where direct edits MUST route through Supervisor re-validation
# (the 3 mandatory scientific axes + metric selection feeding structural invariants).
SCIENTIFIC_EDIT_CATEGORIES = {"scientific_axis", "metric_selection"}


def format_manifest_markdown(manifest: Optional[List[ParameterManifestEntry]]) -> str:
    if not manifest:
        return "_No parameter manifest generated._\n"
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


def write_analysis_plan_markdown(
    plan: Any, manifest: Optional[List[ParameterManifestEntry]], run_id: str,
) -> Path:
    """Persist the Gate 1 plan as a real Markdown artifact for the chat UI."""
    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"analysis_plan_{run_id}.md"
    body = [
        "# EEG Functional Connectivity — Analysis Plan",
        "",
        format_plan_markdown(plan),
        "## Parameter Manifest",
        "",
        format_manifest_markdown(manifest),
        "## Review Notes",
        "",
        "- This document is generated deterministically from the Supervisor's resolved plan and manifest.",
        "- Values marked as needing human input remain subject to Gate 1 review.",
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def _json_for_ui(value: Any) -> str:
    try:
        return mask_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    except Exception:
        return mask_text(str(value))


def _mark_manifest_human_approved(manifest: Optional[List[ParameterManifestEntry]]) -> None:
    """Record the explicit Gate 1 approval on any row that required HITL.

    A low-confidence Supervisor/tool resolution is allowed to reach Gate 1,
    where the human can inspect the concrete proposed value. Clicking the
    explicit Gate 1 **Approve** action is the human approval for that value.
    Data Prep must receive that approval in the graph checkpoint; otherwise it
    would reject the same value as ``HUMAN_APPROVAL_REQUIRED`` even though the
    user just approved the plan.
    """
    for entry in manifest or []:
        if entry.needs_human_input and entry.human_approved_value is None:
            proposed = str(entry.proposed_value or "").strip()
            if proposed:
                entry.human_approved_value = proposed


def _fallback_clarification_options(output_state: Dict[str, Any]) -> List[Dict[str, str]]:
    """Rebuild bounded HITL choices from dataset metadata persisted in GraphState.

    Only used when the authoritative clarification payload carries no options.
    Choices are always grounded in the loaded dataset (never invented) and reuse
    the exact builders the Supervisor uses, so a fallback choice is always valid.
    """
    kind = str((get_clarification(output_state) or {}).get("kind") or output_state.get("clarification_kind") or "")
    if kind == "channel_selection":
        return _channel_options(output_state.get("dataset_available_channels"))
    if kind == "frequency_band":
        return _valid_frequency_band_options(
            output_state.get("dataset_sfreq"), output_state.get("dataset_duration_seconds")
        )
    if kind == "condition":
        candidates = output_state.get("condition_candidates") or []
        return [
            {"name": f"condition_{i}", "value": str(c), "label": str(c)}
            for i, c in enumerate(candidates) if str(c).strip()
        ]
    return []


def _build_clarification_actions(output_state: Dict[str, Any]) -> List[cl.Action]:
    """Build the HITL controls for an unresolved clarification.

    Reads the ONE authoritative payload (``GraphState["clarification"]``, or the
    legacy flat fields via ``get_clarification``).  Bounded choices come first;
    ``Type manually`` is ALWAYS appended, so an unresolved clarification can
    never reach the user with only the ordinary text box.
    """
    clarification = get_clarification(output_state) or build_clarification(
        "clarification", "Please provide the missing analysis parameter(s)."
    )
    options = list(clarification.get("options") or [])
    if not options:
        options = _fallback_clarification_options(output_state)

    actions: List[cl.Action] = []
    for i, option in enumerate(options):
        value = str(option.get("value", option.get("label", ""))).strip()
        if not value:
            continue
        actions.append(
            cl.Action(
                name=str(option.get("name") or f"clarification_option_{i}"),
                payload={"value": value},
                label=str(option.get("label") or value),
                tooltip="Use this explicit value and resume the pipeline.",
            )
        )

    actions.append(
        cl.Action(
            name="manual_clarification",
            payload={"value": "manual"},
            label="✍️ Type manually",
            tooltip="Provide a custom value instead of a suggested choice.",
        )
    )
    return actions


async def _stream_supervisor_events(queue: asyncio.Queue, done_event: asyncio.Event):
    """Render deterministic tool-call/observation events while the sync graph runs."""
    while True:
        event = await queue.get()
        if event is None:
            break
        kind = event.get("event_type")
        payload = event.get("payload") or {}
        if kind == "tool_call":
            tool = payload.get("tool", "tool")
            try:
                async with cl.Step(name=f"Tool: {tool}", type="tool", show_input=True) as step:
                    step.input = _json_for_ui(payload.get("args", {}))
                    step.output = "Running tool…"
            except (GeneratorExit, RuntimeError, Exception):
                pass
        elif kind == "tool_observation":
            tool = payload.get("tool", "tool")
            observation = payload.get("observation", {})
            elements = []
            ui_observation = observation
            if isinstance(observation, dict):
                # Render real plot artifacts inline, but never expose their filesystem
                # paths in the textual tool result.  The user sees the image itself.
                plot_path = observation.get("plot_path")
                if plot_path and Path(str(plot_path)).exists():
                    elements.append(cl.Image(path=str(plot_path), name=f"{tool}_plot", display="inline"))
                ui_observation = dict(observation)
                for sensitive_key in ("plot_path", "data_path", "raw_data_path", "output_path", "preprocessed_data_path"):
                    ui_observation.pop(sensitive_key, None)
                if plot_path and "plot_path" not in ui_observation:
                    ui_observation["plot"] = "Rendered inline in the chat."
            try:
                async with cl.Step(name=f"Tool result: {tool}", type="tool", elements=elements) as step:
                    step.output = _json_for_ui(ui_observation)
            except (GeneratorExit, RuntimeError, Exception):
                pass
        elif kind == "manifest_compiled":
            try:
                async with cl.Step(name="Supervisor: analysis plan compiled", type="run") as step:
                    step.output = _json_for_ui({
                        "plan": payload.get("plan"),
                        "manifest_rows": payload.get("manifest_rows"),
                    })
            except (GeneratorExit, RuntimeError, Exception):
                pass
        elif kind in {"supervisor_informational_complete", "supervisor_halt_unresolved_axes", "structural_validation_failed"}:
            try:
                async with cl.Step(name=f"Supervisor: {kind.replace('_', ' ')}", type="run") as step:
                    step.output = _json_for_ui(payload)
            except (GeneratorExit, RuntimeError, Exception):
                pass
        elif kind == "data_prep_stage":
            stage = payload.get("stage", "data preparation")
            try:
                async with cl.Step(name=f"Data Prep: {str(stage).replace('_', ' ').title()}", type="run") as step:
                    step.output = _json_for_ui(payload)
            except (GeneratorExit, RuntimeError, Exception):
                pass
        elif kind == "data_prep_failed":
            try:
                async with cl.Step(name="Data Prep: failed", type="run") as step:
                    step.output = _json_for_ui(payload)
            except (GeneratorExit, RuntimeError, Exception):
                pass
        elif kind == "supervisor_start":
            try:
                async with cl.Step(name="Supervisor: start", type="run") as step:
                    step.output = "Scientific parameter resolution started."
            except (GeneratorExit, RuntimeError, Exception):
                pass
        # agent_thought is intentionally not surfaced: the UI gets the factual
        # tool inputs/results and deterministic state transitions, not hidden model reasoning.
    done_event.set()


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
    """Collect a real dataset, then immediately start the analysis workflow."""
    try:
        data_path, is_real_data, user_query = await _prompt_for_dataset()
    except Exception as exc:
        await cl.Message(content=f"❌ **Dataset input stopped:** {mask_text(str(exc))}").send()
        return

    try:
        _ = get_supervisor_llm()
        llm_ok = True
    except Exception as e:
        llm_ok = False
        await cl.Message(
            content=f"❌ **LLM service is not configured:** {mask_text(str(e))}"
        ).send()

    if not llm_ok:
        return

    session_state = {
        "original_query": user_query,
        "accumulated_query": user_query,
        "awaiting_clarification": False,
        "awaiting_transformer_clarification": False,
        "base_run_id": None,
        "gate_1_revision": 0,
        "gate_1_approved": False,
        "preflight_confirmed": False,
        "condensed_query": None,
        "data_path": str(data_path),
        "is_real_data": is_real_data,
        "graph": compile_pipeline_app(),
        "counter": 0,
        "completed_run_active": False,
        "completed_run_state": None,
        "post_run_action_active": False,
    }
    for k, v in session_state.items():
        cl.user_session.set(k, v)

    # Deliberately do not show a Dataset Overview here. Metadata is available
    # to the Supervisor tools; the chat should proceed directly into analysis.
    await cl.Message(
        content=(
            "# EEG Functional Connectivity Pipeline\n\n"
            f"✅ **Dataset accepted:** `{mask_text(data_path.name)}`\n\n"
            "🔎 **Starting analysis automatically…**\n\n"
            f"📝 **Analysis request:** {mask_text(user_query)}\n\n"
            "The Supervisor will inspect the dataset and your request, resolve the analysis parameters, "
            "ask HITL questions when needed, create the Analysis Plan, and pause at Gate 1 for review."
        )
    ).send()

    counter = 1
    run_id = f"chainlit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter:03d}"
    cl.user_session.set("counter", counter)
    cl.user_session.set("base_run_id", run_id)
    cl.user_session.set("latest_user_message", user_query)

    msg = cl.Message(content="🔄 **Supervisor is analyzing the dataset…**")
    await msg.send()

    event_queue: asyncio.Queue = asyncio.Queue()
    event_done = asyncio.Event()
    event_task = asyncio.create_task(_stream_supervisor_events(event_queue, event_done))
    loop = asyncio.get_running_loop()

    def _event_sink(event: Dict[str, Any]):
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, event)
        except Exception:
            pass

    try:
        output_state, executed_nodes = await cl.make_async(_run_pipeline_sync)(
            user_query,
            run_id,
            tags={"scenario_id": run_id, "parent_scenario_id": run_id, "revision": 0},
            latest_user_message=user_query,
            event_sink=_event_sink,
        )
    except Exception as exc:
        msg.content = f"❌ **Pipeline execution failed:** {mask_text(str(exc))}"
        await msg.update()
        await event_queue.put(None)
        await event_task
        return

    await event_queue.put(None)
    await event_task
    if output_state.get("user_request"):
        cl.user_session.set("condensed_query", output_state.get("user_request"))

    await _handle_pipeline_output(
        output_state,
        executed_nodes,
        user_query,
        run_id,
        msg,
    )


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
        "clarification": None,
        "clarification_question": None,
        "clarification_kind": None,
        "clarification_options": [],
        "condition_candidates": [],
        "clarification_response": None,
        "clarification_resume_kind": None,
        "resolved_frequency_band_info": None,
        "resolved_channel_info": None,
        "resolved_condition_value": None,
        "dataset_sfreq": None,
        "dataset_duration_seconds": None,
        "dataset_available_channels": None,
        "dataset_reference": None,
        "informational_response": None,
        "informational_artifacts": None,
        "decision_context": None,
        "input_rail_cleared": False,
        "bad_channels_dropped": None,
        "channel_plot_paths": None,
        "preprocessed_data_path": None,
        "data_prep_error": None,
        "data_prep_summary": None,
        "metric_csv_paths": None,
        "heatmap_image_paths": None,
        "network_image_paths": None,
        "numerical_summaries": None,
        "evidence_summary": None,
        "evaluation_verdict": None,
        "evaluation_summary_csv_path": None,
        "final_report_path": None,
    }


def _interrupt_node_name(payload: Dict[str, Any], current_state: Optional[Dict[str, Any]] = None) -> str:
    """Identify which native LangGraph interrupt paused the graph."""
    interrupts = payload.get("__interrupt__") if isinstance(payload, dict) else None
    if interrupts:
        if not isinstance(interrupts, (list, tuple)):
            interrupts = [interrupts]
        for item in interrupts:
            value = getattr(item, "value", item)
            if isinstance(value, dict) and value.get("type") == "clarification":
                return "clarification_pause"
    state = current_state or {}
    if state.get("clarification_question"):
        return "clarification_pause"
    return "gate_1_review"


def _resume_pipeline_sync(
    run_id: str,
    action: Any,
    event_sink=None,
    approved_manifest: Optional[List[ParameterManifestEntry]] = None,
) -> tuple[Dict[str, Any], Optional[str]]:
    """Resume a paused LangGraph checkpoint while preserving live trace events.

    When Gate 1 is explicitly approved, ``approved_manifest`` is written into
    the checkpoint before resuming. This is important because the Chainlit
    review UI owns the human-edited manifest object, while LangGraph owns the
    paused graph state. Without synchronising the two, Data Prep would see the
    pre-review manifest and could correctly reject a low-confidence row as
    lacking an explicit human-approved value.
    """
    pipeline_app = cl.user_session.get("graph")
    if pipeline_app is None:
        return {}, "Graph is not initialized."
    config = {"configurable": {"thread_id": run_id}}
    try:
        if approved_manifest is not None:
            # This update is only supplied by the explicit Gate 1 Approve
            # handler. It does not itself set gate_1_approved; the native
            # LangGraph interrupt resume below remains the sole approval path.
            pipeline_app.update_state(
                config,
                {"parameter_manifest": approved_manifest},
            )
        resume_payload = action if isinstance(action, dict) else {"action": action}
        executed_nodes: List[str] = []
        with supervisor_event_sink(event_sink):
            for mode, payload in pipeline_app.stream(
                Command(resume=resume_payload),
                config=config,
                stream_mode=["updates", "values"],
            ):
                if mode == "updates":
                    for k in payload.keys():
                        executed_nodes.append(
                            _interrupt_node_name(payload, None)
                            if k == "__interrupt__" else k
                        )
                elif mode == "values":
                    pass

        snapshot = pipeline_app.get_state(config)
        state = dict(snapshot.values) if snapshot and snapshot.values else {}
        if snapshot and snapshot.next:
            for paused_node in ("clarification_pause", "gate_1_review"):
                if paused_node in snapshot.next and paused_node not in executed_nodes:
                    executed_nodes.append(paused_node)
        state["_executed_nodes"] = executed_nodes
        if executed_nodes:
            state["_routed_node"] = executed_nodes[-1]
        return state, None
    except Exception as exc:
        return {}, f"Graph resume failed: {type(exc).__name__}: {exc}"


def _run_pipeline_sync(
    user_request: str,
    run_id: str,
    tags: Optional[Dict[str, Any]] = None,
    *,
    latest_user_message: Optional[str] = None,
    resume_action: Optional[str] = None,
    event_sink=None,
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
        with supervisor_event_sink(event_sink):
            stream_iter = pipeline_app.stream(
                stream_input,
                config=config,
                stream_mode=["updates", "values"],
            )
            for mode, payload in stream_iter:
                if mode == "updates":
                    for k in payload.keys():
                        if k == "__interrupt__":
                            executed_nodes.append(_interrupt_node_name(payload, output_state))
                        else:
                            executed_nodes.append(k)
                elif mode == "values":
                    output_state = payload

        # Check snapshot for interrupt state (e.g. paused at gate_1_review)
        snapshot = pipeline_app.get_state(config)
        if snapshot and snapshot.values:
            output_state = dict(snapshot.values)
            if snapshot.next:
                for paused_node in ("clarification_pause", "gate_1_review"):
                    if paused_node in snapshot.next and paused_node not in executed_nodes:
                        executed_nodes.append(paused_node)
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
        or user_text_lc == "/reset"
        or user_text_lc == "reset"
        or user_text_lc.startswith("reset ")
    )

    # A completed run is intentionally read-only for ordinary follow-ups.
    # Do this before any query transformation/pipeline invocation so a user
    # asking for a plot or summary cannot accidentally restart Supervisor/Data Prep.
    if cl.user_session.get("completed_run_active", False) and not is_explicit_new:
        handled = await _handle_completed_run_followup(user_text)
        if handled:
            return

    # A typed answer to a pending HITL clarification resumes the SAME LangGraph
    # checkpoint (never a fresh run that would forget already-resolved axes).
    pending_run_id = cl.user_session.get("pending_clarification_run_id")
    if pending_run_id and not is_explicit_new:
        pipeline_app = cl.user_session.get("graph")
        snap = pipeline_app.get_state({"configurable": {"thread_id": pending_run_id}}) if pipeline_app else None
        if snap is not None and snap.next and "clarification_pause" in snap.next:
            await _resume_clarification(pending_run_id, user_text, accumulated_query or "")
            return
        cl.user_session.set("pending_clarification_run_id", None)

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
        cl.user_session.set("pending_clarification_run_id", None)
        cl.user_session.set("gate_1_revision", 0)
        cl.user_session.set("gate_1_approved", False)
        cl.user_session.set("completed_run_active", False)
        cl.user_session.set("completed_run_state", None)
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

    # --- Invoke Pipeline ---
    msg = cl.Message(content="🔄 Processing your request…")
    await msg.send()

    event_queue: asyncio.Queue = asyncio.Queue()
    event_done = asyncio.Event()
    event_task = asyncio.create_task(_stream_supervisor_events(event_queue, event_done))
    loop = asyncio.get_running_loop()

    def _event_sink(event: Dict[str, Any]):
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, event)
        except Exception:
            pass

    try:
        output_state, executed_nodes = await cl.make_async(_run_pipeline_sync)(
            accumulated_query,
            run_id,
            tags={"scenario_id": run_id, "parent_scenario_id": run_id, "revision": 0},
            event_sink=_event_sink,
        )
    except Exception as e:
        err = f"**Pipeline execution failed**: `{type(e).__name__}: {e}`"
        msg.content = mask_text(err)
        await msg.update()
        await event_queue.put(None)
        await event_task
        return

    await event_queue.put(None)
    await event_task

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


# ---------------------------------------------------------------------------
# Post-run follow-up helpers.
# Read-only inspection of the COMPLETED run's stored artifacts. Nothing here
# invokes the Supervisor, Data Prep, HITL, Gate 1 or the LangGraph app.
# ---------------------------------------------------------------------------
_FOLLOWUP_MAX_CHANNELS = 8


def _followup_load_sources(state: Dict[str, Any]):
    """Open the completed run's artifacts read-only -> (epochs | None, raw | None)."""
    epochs = None
    raw = None
    pre = state.get("preprocessed_data_path")
    if pre and Path(str(pre)).exists():
        try:
            epochs = mne.read_epochs(str(pre), preload=True, verbose=False)
        except Exception as exc:
            logger.warning("Post-run: could not read preprocessed epochs: %s: %s", type(exc).__name__, exc)
    raw_path = state.get("raw_data_path")
    if raw_path and Path(str(raw_path)).exists():
        try:
            raw = mne.io.read_raw(str(raw_path), preload=False, verbose=False)
        except Exception as exc:
            logger.warning("Post-run: could not read raw dataset: %s: %s", type(exc).__name__, exc)
    return epochs, raw


def _followup_find_channels(text: str, names: List[str]) -> List[str]:
    """Return channel names mentioned in `text`, in order of appearance."""
    hits: List[tuple] = []
    for name in sorted(set(names), key=len, reverse=True):
        m = re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text, re.IGNORECASE)
        if m:
            hits.append((m.start(), name))
    hits.sort()
    return [name for _, name in hits]


def _followup_extract(epochs, raw, channels: List[str]) -> Dict[str, Dict[str, Any]]:
    """Signal + PSD per channel. Prefers the preprocessed epochs; falls back to raw."""
    from scipy.signal import welch as scipy_welch

    out: Dict[str, Dict[str, Any]] = {}
    for ch in channels:
        if epochs is not None and ch in epochs.ch_names:
            sig = epochs.get_data(picks=[ch])[0, 0]  # first epoch
            sf = float(epochs.info["sfreq"])
            spec = epochs.compute_psd(
                method="welch", fmin=0.5, fmax=min(60.0, sf / 2.0 - 0.1), picks=[ch], verbose=False
            )
            out[ch] = {
                "source": "preprocessed",
                "t": np.asarray(epochs.times),
                "sig": np.asarray(sig),
                "freqs": np.asarray(spec.freqs),
                "psd": np.asarray(spec.get_data().mean(axis=0)[0]),
            }
        elif raw is not None and ch in raw.ch_names:
            sf = float(raw.info["sfreq"])
            full = raw.get_data(picks=[ch])[0]
            n = int(min(4.0 * sf, full.shape[0]))
            freqs, psd = scipy_welch(full, fs=sf, nperseg=min(int(sf * 2), full.shape[0]))
            mask = (freqs >= 0.5) & (freqs <= min(60.0, sf / 2.0))
            out[ch] = {
                "source": "raw",
                "t": np.arange(n) / sf,
                "sig": full[:n],
                "freqs": freqs[mask],
                "psd": psd[mask],
            }
    return out


def _followup_plot(kind: str, series: Dict[str, Dict[str, Any]], run_id: str, overlay: bool) -> Optional[str]:
    """Render a signal or PSD figure for the requested channels; returns the saved path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chs = list(series.keys())
    out_dir = Path("outputs") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"followup_{run_id}_{kind}_{'-'.join(chs)}")
    path = out_dir / f"{safe}.png"

    def _label(ch: str) -> str:
        return f"{ch} (raw)" if series[ch]["source"] == "raw" else ch

    if kind == "psd":
        fig, ax = plt.subplots(figsize=(8, 4))
        for ch in chs:
            ax.semilogy(series[ch]["freqs"], series[ch]["psd"], linewidth=1.0, label=_label(ch))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.set_title("Power Spectral Density — " + ", ".join(chs))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    elif overlay:
        fig, ax = plt.subplots(figsize=(10, 4))
        for ch in chs:
            ax.plot(series[ch]["t"], series[ch]["sig"] * 1e6, linewidth=0.8, label=_label(ch))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title("EEG signal — " + " vs ".join(chs))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    else:
        fig, axes = plt.subplots(len(chs), 1, figsize=(10, max(2.4, 2.0 * len(chs))), sharex=False, squeeze=False)
        for ax, ch in zip(axes[:, 0], chs):
            ax.plot(series[ch]["t"], series[ch]["sig"] * 1e6, linewidth=0.8, color="#4a90d9")
            ax.set_ylabel(f"{_label(ch)}\n(µV)", fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("Time (s)")
        axes[0, 0].set_title("EEG signal — " + ", ".join(chs))
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _followup_compare_stats(series: Dict[str, Dict[str, Any]]) -> List[str]:
    """Descriptive-only comparison lines for the plotted signals (no inference)."""
    chs = list(series.keys())
    lines: List[str] = []
    for ch in chs:
        sig = series[ch]["sig"] * 1e6
        lines.append(f"- **{ch}:** mean `{sig.mean():.2f} µV`, std `{sig.std():.2f} µV`")
    a, b = series[chs[0]], series[chs[1]]
    if a["source"] == b["source"] and len(a["sig"]) == len(b["sig"]) and a["sig"].std() > 0 and b["sig"].std() > 0:
        r = float(np.corrcoef(a["sig"], b["sig"])[0, 1])
        lines.append(
            f"- **Pearson r ({chs[0]} vs {chs[1]}, plotted segment):** `{r:.3f}` "
            "— a descriptive statistic of the two plotted traces, not a connectivity result."
        )
    return lines


_PLOT_EXPLANATIONS = {
    "channel_variance": (
        "Bar chart of each retained channel's mean variance across epochs. Bars far above or "
        "below the rest point to channels with residual noise or a nearly flat signal."
    ),
    "psd_overview": (
        "Welch power spectral density for every retained channel. The shaded region is the "
        "approved analysis band, confirming where the band-pass left power in the data."
    ),
}


async def _followup_send_plot_images(plots: Dict[str, str], title: str) -> None:
    if not plots:
        return
    await cl.Message(content=title).send()
    for plot_name, plot_path in plots.items():
        if plot_path and Path(str(plot_path)).exists():
            await cl.Message(
                content=f"#### {plot_name.replace('_', ' ').title()}",
                elements=[cl.Image(path=str(plot_path), name=f"after_{plot_name}", display="inline")],
            ).send()


async def _followup_channel_names_text(state: Dict[str, Any]) -> str:
    epochs, _raw = await cl.make_async(_followup_load_sources)(state)
    if epochs is None:
        return ""
    return ", ".join(f"`{mask_text(c)}`" for c in epochs.ch_names)


async def _followup_channel_request(text: str, low: str, state: Dict[str, Any]) -> bool:
    """Signal / PSD / compare for channels named in the message. True if handled."""
    epochs, raw = await cl.make_async(_followup_load_sources)(state)
    names: List[str] = list(epochs.ch_names) if epochs is not None else []
    if raw is not None:
        names += [c for c in raw.ch_names if c not in names]
    channels = _followup_find_channels(text, names)
    if not channels:
        return False

    run_id = state.get("run_id", "completed_run")
    truncated = len(channels) > _FOLLOWUP_MAX_CHANNELS
    channels = channels[:_FOLLOWUP_MAX_CHANNELS]

    wants_compare = any(k in low for k in ("compar", " vs ", "versus", "difference"))
    wants_psd = any(k in low for k in ("psd", "spectr", "power", "frequency"))
    wants_signal = any(k in low for k in ("signal", "trace", "waveform", "time series", "time-series", "voltage", "amplitude", "eeg"))

    if wants_compare and len(channels) < 2:
        await cl.Message(content="Please name **two channels** to compare, e.g. `Compare C3 and C4`.").send()
        return True

    series = await cl.make_async(_followup_extract)(epochs, raw, channels)
    if not series:
        return False

    notes: List[str] = []
    raw_only = [c for c, d in series.items() if d["source"] == "raw"]
    if raw_only:
        notes.append(
            f"{', '.join(f'`{mask_text(c)}`' for c in raw_only)} "
            "was not part of the approved channel selection, so it is shown from the **raw (unprocessed)** recording."
        )
    if truncated:
        notes.append(f"Showing the first {_FOLLOWUP_MAX_CHANNELS} channels you mentioned.")

    if wants_compare:
        jobs = [("signal", True), ("psd", True)]
    elif wants_psd and not wants_signal:
        jobs = [("psd", True)]
    elif wants_psd and wants_signal:
        jobs = [("signal", False), ("psd", True)]
    else:
        jobs = [("signal", False)]

    header = "### 📈 Follow-up on the completed analysis\nReusing the completed run — no Supervisor or Data Prep was triggered."
    if notes:
        header += "\n\n" + "\n".join(f"ℹ️ {n}" for n in notes)
    await cl.Message(content=header).send()

    for kind, overlay in jobs:
        try:
            img = await cl.make_async(_followup_plot)(kind, series, run_id, overlay)
        except Exception as exc:
            logger.warning("Post-run %s plot failed: %s: %s", kind, type(exc).__name__, exc)
            img = None
        title = "Signal" if kind == "signal" else "Power Spectral Density"
        if img and Path(img).exists():
            await cl.Message(
                content=f"#### {title}",
                elements=[cl.Image(path=img, name=f"followup_{kind}", display="inline")],
            ).send()
        else:
            await cl.Message(content=f"⚠️ Could not render the {title.lower()} plot for the requested channel(s).").send()

    if wants_compare:
        await cl.Message(content="#### Descriptive comparison\n" + "\n".join(_followup_compare_stats(series))).send()
    return True


async def _followup_explain(state: Dict[str, Any]) -> None:
    """Deterministic explanation of the completed run + its stored diagnostic plots."""
    summary = state.get("data_prep_summary")
    dropped = state.get("bad_channels_dropped")
    if dropped is None:
        dropped = list(getattr(summary, "dropped_channels", None) or [])
    view_state = {
        "bad_channels_dropped": dropped,
        "channel_plot_paths": state.get("channel_plot_paths") or {},
        "data_prep_summary": summary,
        "preprocessed_data_path": state.get("preprocessed_data_path"),
    }
    lines = ["### 📖 Explaining the completed analysis", "", "Reusing the stored results — nothing was re-run."]
    lines += _format_data_prep_explanation(view_state, state.get("parameter_manifest") or [])
    plots = state.get("channel_plot_paths") or {}
    if plots:
        lines += ["", "### 🖼️ What each plot shows", ""]
        for name in plots:
            desc = _PLOT_EXPLANATIONS.get(name, "Deterministic Data Prep diagnostic plot.")
            lines.append(f"- **{name.replace('_', ' ').title()}:** {desc}")
    await cl.Message(content="\n".join(lines)).send()
    await _followup_send_plot_images(plots, "### Diagnostic plots from the completed run")



async def _handle_completed_run_followup(user_text: str) -> bool:
    """Handle post-run questions without restarting Supervisor/Data Prep.

    Once a pipeline run has completed, ordinary chat messages are treated as
    post-run inspection requests. They may render artifacts from the completed
    run (for example, raw-before vs preprocessed-after plots), but they never
    invoke the Supervisor or Data Prep. A fresh pipeline run requires the
    explicit ``New query`` action.
    """
    text = user_text.strip()
    low = text.lower()
    state = cl.user_session.get("completed_run_state") or {}
    if not state:
        return False

    wants_plot = any(k in low for k in ("plot", "graph", "visual", "visualize", "figure"))
    wants_before = any(k in low for k in ("before", "pre-prep", "preprocessing", "raw"))
    wants_after = any(k in low for k in ("after", "post-prep", "postprocessing", "preprocessed", "data prep"))
    wants_both = (wants_before and wants_after) or any(
        k in low for k in ("before and after", "pre and post", "pre/post", "before vs after", "before versus after")
    )

    # --- Follow-ups added for the completed-run feature (read-only; no pipeline calls) ---
    if any(k in low for k in ("connectivity", "coherence", "wpli", "plv")):
        chan_text = await _followup_channel_names_text(state)
        await cl.Message(
            content=(
                "### ℹ️ No connectivity result exists for this run\n\n"
                "This completed analysis finished at **Data Preparation**; a connectivity stage was not "
                "executed for it, so there is nothing stored to display. I did not compute anything new.\n\n"
                "From the completed run I can show the **signal** or **PSD** of any retained channel, "
                "**compare two channels**, show the **before/after Data Prep** plots, or **explain the plots**."
                + (f"\n\nRetained channels: {chan_text}" if chan_text else "")
            )
        ).send()
        return True

    if await _followup_channel_request(text, low, state):
        return True

    if any(k in low for k in ("explain", "describe", "interpret", "summar", "what did", "what does", "walk me through")) \
            and any(k in low for k in ("plot", "figure", "graph", "diagnostic", "analysis", "data prep", "preprocess", "result", "run")) \
            and not wants_both:
        await _followup_explain(state)
        return True

    if wants_plot and (wants_both or (wants_before and wants_after)):
        raw_path = state.get("raw_data_path")
        preprocessed = state.get("preprocessed_data_path")
        run_id = state.get("run_id", "completed_run")

        await cl.Message(
            content="### 📊 Post-run comparison\nShowing the completed run's **raw dataset (before Data Prep)** and **preprocessed diagnostics (after Data Prep)**.\n\nNo Supervisor or Data Prep run is triggered by this request."
        ).send()

        # Before: generate/reuse a raw overview plot. This is an informational
        # post-run inspection only; it does not alter the completed pipeline.
        before_path = state.get("before_overview_plot")
        if not before_path or not Path(str(before_path)).exists():
            if raw_path and Path(str(raw_path)).exists():
                try:
                    result = generate_dataset_overview_plot.invoke({
                        "data_path": str(raw_path),
                        "run_id": f"{run_id}_postrun_before",
                    })
                    before_path = result.get("plot_path") if isinstance(result, dict) else None
                    if before_path:
                        state["before_overview_plot"] = before_path
                        cl.user_session.set("completed_run_state", state)
                except Exception as exc:
                    logger.warning("Post-run before-plot generation failed: %s", exc)

        if before_path and Path(str(before_path)).exists():
            await cl.Message(
                content="### Before Data Prep — Raw EEG overview",
                elements=[cl.Image(path=str(before_path), name="before_data_prep", display="inline")],
            ).send()
        else:
            await cl.Message(content="⚠️ The raw before-Data-Prep overview could not be generated.").send()

        # After: reuse the deterministic diagnostic plots produced by Data Prep.
        after_plots = state.get("channel_plot_paths") or {}
        if after_plots:
            await cl.Message(content="### After Data Prep — Diagnostic plots").send()
            for plot_name, plot_path in after_plots.items():
                if plot_path and Path(str(plot_path)).exists():
                    await cl.Message(
                        content=f"#### {plot_name.replace('_', ' ').title()}",
                        elements=[cl.Image(path=str(plot_path), name=f"after_{plot_name}", display="inline")],
                    ).send()
        elif preprocessed:
            await cl.Message(
                content="ℹ️ The completed run has a preprocessed EEG artifact, but no Data Prep diagnostic image was recorded."
            ).send()
        return True

    if wants_plot and (wants_before or wants_after):
        raw_path = state.get("raw_data_path")
        run_id = state.get("run_id", "completed_run")
        if wants_before and raw_path and Path(str(raw_path)).exists():
            before_path = state.get("before_overview_plot")
            if not before_path or not Path(str(before_path)).exists():
                try:
                    result = generate_dataset_overview_plot.invoke({
                        "data_path": str(raw_path),
                        "run_id": f"{run_id}_postrun_before",
                    })
                    before_path = result.get("plot_path") if isinstance(result, dict) else None
                    state["before_overview_plot"] = before_path
                    cl.user_session.set("completed_run_state", state)
                except Exception:
                    before_path = None
            if before_path and Path(str(before_path)).exists():
                await cl.Message(
                    content="### Before Data Prep — Raw EEG overview",
                    elements=[cl.Image(path=str(before_path), name="before_data_prep", display="inline")],
                ).send()
                return True

        if wants_after:
            after_plots = state.get("channel_plot_paths") or {}
            if after_plots:
                await cl.Message(content="### After Data Prep — Diagnostic plots").send()
                for plot_name, plot_path in after_plots.items():
                    if plot_path and Path(str(plot_path)).exists():
                        await cl.Message(
                            content=f"#### {plot_name.replace('_', ' ').title()}",
                            elements=[cl.Image(path=str(plot_path), name=f"after_{plot_name}", display="inline")],
                        ).send()
                return True

    # PSD without a channel: reuse the stored Data Prep PSD overview.
    if any(k in low for k in ("psd", "spectr", "power")):
        psd_plot = (state.get("channel_plot_paths") or {}).get("psd_overview")
        if psd_plot and Path(str(psd_plot)).exists():
            await cl.Message(
                content="### Power Spectral Density — completed run",
                elements=[cl.Image(path=str(psd_plot), name="after_psd_overview", display="inline")],
            ).send()
            return True

    # Conservative heuristic: messages that clearly describe a NEW scientific
    # analysis intent (new band / channel / condition / metric / "compute" etc.)
    # fall through to the normal on_message path so the Query Transformer and
    # Supervisor run a fresh cycle against the same accepted dataset.  Only
    # checked here (after all follow-up branches above have already consumed
    # their matches), so "explain the PLI plot" / "show channel C3" cannot
    # false-positive.
    NEW_INTENT_KEYWORDS: tuple[str, ...] = (
        "analy", "compute", "calculate", "pipeline",
        "fc", "functional connectivity",
        "metric", "pli", "wpli", "coherence", "plv", "imcoh",
        "alpha", "beta", "gamma", "theta", "delta", "band",
        "channel", "condition", "during", "compare",
        "re-analyse", "reanalyze", "recompute", "repeat",
    )
    user_text_lc = user_text.lower()
    if any(kw in user_text_lc for kw in NEW_INTENT_KEYWORDS):
        return False

    # Not an explicit follow-up AND not a clear new scientific intent — give
    # the user the existing hint but with relaxed wording so they know a
    # typed analysis request will also auto-start a fresh cycle.
    chan_text = await _followup_channel_names_text(state)
    await cl.Message(
        content=(
            "### ℹ️ I can answer questions about this completed analysis\n\n"
            "Nothing is re-run when you ask. Try, for example: "
            "`Show me the EEG signal for C3`, `Plot the PSD for C4`, `Compare C3 and C4`, "
            "`Show the dataset before and after Data Prep`, or `Explain the plots generated by the completed analysis`."
            + (f"\n\nChannels in this run: {chan_text}" if chan_text else "")
            + "\n\nOr just type what you want to do next; new analysis requests automatically start a fresh cycle while keeping this dataset loaded. Press **🆕 New query** to force a clean restart."
        )
    ).send()
    return True


async def _offer_new_query_action() -> None:
    """Show a persistent, non-blocking "New query" button under the finished analysis.

    This is a normal message with an action button (NOT a pending ask), so the
    chat input stays free for follow-up questions about the completed run.
    Only clicking the button starts a fresh analysis cycle.
    """
    if cl.user_session.get("post_run_action_active", False):
        return
    cl.user_session.set("post_run_action_active", True)
    await cl.Message(
        content=(
            "**This analysis is complete.** Ask follow-up questions about it any time - "
            "nothing is re-run. One way to restart: press **New query** for a clean cycle; "
            "typing a new analysis request also works and keeps the current dataset loaded."
        ),
        actions=[
            cl.Action(
                name="new_query_button",
                payload={"value": "new_query"},
                label="🆕 New query",
                description="Start a new Supervisor → HITL → Gate 1 → Data Prep run using the current dataset.",
            ),
        ],
    ).send()


@cl.action_callback("new_query_button")
async def _on_new_query_button(action: cl.Action):
    """Explicit unlock: forces a clean restart of a completed run for a fresh analysis.

    Typed new-analysis requests also auto-start a fresh cycle (keeping the
    dataset); this button provides an explicit, discoverable alternative that
    clears `completed_run_state` fully.
    """
    if not cl.user_session.get("completed_run_active", False):
        # Stale button (a new cycle is already active or nothing has completed).
        try:
            await action.remove()
        except Exception:
            pass
        return
    # Preserve the accepted dataset/graph, but clear the previous analysis
    # conversation so the next typed request is the new Supervisor input.
    _reset_conversation_state()
    cl.user_session.set("completed_run_active", False)
    cl.user_session.set("completed_run_state", None)
    cl.user_session.set("post_run_action_active", False)
    try:
        await action.remove()
    except Exception:
        pass
    await cl.Message(
        content="🆕 **New query enabled.** Type the new EEG analysis request to start the Supervisor workflow."
    ).send()


def _reset_conversation_state():
    """Reset conversational accumulation and gate state so next message starts clean."""
    cl.user_session.set("original_query", None)
    cl.user_session.set("accumulated_query", None)
    cl.user_session.set("pending_clarification_run_id", None)
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
        attention_lines.append("> ⚠️ **Parameters Requiring Review**:")
        for fe in flagged:
            attention_lines.append(f"> - **`{fe.name}`**: {mask_text(str(fe.proposed_value))}")
        attention_lines.append("")

    body = [
        "## 📋 Analysis Plan & Parameters",
        "",
        format_plan_markdown(plan),
        "### Parameters",
        "",
        format_manifest_markdown(manifest),
    ]
    if attention_lines:
        body.extend(attention_lines)

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





async def _handle_rail_decision(
    output_state: Dict[str, Any],
    run_id: str,
    msg: cl.Message,
):
    """Presents a clean guardrail block message with no scenario ID or internal info."""
    _reset_conversation_state()
    msg.content = "🛡️ Guardrail: Cannot process this query."
    await msg.update()


def _manifest_value(manifest: List[ParameterManifestEntry], name: str) -> Optional[str]:
    for entry in manifest:
        if getattr(entry, "name", None) == name:
            return str(getattr(entry, "human_approved_value", None) or getattr(entry, "proposed_value", ""))
    return None


def _format_data_prep_explanation(state: Dict[str, Any], manifest: List[ParameterManifestEntry]) -> List[str]:
    """Build a deterministic explanation from Data Prep state and approved manifest values."""
    lines: List[str] = ["", "### 🔎 What Data Preparation Did", ""]
    dropped = state.get("bad_channels_dropped") or []
    variance = _manifest_value(manifest, "bad_channel_variance_threshold")
    reference = _manifest_value(manifest, "reference")
    if dropped:
        reason = f" using the configured variance threshold `{mask_text(variance)}`" if variance else " using the configured bad-channel screening rule"
        lines.append(f"- **Bad-channel screening:** removed {', '.join(f'`{mask_text(c)}`' for c in dropped)}{reason}.")
    else:
        lines.append("- **Bad-channel screening:** no channels were removed by the deterministic screening stage.")
    if reference:
        lines.append(f"- **Reference:** applied `{mask_text(reference)}` as specified by the approved manifest.")
    else:
        lines.append("- **Reference:** followed the approved preprocessing configuration; no separate reference override was recorded in the manifest.")
    plots = state.get("channel_plot_paths") or {}
    if plots:
        names = ", ".join(name.replace("_", " ").title() for name in plots)
        lines.append(f"- **Diagnostics:** generated {len(plots)} channel-level diagnostic plot(s): {names}.")
        lines.append("  These plots are evidence for the deterministic preprocessing checks; they are not an LLM interpretation.")
    summary = state.get("data_prep_summary")
    if summary:
        sfreq = getattr(summary, "sampling_frequency", None)
        epoch_count = getattr(summary, "epoch_count", None)
        epoch_duration = getattr(summary, "epoch_duration_seconds", None)
        f_low = getattr(summary, "filter_l_freq", None)
        f_high = getattr(summary, "filter_h_freq", None)
        if sfreq is not None:
            lines.append(f"- **Recording:** processed at `{sfreq:g} Hz`; `{getattr(summary, 'retained_channel_count', '—')}` channel(s) remained after screening.")
        if f_low is not None and f_high is not None:
            lines.append(f"- **Filtering:** applied the approved `{f_low:g}–{f_high:g} Hz` band before epoching.")
        if epoch_count is not None:
            duration_text = f" of `{epoch_duration:g}s` each" if epoch_duration is not None else ""
            lines.append(f"- **Epoching:** created `{epoch_count}` epoch(s){duration_text} for the selected condition.")
        if getattr(summary, "reference_applied", None):
            lines.append(f"- **Reference result:** `{mask_text(str(summary.reference_applied))}`.")
    output = state.get("preprocessed_data_path")
    if output:
        lines.append("- **Output:** the preprocessed EEG epochs were written for the downstream connectivity stage.")
    return lines


async def _resume_clarification(
    run_id: str,
    reply: str,
    accumulated_query: str,
    msg: Optional[cl.Message] = None,
) -> None:
    """Resume the SAME LangGraph checkpoint with a validated-by-graph reply.

    The graph's ``clarification_pause`` validates/normalises/persists the value
    (band / channels / condition), the Supervisor then skips resolved axes and
    asks only for the next unresolved one (which comes back through
    ``_handle_pipeline_output`` -> ``_present_clarification``).
    """
    cl.user_session.set("awaiting_clarification", False)
    cl.user_session.set("awaiting_transformer_clarification", False)
    cl.user_session.set("pending_clarification_run_id", None)
    new_accumulated = f"{accumulated_query}\n[User clarification reply]: {reply}"
    cl.user_session.set("accumulated_query", new_accumulated)

    resume_msg = cl.Message(content="🔄 Applying your clarification…")
    await resume_msg.send()
    resume_queue: asyncio.Queue = asyncio.Queue()
    resume_done = asyncio.Event()
    resume_event_task = asyncio.create_task(_stream_supervisor_events(resume_queue, resume_done))
    resume_loop = asyncio.get_running_loop()

    def _resume_event_sink(event: Dict[str, Any]):
        try:
            resume_loop.call_soon_threadsafe(resume_queue.put_nowait, event)
        except Exception:
            pass

    try:
        resume_action = {"reply": reply, "value": reply, "action": "clarification"}
        new_state, new_error = await cl.make_async(_resume_pipeline_sync)(
            run_id, resume_action, _resume_event_sink
        )
    except Exception as exc:
        new_state, new_error = {}, f"{type(exc).__name__}: {exc}"
    finally:
        await resume_queue.put(None)
        await resume_event_task

    if new_error:
        resume_msg.content = f"❌ Clarification resume failed: {mask_text(new_error)}"
        await resume_msg.update()
        return
    await _handle_pipeline_output(
        new_state, new_state.get("_executed_nodes", []), new_accumulated, run_id, resume_msg
    )


async def _present_clarification(
    output_state: Dict[str, Any],
    clarification: Dict[str, Any],
    accumulated_query: str,
    run_id: str,
    msg: cl.Message,
) -> None:
    """Render an unresolved clarification as clickable HITL controls.

    Bounded options (from the authoritative payload) + ``Type manually``.  If the
    action UI cannot be shown or times out, the question is still displayed and
    the pending checkpoint is remembered, so the user's next typed message resumes
    the SAME checkpoint (never a silent dead end).
    """
    kind = clarification.get("kind") or "clarification"
    if kind == "query_contradiction":
        cl.user_session.set("awaiting_transformer_clarification", True)
    else:
        cl.user_session.set("awaiting_clarification", True)
    cl.user_session.set("pending_clarification_run_id", run_id)

    question_text = clarification.get("question") or "Please provide the missing analysis parameter(s)."
    if kind != "query_contradiction":
        question_text = f"### ❓ Clarification Required\n\n{question_text}"
    msg.content = mask_text(question_text)
    await msg.update()

    actions = _build_clarification_actions(output_state)
    try:
        choice = await cl.AskActionMessage(
            content="Choose a suggested value, or use **Type manually** for an open-ended response:",
            actions=actions,
            timeout=300,
        ).send()
    except Exception as exc:
        logger.warning("Clarification action UI failed: %s: %s", type(exc).__name__, exc)
        choice = None

    if choice is None:
        await cl.Message(
            content="⏳ Waiting for your answer. Type it in the chat box to continue this same analysis."
        ).send()
        return

    chosen_name = choice.get("name") if isinstance(choice, dict) else getattr(choice, "name", None)
    payload = choice.get("payload", {}) if isinstance(choice, dict) else getattr(choice, "payload", {})

    if chosen_name == "new_query" or (payload or {}).get("value") == "new query:":
        _reset_conversation_state()
        cl.user_session.set("pending_clarification_run_id", None)
        await cl.Message(
            content="↻ **New query started.** Send the new EEG analysis request when ready."
        ).send()
        return

    if chosen_name == "manual_clarification":
        manual = await cl.AskUserMessage(content="Type your clarification:", timeout=300).send()
        reply = (
            manual.get("output", "") if isinstance(manual, dict) else getattr(manual, "output", "")
        ).strip() if manual else ""
        if not reply:
            await cl.Message(
                content="⏳ Waiting for your answer. Type it in the chat box to continue this same analysis."
            ).send()
            return
    else:
        reply = str((payload or {}).get("value", "")).strip()

    if reply:
        await _resume_clarification(run_id, reply, accumulated_query)


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
        body = [
            "## 🛑 Processing Error",
            "",
            "An error occurred while processing your request. Please try again or rephrase.",
        ]
        if pipe_err:
            body.append("")
            body.append(f"```\n{mask_text(pipe_err)}\n```")
        msg.content = "\n".join(body)
        await msg.update()
        return

    clarification = get_clarification(output_state)
    if clarification and (
        "clarification_pause" in routed_node
        or ("gate_1_review" not in routed_node and "informational_complete" not in routed_node)
    ):
        await _present_clarification(
            output_state, clarification, accumulated_query, run_id, msg
        )
        return

    if "informational_complete" in routed_node:
        cl.user_session.set("awaiting_clarification", False)
        info_text = output_state.get("informational_response") or "Request completed."
        msg.content = mask_text(info_text)
        await msg.update()

        informational_artifacts = output_state.get("informational_artifacts") or {}
        overview_obs = informational_artifacts.get("generate_dataset_overview_plot") or {}
        overview_plot_value = overview_obs.get("plot_path") if isinstance(overview_obs, dict) else None
        if overview_plot_value and Path(overview_plot_value).exists():
            image = cl.Image(
                path=str(overview_plot_value),
                name="Dataset Overview",
                display="inline",
            )
            await cl.Message(
                content="### Dataset Overview Plot",
                elements=[image],
            ).send()
        elif overview_plot_value:
            await cl.Message(content="⚠️ The overview plot was reported by the tool but the file is no longer present.").send()
        return

    if "gate_1_review" in routed_node:
        cl.user_session.set("awaiting_clarification", False)

        # Generalized gate: a NeMo rail pause carries a decision_context and is
        # presented through the same AskActionMessage/resume mechanics.
        if (output_state.get("decision_context") or {}).get("kind") in ("input_rail", "output_rail"):
            await _handle_rail_decision(output_state, run_id, msg)
            return

        # --- Gate 1 HITL loop (edit → regenerate artifact → re-render → action) ---
        # Uses a while loop so multiple sequential edits stay in the same
        # turn without recursive calls into _handle_pipeline_output.
        while True:
            # Regenerate the Markdown artifact on every Gate 1 render so any
            # human edits are reflected in the openable document.
            try:
                plan_md_path = write_analysis_plan_markdown(plan, manifest, run_id)
                plan_text = cl.Text(
                    name="analysis_plan.md",
                    content=plan_md_path.read_text(encoding="utf-8"),
                    display="side",
                )
                plan_file = cl.File(name="analysis_plan.md", path=str(plan_md_path), display="inline")
                await cl.Message(
                    content="### 📄 Analysis Plan\nOpen the generated Markdown document to review the exact Supervisor plan and manifest.",
                    elements=[plan_text, plan_file],
                ).send()
            except Exception as exc:
                logger.warning("Could not create plan Markdown artifact: %s", exc)

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
                    # Gate 1 Approve is the explicit human approval for the
                    # concrete manifest currently shown in the review UI.
                    # Materialize that approval on any low-confidence row before
                    # resuming the graph, so Data Prep consumes the reviewed
                    # manifest rather than the stale pre-review checkpoint copy.
                    _mark_manifest_human_approved(manifest)

                    # Resume LangGraph checkpoint — Data Prep now executes within this stream
                    pipeline_app = cl.user_session.get("graph")
                    config = {"configurable": {"thread_id": run_id}}
                    dp_error = None
                    final_state: Dict[str, Any] = {}

                    # Data Prep emits the same live Chainlit trace used by the
                    # Supervisor: each deterministic stage appears as it runs.
                    event_queue: asyncio.Queue = asyncio.Queue()
                    event_done = asyncio.Event()
                    event_task = asyncio.create_task(_stream_supervisor_events(event_queue, event_done))
                    loop = asyncio.get_running_loop()

                    def _event_sink(event: Dict[str, Any]):
                        try:
                            loop.call_soon_threadsafe(event_queue.put_nowait, event)
                        except Exception:
                            pass

                    try:
                        final_state, dp_error = await cl.make_async(_resume_pipeline_sync)(
                            run_id, "approve", _event_sink, approved_manifest=manifest
                        )
                    finally:
                        await event_queue.put(None)
                        await event_task

                    # Re-log the final manifest (with any human_approved_value edits) to MLflow
                    try:
                        mlflow_tracker.log_manifest(manifest)
                    except Exception as e:
                        logger.warning(
                            "MLflow log_manifest failed during Gate 1 Approve (post-edit artifact): %s: %s",
                            type(e).__name__, e,
                        )
                    # Extract Data Prep results from final state before changing session mode.
                    dp_error = dp_error or final_state.get("data_prep_error")
                    dp_dropped = final_state.get("bad_channels_dropped") or []
                    dp_plots = final_state.get("channel_plot_paths") or {}
                    dp_output = final_state.get("preprocessed_data_path")
                    dp_summary = final_state.get("data_prep_summary") or {}
                    dp_manifest = final_state.get("parameter_manifest") or manifest or []

                    # Preserve the completed run as a read-only artifact context.
                    # Ordinary follow-up messages must not restart Supervisor/Data Prep.
                    cl.user_session.set("completed_run_active", dp_error is None and bool(dp_output))
                    cl.user_session.set("completed_run_state", {
                        "run_id": run_id,
                        "raw_data_path": final_state.get("raw_data_path") or output_state.get("raw_data_path") or cl.user_session.get("data_path"),
                        "preprocessed_data_path": dp_output,
                        "channel_plot_paths": dict(dp_plots),
                        "bad_channels_dropped": list(dp_dropped),
                        "data_prep_summary": dp_summary,
                        "parameter_manifest": dp_manifest,
                        "plan": plan,
                        "informational_artifacts": final_state.get("informational_artifacts") or output_state.get("informational_artifacts") or {},
                    } if dp_error is None and dp_output else None)
                    _reset_conversation_state()
                    cl.user_session.set("gate_1_approved", True)
                    cl.user_session.set("preflight_confirmed", True)
                    output_state["gate_1_approved"] = True
                    output_state["preflight_confirmed"] = True

                    # Build the result message
                    result_lines = [
                        "### ✅ Gate 1 Approved",
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
                            "- **Status**: Preprocessing successful ✓",
                        ])
                        result_lines.extend(_format_data_prep_explanation(final_state, dp_manifest))
                        if dp_dropped:
                            result_lines.append(
                                f"- **Bad channels removed**: {', '.join(f'`{c}`' for c in dp_dropped)}"
                            )
                        else:
                            result_lines.append("- **Bad channels removed**: 0 (all channels healthy ✓)")
                        if dp_plots:
                            plot_names = ", ".join(pname.replace("_", " ").title() for pname in dp_plots.keys())
                            result_lines.append(f"- **Diagnostic plots**: {len(dp_plots)} generated ({plot_names})")
                        result_lines.append("")
                    else:
                        result_lines.extend([
                            "### ⚠️ Data Preparation",
                            "",
                            "Data Preparation executed but produced no output path.",
                            "",
                        ])

                    await cl.Message(content="\n".join(result_lines)).send()

                    # Display diagnostic plot images inline. A rendering/serialization
                    # error here must not kill the whole pipeline: fall back to a
                    # short textual hint per-plot so the user still sees "what was
                    # produced" and can ask follow-ups (e.g. "plot channel variance")
                    # to re-render. This also guards against Chainlit's internal
                    # "generator didn't stop after throw()" async generator error.
                    for plot_name, plot_path in dp_plots.items():
                        if Path(plot_path).exists():
                            image = cl.Image(
                                path=plot_path,
                                name=plot_name,
                                display="inline",
                            )
                            try:
                                await cl.Message(
                                    content=f"### {plot_name.replace('_', ' ').title()}",
                                    elements=[image],
                                ).send()
                            except (GeneratorExit, RuntimeError, Exception) as e:
                                logger.warning(
                                    "Failed to inline diagnostic image %r: %s",
                                    plot_name, e,
                                )
                                try:
                                    await cl.Message(
                                        content=(
                                            f"### {plot_name.replace('_', ' ').title()}\n\n"
                                            "⚠️ Rendering the inline image failed. Say "
                                            f"`Plot the {plot_name.replace('_', ' ')}` and I'll re-generate it."
                                        )
                                    ).send()
                                except (GeneratorExit, RuntimeError, Exception):
                                    pass

                    if dp_error is None and dp_output:
                        # Non-blocking: users may ask post-run inspection questions immediately.
                        # The explicit button is the only action that unlocks a fresh pipeline run.
                        asyncio.create_task(_offer_new_query_action())

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
                        content=f"🔄 Updating `{mask_text(entry_name)}` to: *'{mask_text(edit_value)}'*…"
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
                    content=f"🔄 Updating plan with: *'{mask_text(feedback_text)}'*…"
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
        "## ⚠️ Request Could Not Be Processed",
        "",
        "Please provide more specific parameters (e.g. frequency band, channels, condition).",
    ]
    cl.user_session.set("awaiting_clarification", False)
    msg.content = "\n".join(fallback)
    await msg.update()
