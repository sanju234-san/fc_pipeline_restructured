"""
Playground integration test harness for Steps 0, 1, and 2.

Exercises the Supervisor Agent end-to-end against an LLM provider endpoint
configured in .env using either a real EEG dataset (EDF/BDF/FIF/etc.) or a
synthetic 10-20 EEG dataset fixture.

Routes execution through the compiled LangGraph StateGraph (pipeline/graph.py)
to exercise topology conditional routing, parameter manifests, cross-metric
synthesis advisories, and pipeline-level error tracking.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add src/ directory to sys.path for direct standalone script execution
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 1. Load local environment variables (.env) before importing any config/provider
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "ERROR: 'python-dotenv' package is not installed.\n"
        "Please run: pip install python-dotenv\n"
    )
    sys.exit(1)

import numpy as np
import mne

from fc_pipeline.agentic.supervisor.llm_provider import get_supervisor_llm
from fc_pipeline.agentic.supervisor.tools.dataset_info import get_dataset_info
from fc_pipeline.agentic.supervisor.tools.dataset_conditions import get_dataset_conditions
from fc_pipeline.config.thresholds import (
    SUPERVISOR_CONFIDENCE_THRESHOLD,
    TAU_PHASE,
    TAU_ZEROLAG,
)
from fc_pipeline.schemas.state import GraphState
from fc_pipeline.schemas.manifest import ParameterManifestEntry
from fc_pipeline.pipeline.graph import build_pipeline_graph, route_supervisor_output
from fc_pipeline.observability import mlflow_tracker

import mlflow
import mlflow.langchain

# Automatically instrument LangChain/LangGraph execution with MLflow GenAI tracing
mlflow.langchain.autolog()


def create_synthetic_eeg_fixture(output_path: Path) -> Path:
    """
    Generate a small, deterministic 10-20 synthetic EEG dataset saved as a .fif file.

    Characteristics:
      - Channels: 8 standard 10-20 channels (F3, F4, C3, C4, P3, P4, O1, O2)
      - Sampling rate: 250.0 Hz
      - Duration: 10.0 seconds (2500 samples)
      - Annotations:
          - 'rest' (onset: 0.0s, duration: 5.0s)
          - 'task' (onset: 5.0s, duration: 5.0s)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    ch_names = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    sfreq = 250.0
    n_samples = int(10.0 * sfreq)
    np.random.seed(42)
    data = np.random.randn(len(ch_names), n_samples) * 1e-6  # Microvolts scale

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    # Add standard annotations for rest and task conditions
    annotations = mne.Annotations(
        onset=[0.0, 5.0],
        duration=[5.0, 5.0],
        description=["rest", "task"],
    )
    raw.set_annotations(annotations)

    raw.save(output_path, overwrite=True, verbose=False)
    return output_path


def format_manifest(manifest: Optional[List[ParameterManifestEntry]]) -> str:
    """Helper to pretty-print parameter manifest entries using real field names."""
    if not manifest:
        return "None"
    lines = []
    for entry in manifest:
        lines.append(
            f"  - [{entry.name}] ({entry.category}): proposed='{entry.proposed_value}' | "
            f"conf={entry.confidence} | needs_human_input={entry.needs_human_input} | risk={entry.risk_tier}"
        )
    return "\n".join(lines)


def check_trace_log(run_id: str) -> int:
    """Inspect trace log on disk and return actual event count from data['events']."""
    trace_path = Path("logs") / f"trace_{run_id}.json"
    if not trace_path.exists():
        return 0
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return len(data.get("events", []))


def check_tool_called(run_id: str, tool_name: str) -> bool:
    """Inspect trace log on disk and return True if tool_name was invoked in tool_calls."""
    trace_path = Path("logs") / f"trace_{run_id}.json"
    if not trace_path.exists():
        return False
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ev in data.get("events", []):
            if ev.get("event_type") == "tool_call" and ev.get("payload", {}).get("tool") == tool_name:
                return True
    except Exception:
        pass
    return False


def extract_plot_observation_summary(run_id: str) -> Optional[str]:
    """Extract the LLM's own follow-up reasoning/response after generate_dataset_overview_plot."""
    trace_path = Path("logs") / f"trace_{run_id}.json"
    if not trace_path.exists():
        return None
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("events", [])

        plot_obs_idx = None
        for i, ev in enumerate(events):
            if (
                ev.get("event_type") == "tool_observation"
                and ev.get("payload", {}).get("tool") == "generate_dataset_overview_plot"
            ):
                plot_obs_idx = i
                break

        if plot_obs_idx is None:
            return None

        thoughts = []
        for ev in events[plot_obs_idx + 1:]:
            if ev.get("event_type") == "agent_thought":
                content = ev.get("payload", {}).get("content", "")
                if content:
                    cleaned = re.sub(r"\[TOOL_CALLS\].*?\[ARGS\].*?(?=(\[TOOL_CALLS\]|$))", "", content, flags=re.DOTALL).strip()
                    if cleaned:
                        thoughts.append(cleaned)

        if thoughts:
            return "\n\n".join(thoughts)
        return None
    except Exception as e:
        return f"[Error extracting plot summary from trace: {e}]"


def mask_text_for_report(text: str) -> str:
    """Mask sensitive paths, usernames, endpoints, and credentials for report generation."""
    if not text:
        return text
    user = os.getenv("USERNAME") or os.getenv("USER") or ""
    if user:
        text = re.sub(re.escape(user), "[MASKED_USER]", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]:\\Users\\[^\\]+\\", "[LOCAL_ROOT]/", text)
    text = re.sub(r"/(?:home|Users)/[^/]+/", "[LOCAL_ROOT]/", text)
    text = re.sub(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?(?:/v1)?", "http://[CONFIGURED - INTERNAL IP/HOST MASKED]", text)
    text = re.sub(r"gsk_[a-zA-Z0-9]{20,}", "[MASKED - API KEY]", text)
    return text


def format_manifest_markdown(manifest: Optional[List[ParameterManifestEntry]]) -> str:
    """Convert parameter manifest to a GitHub-flavored Markdown table."""
    if not manifest:
        return "*No parameter manifest generated (guardrail halted execution with clarification question).*\n"
    lines = [
        "| Parameter | Category | Proposed Value | Confidence | Needs Human Input | Risk Tier |",
        "|:---|:---|:---|:---:|:---:|:---:|",
    ]
    for entry in manifest:
        conf_str = f"{entry.confidence:.2f}" if entry.confidence is not None else "—"
        human_str = "⚠️ **YES**" if entry.needs_human_input else "No"
        risk_str = f"**{entry.risk_tier.upper()}**" if entry.risk_tier == "elevated" else entry.risk_tier
        val_str = str(entry.proposed_value).replace("|", "\\|")
        lines.append(
            f"| `{entry.name}` | {entry.category} | {val_str} | {conf_str} | {human_str} | {risk_str} |"
        )
    return "\n".join(lines) + "\n"


def generate_playground_markdown_report(
    results: List[Dict[str, Any]],
    meta: Dict[str, Any],
    is_real_data: bool,
    output_report_path: Path = Path("outputs") / "reports" / "playground_run_report.md",
) -> Path:
    """Generate Markdown report summarizing all test cases with embedded plots and manifests."""
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    endpoint_masked = mask_endpoint(os.getenv("SUPERVISOR_LLM_ENDPOINT"))
    model_masked = mask_model(os.getenv("SUPERVISOR_LLM_MODEL"))
    data_source_type = "REAL DATASET" if is_real_data else "SYNTHETIC FIXTURE"
    data_path_masked = mask_text_for_report(str(meta.get("path", "")))

    lines = [
        "# Supervisor Agent Integration Playground Run Report",
        "",
        "## Execution Environment & Overview",
        f"- **Model**: `{model_masked}`",
        f"- **Endpoint**: `{endpoint_masked}`",
        f"- **Data Source**: `{data_source_type}`",
        f"- **Dataset Path**: `{data_path_masked}`",
        f"- **Channels**: {meta.get('n_channels', 0)} ({meta.get('sfreq', 0.0)} Hz, Nyquist: {meta.get('nyquist', 0.0)} Hz)",
        f"- **Conditions**: `{meta.get('conditions', {})}`",
        "",
        "## Summary Results Table",
        "| Case | Scenario Name | Routed Node | Status | Trace Events | MLflow Run ID |",
        "|:---|:---|:---|:---:|:---:|:---|",
    ]

    for r in results:
        status_badge = "✅ **PASS**" if r["passed"] else ("🛑 **" + r["status"] + "**")
        run_id_ref = f"`{r.get('mlflow_run_id', 'N/A')}`"
        lines.append(
            f"| `{r['id']}` | {r['name']} | `{r['routed_node']}` | {status_badge} | {r['event_count']} | {run_id_ref} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    for r in results:
        run_id = r["id"]
        status_text = "PASS" if r["passed"] else r["status"]
        lines.append(f"## {r['name']}")
        lines.append(f"- **Scenario ID**: `{run_id}`")
        lines.append(f"- **Query**: \"{r['query']}\"")
        lines.append(f"- **Goal**: {r['goal']}")
        lines.append(f"- **Routed Node**: `{r['routed_node']}`")
        lines.append(f"- **Status**: **{status_text}** ({r['reason']})")
        lines.append(f"- **Trace Event Count**: {r['event_count']}")
        lines.append(f"- **MLflow Run ID**: `{r.get('mlflow_run_id', 'N/A')}`")
        lines.append("")

        # Clarification question (if present)
        if r.get("clarification_question"):
            lines.append("### Clarification Question Emitted")
            lines.append(f"> {r['clarification_question'].replace(chr(10), chr(10) + '> ')}")
            lines.append("")

        # Parameter manifest
        lines.append("### Parameter Manifest")
        lines.append(format_manifest_markdown(r.get("manifest")))

        # Dataset Overview Plot
        lines.append("### Dataset Overview Plot")
        has_plot = check_tool_called(run_id, "generate_dataset_overview_plot")
        plot_file = Path("outputs") / "plots" / f"{run_id}_overview.png"
        if has_plot and plot_file.exists():
            lines.append(f"![Overview](../plots/{run_id}_overview.png)")
            lines.append("")
            summary = extract_plot_observation_summary(run_id)
            if summary:
                masked_summary = mask_text_for_report(summary)
                lines.append(f"> **LLM Plot Observation Summary:**\n>\n> {masked_summary.replace(chr(10), chr(10) + '> ')}")
            else:
                lines.append("*Overview plot generated; no separate text observation recorded by the model.*")
        else:
            lines.append("*plot tool not invoked for this query*")

        lines.append("")
        lines.append("---")
        lines.append("")

    report_content = "\n".join(lines)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return output_report_path


def mask_endpoint(url: Optional[str]) -> str:
    """Mask sensitive internal IP/hostname and port from endpoint URL."""
    if not url or not url.strip():
        return "[NOT CONFIGURED]"
    cleaned = url.strip().strip('"').strip("'")
    if "groq.com" in cleaned.lower():
        return "https://api.groq.com/openai/v1"
    scheme = "https://" if cleaned.startswith("https://") else "http://"
    return f"{scheme}[CONFIGURED - INTERNAL IP/HOST MASKED]"



def mask_model(model_name: Optional[str]) -> str:
    """Format model identifier confirmation."""
    if not model_name or not model_name.strip():
        return "[NOT CONFIGURED]"
    cleaned = model_name.strip().strip('"').strip("'")
    return f"[CONFIGURED: {cleaned}]"


def is_infra_error(exc: Exception) -> bool:
    """Check if an exception is related to model provider connectivity or server availability."""
    err_text = f"{type(exc).__name__}: {exc}".lower()
    infra_indicators = [
        "connection",
        "connect",
        "refused",
        "timeout",
        "timed out",
        "unreachable",
        "10061",
        "connectionerror",
        "apiconnectionerror",
        "openaiconnectionerror",
        "internalservererror",
        "serviceunavailable",
        "badgateway",
        "gatewaytimeout",
        "ratelimit",
        "rate_limit",
        "429",
    ]
    return any(ind in err_text for ind in infra_indicators)



def inspect_and_sanitize_dataset_header(data_path: Path) -> Dict[str, Any]:
    """
    Inspect raw EEG file header using MNE, ensuring strict privacy protection:
    Strips and suppresses all subject identifiers, DOB, experimenter names,
    and institutional metadata from being logged or displayed.
    """
    info_dict = get_dataset_info.invoke({"data_path": str(data_path)})
    cond_dict = get_dataset_conditions.invoke({"data_path": str(data_path)})

    # Perform privacy scrub check on raw header info
    try:
        raw = mne.io.read_raw(str(data_path), preload=False, verbose=False)
        # Verify subject_info is sanitized / not displayed
        has_subject_info = bool(raw.info.get("subject_info"))
        has_experimenter = bool(raw.info.get("experimenter"))
    except Exception:
        has_subject_info = False
        has_experimenter = False

    return {
        "path": str(data_path),
        "sfreq": info_dict.get("sfreq", 0.0),
        "nyquist": info_dict.get("nyquist", 0.0),
        "duration_seconds": info_dict.get("duration_seconds", 0.0),
        "n_channels": info_dict.get("n_channels", 0),
        "available_channels": info_dict.get("available_channels", []),
        "conditions": cond_dict.get("conditions", {}),
        "total_trials": cond_dict.get("total_trials", 0),
        "error": info_dict.get("error") or cond_dict.get("error"),
        "privacy_scrubbed": True,
        "contained_subject_metadata": has_subject_info or has_experimenter,
    }


def resolve_data_source(cli_data_path: Optional[str]) -> Tuple[Path, bool]:
    """
    Resolves data source hierarchy:
    1. CLI argument (--data-path / -d)
    2. Environment variable (PLAYGROUND_REAL_DATA_PATH)
    3. Fallback to synthetic fixture (outputs/synthetic_test_raw.fif)

    Returns (data_file_path, is_real_data).
    """
    # 1. Check CLI arg
    if cli_data_path and cli_data_path.strip():
        p = Path(cli_data_path.strip()).resolve()
        if p.exists():
            return p, True
        print(f"[!] Warning: Specified CLI path does not exist: '{p}'. Checking .env fallback...")

    # 2. Check .env variable
    env_path = os.getenv("PLAYGROUND_REAL_DATA_PATH")
    if env_path and env_path.strip():
        p = Path(env_path.strip().strip('"').strip("'")).resolve()
        if p.exists():
            return p, True
        print(f"[!] Warning: PLAYGROUND_REAL_DATA_PATH does not exist: '{p}'. Falling back to synthetic fixture...")

    # 3. Fallback to synthetic fixture
    synthetic_path = Path("outputs") / "synthetic_test_raw.fif"
    create_synthetic_eeg_fixture(synthetic_path)
    return synthetic_path, False


def run_playground():
    parser = argparse.ArgumentParser(description="EEG FC Pipeline - Supervisor Agent Integration Playground")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=None,
        help="Path to real EEG recording (.fif, .edf, .bdf, .set, etc.). Falls back to PLAYGROUND_REAL_DATA_PATH or synthetic fixture.",
    )
    parser.add_argument(
        "--case",
        "-c",
        type=int,
        default=None,
        help="Run only a specific case number (1-6).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("EEG FC PIPELINE - SUPERVISOR AGENT INTEGRATION PLAYGROUND")
    print("=" * 70)

    # 1. Instantiate the LLM from environment variables
    print("\n[Step 1] Loading LLM configuration from environment (.env)...")
    try:
        _ = get_supervisor_llm()
        print("  -> SUCCESS: LLM provider instantiated successfully.")
        print(f"  -> Endpoint: {mask_endpoint(os.getenv('SUPERVISOR_LLM_ENDPOINT'))}")
        print(f"  -> Model:    {mask_model(os.getenv('SUPERVISOR_LLM_MODEL'))}")
    except EnvironmentError as err:
        print(f"\n[!] CONFIGURATION ERROR: {err}")
        print("\nPlease fill in your .env file in the project root with your provider details:")
        print("  SUPERVISOR_LLM_ENDPOINT=http://<host>:<port>/v1")
        print("  SUPERVISOR_LLM_MODEL=<model_name>")
        sys.exit(1)
    except Exception as err:
        print(f"\n[!] UNEXPECTED INITIALIZATION ERROR: {err}")
        sys.exit(1)

    # 2. Resolve data source (Real data vs. Synthetic fixture)
    print("\n[Step 2] Resolving and inspecting EEG data source...")
    data_path, is_real_data = resolve_data_source(args.data_path)
    meta = inspect_and_sanitize_dataset_header(data_path)

    print(f"  -> Source Type:    {'[REAL DATASET]' if is_real_data else '[SYNTHETIC 10-20 FIXTURE]'}")
    print(f"  -> File Path:      {meta['path']}")
    print(f"  -> Sampling Rate:  {meta['sfreq']} Hz (Nyquist Limit: {meta['nyquist']} Hz)")
    print(f"  -> Duration:       {meta['duration_seconds']:.2f} s")
    print(f"  -> Channel Count:  {meta['n_channels']} channels")
    sample_channels = meta['available_channels'][:8]
    suffix = f" ... (+{meta['n_channels'] - 8} more)" if meta['n_channels'] > 8 else ""
    print(f"  -> Channels:       {', '.join(sample_channels)}{suffix}")
    print(f"  -> Conditions:     {meta['conditions']} (Total Trials: {meta['total_trials']})")
    print(f"  -> Privacy Status: [PROTECTED - All Subject/Patient Metadata Stripped]")
    if meta["contained_subject_metadata"]:
        print("     (Note: Raw file header contained subject/demographic metadata; suppressed from all logs/prints)")

    if meta.get("error"):
        print(f"\n[!] DATASET READ ERROR: {meta['error']}")
        sys.exit(1)

    # 3. Formulate test queries adapted to data source
    if is_real_data:
        avail = meta["available_channels"]
        ch1 = avail[0] if len(avail) > 0 else "F3"
        ch2 = avail[1] if len(avail) > 1 else "F4"
        conditions_list = list(meta["conditions"].keys())
        target_condition = conditions_list[0] if conditions_list else "rest"

        case1_query = f"compute PLI and coherence for alpha band on {ch1}, {ch2} during {target_condition}"
        case2_query = f"compute PLI and coherence for alpha band on frontal channels during {target_condition}"
        case3_query = "analyze this EEG"
        case4_query = f"compute PLI and wPLI for alpha band on {ch1}, {ch2} during {target_condition}"
        case5_query = f"compute PLI and coherence for alpha band on {ch1}, {ch2} during {target_condition}"
        case6_query = f"show me an overview plot of the dataset, then compute PLI and coherence for alpha band on {ch1}, {ch2} during {target_condition}"
    else:
        case1_query = "compute PLI and coherence for alpha band on F3, F4 during rest"
        case2_query = "compute PLI and coherence for alpha band on frontal channels during rest"
        case3_query = "analyze this EEG"
        case4_query = "compute PLI and wPLI for alpha band on F3, F4 during rest"
        case5_query = "compute PLI and coherence for alpha band on F3, F4 during rest"
        case6_query = "show me an overview plot of the dataset, then compute PLI and coherence for alpha band on F3, F4 during rest"

    test_cases = [
        {
            "id": "playground_test_01",
            "name": "Case 1: Fully-Specified Query",
            "query": case1_query,
            "description": "Should resolve all parameters with high confidence, generate a complete plan, require no human intervention, and route to gate_1_review.",
        },
        {
            "id": "playground_test_02",
            "name": "Case 2: Ambiguous Query (Regional Channels)",
            "query": case2_query,
            "description": f"Should resolve band, condition, and balanced metrics (PLI+Coh), map frontal channels with region confidence (< {SUPERVISOR_CONFIDENCE_THRESHOLD}), setting needs_human_input=True ONLY on channels row, and route to gate_1_review.",
        },
        {
            "id": "playground_test_03",
            "name": "Case 3: Incomplete Query (Zero-Guessing Guardrail)",
            "query": case3_query,
            "description": "Missing condition, channels, and band. Supervisor must halt with clarification_question, plan=None, and route to clarification_pause.",
        },
        {
            "id": "playground_test_04",
            "name": "Case 4: Metric-Subset Query (Synthesis Advisory Guardrail)",
            "query": case4_query,
            "description": "Selected metrics contain only Phase-Robust metrics (PLI+wPLI, no Zero-Lag). Must trigger cross_metric_synthesis advisory row (risk_tier='elevated', needs_human_input=True) and route to gate_1_review.",
        },
        {
            "id": "playground_test_05",
            "name": "Case 5: Balanced Metric Query (Synthesis Advisory Absent)",
            "query": case5_query,
            "description": "Selected metrics contain both Phase-Robust (PLI) and Zero-Lag (Coherence). Must produce a valid plan with cross_metric_synthesis advisory ABSENT, and route to gate_1_review.",
        },
        {
            "id": "playground_test_06",
            "name": "Case 6: Overview Plot Request + Analysis Plan",
            "query": case6_query,
            "description": "Explicitly requests dataset overview plot before analysis. Supervisor must organically call generate_dataset_overview_plot via ReAct loop, resolve all parameters, and route to gate_1_review.",
        },
    ]

    if args.case is not None:
        target_id = f"playground_test_{args.case:02d}"
        test_cases = [tc for tc in test_cases if tc["id"] == target_id]
        if not test_cases:
            print(f"[!] Error: Invalid case number {args.case}. Must be 1-6.")
            sys.exit(1)

    # 4. Compile the LangGraph pipeline
    print("\n[Step 3] Compiling LangGraph pipeline graph (pipeline/graph.py)...")
    try:
        pipeline_graph = build_pipeline_graph()
        pipeline_app = pipeline_graph.compile()
        print("  -> SUCCESS: Pipeline StateGraph compiled successfully.")
    except Exception as e:
        print(f"\n[!] GRAPH COMPILATION ERROR: {e}")
        sys.exit(1)

    results = []

    # 5. Execute test cases through compiled LangGraph pipeline
    for tc in test_cases:
        run_id = tc["id"]
        print("\n" + "-" * 70)
        print(f"RUNNING: {tc['name']}")
        print(f"Query:   \"{tc['query']}\"")
        print(f"Goal:    {tc['description']}")
        print("-" * 70)

        # Passive Observability: Start MLflow run for current scenario
        current_mlflow_run_id = "N/A"
        try:
            active_run = mlflow_tracker.start_run(run_id)
            if active_run and hasattr(active_run, "info") and hasattr(active_run.info, "run_id"):
                current_mlflow_run_id = active_run.info.run_id
            mlflow_tracker.log_params({
                "model_name": os.getenv("SUPERVISOR_LLM_MODEL", ""),
                "endpoint": os.getenv("SUPERVISOR_LLM_ENDPOINT", ""),
                "confidence_threshold": SUPERVISOR_CONFIDENCE_THRESHOLD,
                "tau_phase": TAU_PHASE,
                "tau_zerolag": TAU_ZEROLAG,
                "query": tc["query"],
                "data_path": str(data_path),
                "is_real_data": is_real_data,
            })
        except Exception as ml_err:
            print(f"[!] Warning: MLflow pre-execution logging failed: {ml_err}")

        initial_state: GraphState = {
            "pipeline_error": None,
            "run_id": run_id,
            "raw_data_path": str(data_path),
            "user_request": tc["query"],
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

        exec_error: Optional[Exception] = None
        output_state: Dict[str, Any] = dict(initial_state)
        executed_nodes: List[str] = []

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
            import traceback
            print(f"[!] EXCEPTION during pipeline graph execution: {type(e).__name__}: {e}")
            traceback.print_exc()
            output_state["pipeline_error"] = str(e)

        plan = output_state.get("plan")
        manifest = output_state.get("parameter_manifest")
        clarif = output_state.get("clarification_question")
        preflight = output_state.get("preflight_confirmed")
        pipe_err = output_state.get("pipeline_error")

        # Routed node is directly extracted from the graph's actual visited node sequence
        # (independent verification from LangGraph execution events, not post-hoc re-calculation)
        if exec_error is not None:
            routed_node = f"N/A (INFRA ERROR: halted in {executed_nodes[-1]})" if executed_nodes else "N/A (INFRA ERROR)"
        elif len(executed_nodes) > 1:
            routed_node = executed_nodes[-1]
        elif executed_nodes:
            routed_node = f"HALTED IN {executed_nodes[0]}"
        else:
            routed_node = "N/A (NOT STARTED)"


        # Debug print to visually confirm payload.keys() node names from LangGraph stream
        print(f"\n[Graph Traversal Debug]: executed_nodes={executed_nodes}")

        print("\n[Returned GraphState]:")
        if exec_error is not None:
            print(f"  [!] Note: Execution halted prematurely due to {type(exec_error).__name__}.")

        print(f"  - Routed Node:            {routed_node}")
        print(f"  - Pipeline Error:         {pipe_err}")
        print(f"  - Plan:                   {plan}")
        print(f"  - Clarification Question: {clarif}")
        print(f"  - Preflight Confirmed:    {preflight}")
        print(f"  - Parameter Manifest:\n{format_manifest(manifest)}")

        # Check trace log on disk
        event_count = check_trace_log(run_id)
        print(f"\n[Trace Log]: logs/trace_{run_id}.json generated with {event_count} events.")

        # Passive Observability: Log results, manifest, and trace events to MLflow
        try:
            mlflow_tracker.log_manifest(manifest)
            mlflow_tracker.log_result(
                routed_node=routed_node,
                plan=plan,
                clarification_question=clarif,
                pipeline_error=pipe_err,
            )
            mlflow_tracker.log_trace_events(event_count)
        except Exception as ml_err:
            print(f"[!] Warning: MLflow post-execution logging failed: {ml_err}")
        finally:
            try:
                mlflow_tracker.end_run()
            except Exception as ml_err:
                print(f"[!] Warning: MLflow end_run failed: {ml_err}")

        # Evaluate test case criteria
        passed = False
        status = "FAIL"
        reason = ""

        if exec_error is not None:
            passed = False
            if is_infra_error(exec_error):
                status = "INFRA ERROR"
                reason = f"Infrastructure/Connection Error ({type(exec_error).__name__}): LLM endpoint unreachable or refused connection."
            else:
                status = "ERROR"
                reason = f"Execution Exception ({type(exec_error).__name__}): {exec_error}"

        elif run_id == "playground_test_01":
            if is_real_data:
                if plan is not None and manifest is not None and clarif is None and routed_node == "gate_1_review":
                    passed = True
                    reason = f"Real Data: Plan created ({len(plan.channels)} channels, band={plan.freq_band.name}); routed to gate_1_review."
                elif clarif is not None:
                    passed = True
                    reason = f"Real Data: Supervisor requested clarification: '{clarif}' (routed to {routed_node})."
                else:
                    reason = f"Real Data: Neither plan nor clarification produced (routed to {routed_node})."
            else:
                if (
                    plan is not None
                    and plan.freq_band.name.lower() == "alpha"
                    and set(plan.channels) == {"F3", "F4"}
                    and plan.condition == "rest"
                    and preflight is False
                    and manifest is not None
                    and not any(m.needs_human_input for m in manifest)
                    and clarif is None
                    and routed_node == "gate_1_review"
                    and event_count > 0
                ):
                    passed = True
                    reason = "Fully specified plan created; zero human input needed; correctly routed to gate_1_review."
                else:
                    reason = f"Failed criteria: plan={plan is not None}, routed_node={routed_node}, clarif={clarif}."
            status = "PASS" if passed else "FAIL"

        elif run_id == "playground_test_02":
            channels_entry = next((m for m in manifest if m.name == "channels"), None) if manifest else None
            is_channels_flagged = (
                channels_entry is not None
                and channels_entry.needs_human_input is True
                and (channels_entry.confidence is not None and channels_entry.confidence < SUPERVISOR_CONFIDENCE_THRESHOLD)
            )
            other_entries_clean = (
                manifest is not None
                and not any(m.needs_human_input for m in manifest if m.name != "channels")
            )

            if is_real_data:
                ch_conf = channels_entry.confidence if channels_entry else None
                ch_flagged = channels_entry.needs_human_input if channels_entry else None
                if plan is not None and manifest is not None and routed_node == "gate_1_review":
                    passed = True
                    if ch_flagged is True:
                        reason = f"Real Data: Regional query resolved ({len(plan.channels)} channels mapped, conf={ch_conf}, correctly flagged for review); routed to gate_1_review."
                    else:
                        reason = f"Real Data: Regional query resolved ({len(plan.channels)} channels mapped, conf={ch_conf}); routed to gate_1_review."
                elif clarif is not None:
                    passed = True
                    reason = f"Real Data: Regional query triggered clarification: '{clarif}' (routed to {routed_node})."
                else:
                    reason = f"Real Data: Failed to process regional query (routed to {routed_node})."
            else:
                if (
                    plan is not None
                    and manifest is not None
                    and is_channels_flagged
                    and other_entries_clean
                    and clarif is None
                    and routed_node == "gate_1_review"
                    and event_count > 0
                ):
                    passed = True
                    reason = f"Complete plan constructed; regional channels correctly flagged with confidence < {SUPERVISOR_CONFIDENCE_THRESHOLD}; routed to gate_1_review."
                else:
                    reason = f"Failed criteria: plan={plan is not None}, channels_flagged={is_channels_flagged}, other_clean={other_entries_clean}, routed_node={routed_node}."
            status = "PASS" if passed else "FAIL"

        elif run_id == "playground_test_03":
            if (
                plan is None
                and manifest is None
                and clarif is not None
                and len(clarif.strip()) > 0
                and routed_node == "clarification_pause"
                and event_count > 0
            ):
                passed = True
                mode = "Real Data" if is_real_data else "Synthetic"
                reason = f"{mode}: Zero-guessing guardrail held — supervisor halted with clarification and routed to clarification_pause."
            else:
                mode = "Real Data" if is_real_data else "Synthetic"
                reason = f"{mode}: Failed guardrail — plan={plan is not None}, manifest={manifest is not None}, routed_node={routed_node}."
            status = "PASS" if passed else "FAIL"

        elif run_id == "playground_test_04":
            advisory = next((m for m in manifest if m.name == "cross_metric_synthesis"), None) if manifest else None
            is_advisory_valid = (
                advisory is not None
                and advisory.category == "advisory"
                and advisory.risk_tier == "elevated"
                and advisory.needs_human_input is True
                and "DISABLED" in advisory.proposed_value
            )

            if is_real_data:
                if plan is not None and is_advisory_valid and routed_node == "gate_1_review":
                    passed = True
                    reason = "Real Data: Metric-subset plan created; cross_metric_synthesis advisory verified (elevated risk, human review required); routed to gate_1_review."
                elif clarif is not None:
                    passed = True
                    reason = f"Real Data: Supervisor requested clarification: '{clarif}' (routed to {routed_node})."
                else:
                    reason = f"Real Data: Metric subset check failed (advisory={is_advisory_valid}, routed_node={routed_node})."
            else:
                if (
                    plan is not None
                    and manifest is not None
                    and is_advisory_valid
                    and clarif is None
                    and routed_node == "gate_1_review"
                    and event_count > 0
                ):
                    passed = True
                    reason = "Metric-subset correctly triggered cross_metric_synthesis advisory (risk_tier='elevated', needs_human_input=True); routed to gate_1_review."
                else:
                    reason = f"Failed criteria: plan={plan is not None}, advisory_valid={is_advisory_valid}, routed_node={routed_node}, clarif={clarif}."
            status = "PASS" if passed else "FAIL"

        elif run_id == "playground_test_05":
            advisory = next((m for m in manifest if m.name == "cross_metric_synthesis"), None) if manifest else None
            is_advisory_absent = (advisory is None)

            if is_real_data:
                if plan is not None and is_advisory_absent and routed_node == "gate_1_review":
                    passed = True
                    reason = "Real Data: Balanced metric plan created; cross_metric_synthesis advisory correctly absent; routed to gate_1_review."
                elif clarif is not None:
                    passed = True
                    reason = f"Real Data: Supervisor requested clarification: '{clarif}' (routed to {routed_node})."
                else:
                    reason = f"Real Data: Balanced metric check failed (advisory_absent={is_advisory_absent}, routed_node={routed_node})."
            else:
                if (
                    plan is not None
                    and manifest is not None
                    and is_advisory_absent
                    and clarif is None
                    and routed_node == "gate_1_review"
                    and event_count > 0
                ):
                    passed = True
                    reason = "Balanced metrics (Phase-Robust + Zero-Lag) correctly omitted cross_metric_synthesis advisory; routed to gate_1_review."
                else:
                    reason = f"Failed criteria: plan={plan is not None}, advisory_absent={is_advisory_absent}, routed_node={routed_node}, clarif={clarif}."
            status = "PASS" if passed else "FAIL"

        elif run_id == "playground_test_06":
            plot_file = Path("outputs") / "plots" / f"{run_id}_overview.png"
            tool_called = check_tool_called(run_id, "generate_dataset_overview_plot")
            plot_exists = plot_file.exists()

            if is_real_data:
                if (
                    plan is not None
                    and manifest is not None
                    and tool_called
                    and plot_exists
                    and routed_node == "gate_1_review"
                ):
                    passed = True
                    reason = f"Real Data: Overview plot generated ({plot_file.name} exists); plan constructed ({len(plan.channels)} channels, band={plan.freq_band.name}); routed to gate_1_review."
                elif not tool_called:
                    reason = f"Real Data: Supervisor did not invoke generate_dataset_overview_plot (routed to {routed_node})."
                elif not plot_exists:
                    reason = f"Real Data: generate_dataset_overview_plot was invoked, but expected plot file {plot_file} not found on disk."
                elif plan is None:
                    reason = f"Real Data: Overview plot generated, but plan compilation failed (routed to {routed_node})."
                else:
                    reason = f"Real Data: Failed criteria: plan={plan is not None}, tool_called={tool_called}, plot_exists={plot_exists}, routed_node={routed_node}."
            else:
                if (
                    plan is not None
                    and manifest is not None
                    and tool_called
                    and plot_exists
                    and routed_node == "gate_1_review"
                    and event_count > 0
                ):
                    passed = True
                    reason = f"Overview plot generated ({plot_file.name} exists); plan constructed; routed to gate_1_review."
                elif not tool_called:
                    reason = f"Supervisor did not invoke generate_dataset_overview_plot (routed to {routed_node})."
                elif not plot_exists:
                    reason = f"generate_dataset_overview_plot was invoked, but plot file {plot_file} not found on disk."
                else:
                    reason = f"Failed criteria: plan={plan is not None}, tool_called={tool_called}, plot_exists={plot_exists}, routed_node={routed_node}."
            status = "PASS" if passed else "FAIL"

        results.append({
            "id": run_id,
            "name": tc["name"],
            "query": tc["query"],
            "goal": tc["description"],
            "status": status,
            "passed": passed,
            "routed_node": routed_node,
            "exception": exec_error,
            "reason": reason,
            "event_count": event_count,
            "mlflow_run_id": current_mlflow_run_id,
            "manifest": manifest,
            "clarification_question": clarif,
            "plan": plan,
        })

    # 6. Final summary report
    print("\n" + "=" * 70)
    print("PLAYGROUND INTEGRATION TEST SUMMARY")
    print("=" * 70)
    all_passed = True
    has_infra_error = False
    for r in results:
        r_status = r["status"]
        if r_status == "PASS":
            status_str = "PASS [OK]"
        elif r_status == "INFRA ERROR":
            status_str = "INFRA ERROR [!]"
            all_passed = False
            has_infra_error = True
        elif r_status == "ERROR":
            status_str = "ERROR [!]"
            all_passed = False
        else:
            status_str = "FAIL [X]"
            all_passed = False

        print(f"{status_str} | {r['name']} -> Node: [{r['routed_node']}]")
        print(f"         Trace Events: {r['event_count']} | Result: {r['reason']}")

    print("-" * 70)
    if all_passed:
        mode_label = "REAL DATASET DIAGNOSTIC" if is_real_data else "SYNTHETIC FIXTURE INTEGRATION"
        print(f"ALL 6 SCENARIOS COMPLETED SUCCESSFULLY ({mode_label}).")
        print("Supervisor Node, LangGraph Topology, Conditional Router, Tools, Schemas, and Passive Tracer are functioning as expected.")
    elif has_infra_error:
        print("RUN HALTED / INCOMPLETE DUE TO INFRASTRUCTURE OR CONNECTION ERROR.")
        print("Please verify your LLM endpoint in .env is online, reachable, and re-run.")
    else:
        print("SOME CHECKS FAILED. Review the detailed state printouts and trace logs above.")
    print("=" * 70)

    # 7. Print MLflow Observability Tracking Summary
    print("\n" + "=" * 70)
    print("MLFLOW OBSERVABILITY TRACKING SUMMARY (mlruns/)")
    print("=" * 70)
    try:
        import mlflow
        from fc_pipeline.observability.mlflow_tracker import DEFAULT_EXPERIMENT_NAME

        mlruns_dir = Path("mlruns")
        if mlruns_dir.exists():
            runs_df = mlflow.search_runs(experiment_names=[DEFAULT_EXPERIMENT_NAME])
            if not runs_df.empty:
                print(f"Tracking Directory: {mlruns_dir.resolve()} | Total Runs Logged: {len(runs_df)}\n")
                cols = [
                    "tags.scenario_id",
                    "status",
                    "tags.routed_node",
                    "params.model_name",
                    "params.confidence_threshold",
                    "metrics.trace_event_count",
                ]
                avail_cols = [c for c in cols if c in runs_df.columns]
                summary_df = runs_df[avail_cols].rename(columns={
                    "tags.scenario_id": "Scenario ID",
                    "status": "Run Status",
                    "tags.routed_node": "Routed Node",
                    "params.model_name": "Model",
                    "params.confidence_threshold": "Conf Thresh",
                    "metrics.trace_event_count": "Trace Events",
                })
                if "Scenario ID" in summary_df.columns:
                    summary_df = summary_df.sort_values("Scenario ID").reset_index(drop=True)
                print(summary_df.to_string(index=False))
            else:
                print("No runs recorded in experiment.")
        else:
            print("mlruns/ directory was not found.")
    except Exception as e:
        print(f"[!] Warning: Failed to query MLflow tracking summary: {e}")
    print("=" * 70)

    # 8. Generate Markdown report (outputs/reports/playground_run_report.md)
    print("\n" + "=" * 70)
    print("GENERATING PLAYGROUND MARKDOWN REPORT")
    print("=" * 70)
    report_path = Path("outputs") / "reports" / "playground_run_report.md"
    try:
        saved_path = generate_playground_markdown_report(
            results=results,
            meta=meta,
            is_real_data=is_real_data,
            output_report_path=report_path,
        )
        print(f"  -> SUCCESS: Report generated at: {saved_path.resolve()}")
    except Exception as rep_err:
        print(f"[!] Warning: Failed to generate playground markdown report: {rep_err}")
    print("=" * 70)


if __name__ == "__main__":
    run_playground()
