"""Run context: the stored, read-only record of a completed pipeline run.

Why this exists
---------------
After Data Prep finishes, the user may ask about the run ("which channels were
selected?", "how many epochs?").  Those questions must be answered from what
the run actually did, without restarting the Supervisor or Data Prep.

The run context is a small JSON document (references and plain values only, no
signal data) that is

* built when a run completes,
* persisted next to the run's outputs, so a follow-up still works if the
  in-memory chat session lost its state, and
* queried by :func:`answer_run_question`, which answers *deterministically*
  from the stored values.  No LLM is involved, so an answer can never
  contradict the run.

This module is deliberately free of Chainlit, LangGraph and LLM imports.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

RUN_CONTEXT_SCHEMA_VERSION = 1

_METRIC_LABELS = {
    "pli": "PLI",
    "wpli": "wPLI",
    "imcoh": "|ImCoh|",
    "imaginary_coherence": "|ImCoh|",
    "plv": "PLV",
    "coh": "Coherence",
    "coherence": "Coherence",
}


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object (pydantic model, dataclass)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _jsonable(obj: Any) -> Any:
    """Convert pydantic models, enums and paths into plain JSON values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump(mode="json"))
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    value = getattr(obj, "value", None)  # Enum
    if isinstance(value, (str, int, float)):
        return value
    return str(obj)


def _metric_id(metric: Any) -> str:
    """Normalise a MetricEnum, its value, or 'MetricEnum.PLI' to a lowercase id."""
    raw = str(getattr(metric, "value", metric))
    raw = raw.split(".")[-1].strip().lower()
    return {"imaginary_coherence": "imcoh"}.get(raw, raw)


def metric_label(metric: Any) -> str:
    key = _metric_id(metric)
    return _METRIC_LABELS.get(key, key.upper())


def display_channel(name: Any) -> str:
    """Readable electrode name for the UI: ``C3..`` / ``Fp1.`` -> ``C3`` / ``Fp1``.

    Display only: the real channel names (with the trailing dots some EDF files
    use) stay untouched everywhere they are matched or looked up.
    """
    text = str(name).strip()
    return text.rstrip(".") or text


# ---------------------------------------------------------------------------
# Build / persist / restore
# ---------------------------------------------------------------------------


def build_run_context(
    *,
    run_id: str,
    plan: Any,
    data_prep_summary: Any,
    parameter_manifest: Any = None,
    preprocessed_data_path: Optional[str] = None,
    raw_data_path: Optional[str] = None,
    channel_plot_paths: Optional[Mapping[str, Any]] = None,
    bad_channels_dropped: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the JSON-serialisable run context from the pieces of a finished run."""
    manifest_rows = []
    for entry in parameter_manifest or []:
        manifest_rows.append(
            {
                "name": _get(entry, "name"),
                "category": _jsonable(_get(entry, "category")),
                "proposed_value": _jsonable(_get(entry, "proposed_value")),
                "human_approved_value": _jsonable(_get(entry, "human_approved_value")),
                "risk_tier": _jsonable(_get(entry, "risk_tier")),
            }
        )
    return {
        "schema_version": RUN_CONTEXT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "raw_data_path": _jsonable(raw_data_path),
        "preprocessed_data_path": _jsonable(preprocessed_data_path),
        "plan": _jsonable(plan),
        "data_prep_summary": _jsonable(data_prep_summary),
        "bad_channels_dropped": list(bad_channels_dropped or []),
        "channel_plot_paths": _jsonable(dict(channel_plot_paths or {})),
        "parameter_manifest": manifest_rows,
        # Filled in by the Connectivity stage once it exists.
        "connectivity": None,
    }


def build_run_context_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a run context from the Chainlit ``completed_run_state`` dict."""
    return build_run_context(
        run_id=str(state.get("run_id") or "unknown"),
        plan=state.get("plan"),
        data_prep_summary=state.get("data_prep_summary"),
        parameter_manifest=state.get("parameter_manifest"),
        preprocessed_data_path=state.get("preprocessed_data_path"),
        raw_data_path=state.get("raw_data_path"),
        channel_plot_paths=state.get("channel_plot_paths"),
        bad_channels_dropped=state.get("bad_channels_dropped"),
    )


def run_context_path(
    preprocessed_data_path: Optional[str],
    run_id: str,
    fallback_dir: str = "outputs",
) -> Path:
    """Where the context of ``run_id`` is stored: next to its preprocessed epochs."""
    base = Path(preprocessed_data_path).parent if preprocessed_data_path else Path(fallback_dir)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    return base / f"run_context_{safe_id}.json"


def save_run_context(context: Mapping[str, Any], path: Path) -> Path:
    """Write the context atomically (temp file + rename) so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".run_context_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(context, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def load_run_context(path: Path) -> Optional[Dict[str, Any]]:
    """Load a stored context; returns ``None`` if missing or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not load run context %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("schema_version") != RUN_CONTEXT_SCHEMA_VERSION:
        return None
    return data


def restore_state(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Rebuild the Chainlit ``completed_run_state`` dict from a stored context.

    Typed objects (AnalysisPlan, DataPrepSummary, manifest entries) are
    re-created when possible; otherwise plain dictionaries are kept, which the
    answer functions here accept equally.
    """
    plan: Any = context.get("plan")
    summary: Any = context.get("data_prep_summary")
    manifest: Any = context.get("parameter_manifest") or []
    try:
        from fc_pipeline.schemas.plan import AnalysisPlan

        if isinstance(plan, Mapping):
            plan = AnalysisPlan(**plan)
    except Exception:
        pass
    try:
        from fc_pipeline.deterministic.data_prep.models import DataPrepSummary

        if isinstance(summary, Mapping):
            summary = DataPrepSummary(**summary)
    except Exception:
        pass
    return {
        "run_id": context.get("run_id"),
        "raw_data_path": context.get("raw_data_path"),
        "preprocessed_data_path": context.get("preprocessed_data_path"),
        "channel_plot_paths": dict(context.get("channel_plot_paths") or {}),
        "bad_channels_dropped": list(context.get("bad_channels_dropped") or []),
        "data_prep_summary": summary,
        "parameter_manifest": manifest,
        "plan": plan,
        "informational_artifacts": {},
    }


# ---------------------------------------------------------------------------
# Intent: new analysis vs. question about the completed run
# ---------------------------------------------------------------------------

_NEW_ANALYSIS_VERB = re.compile(
    r"^\s*(please\s+|now\s+|can you\s+|could you\s+)?"
    r"(re-?run|redo|repeat|analy[sz]e|compute|calculate|run|try|do|use|switch to|change)\b",
    re.IGNORECASE,
)
_ANALYSIS_TARGET = re.compile(
    r"\b(theta|alpha|beta|delta|gamma|hz|pli|wpli|plv|coherence|imcoh|band|"
    r"frontal|central|parietal|occipital|temporal|channels?|region|condition|"
    r"another|different|new)\b",
    re.IGNORECASE,
)


def looks_like_new_analysis(text: str) -> bool:
    """True when the message asks for a *different* analysis rather than asking about this run.

    Deliberately conservative: it must start like an instruction (analyse,
    compute, run, ...) AND name something to analyse.  Plain questions
    ("which band was used?") are never classed as new analyses.
    """
    return bool(_NEW_ANALYSIS_VERB.match(text) and _ANALYSIS_TARGET.search(text))


# ---------------------------------------------------------------------------
# Deterministic answers about the completed run
# ---------------------------------------------------------------------------

# Requests that want an artifact (plot/signal/comparison) are left to the
# existing artifact handler.
_ARTIFACT_WORDS = re.compile(
    r"\b(plots?|graphs?|figures?|visuali[sz]\w*|psd|spectr\w*|signals?|compare|comparison|"
    r"images?|pictures?)\b|before (and|vs\.?|versus) after",
    re.IGNORECASE,
)

_TOPICS: Dict[str, re.Pattern[str]] = {
    "summary": re.compile(
        r"\b(summary|summari[sz]e|settings|parameters?|what was done|what did you do|"
        r"run details|details of (the )?run)\b",
        re.IGNORECASE,
    ),
    "run_id": re.compile(r"\b(run[ _]?id|which run)\b", re.IGNORECASE),
    "channels": re.compile(r"\b(channels?|electrodes?|sensors?|region)\b", re.IGNORECASE),
    "band": re.compile(
        r"\b(frequency|freq|band|bandwidth|theta|alpha|beta|delta|gamma)\b", re.IGNORECASE
    ),
    "condition": re.compile(r"\b(condition|event|label|stimulus)\b", re.IGNORECASE),
    "metrics": re.compile(
        r"\b(metrics?|measures?|connectivity|pli|wpli|plv|coherence|imcoh)\b", re.IGNORECASE
    ),
    "reference": re.compile(r"\b(reference|referenc\w*|car)\b", re.IGNORECASE),
    "epochs": re.compile(r"\b(epochs?|segments?|trials?)\b", re.IGNORECASE),
    "sampling": re.compile(r"\b(sampling|sample rate|sfreq|resampl\w*)\b", re.IGNORECASE),
    "filter": re.compile(r"\b(filter\w*|band-?pass)\b", re.IGNORECASE),
    "dropped": re.compile(
        r"\b(bad|dropped|removed|rejected|discard\w*|excluded|flatline\w*)\b", re.IGNORECASE
    ),
    "output": re.compile(r"\b(files?|saved|outputs?|paths?)\b", re.IGNORECASE),
}


def _plan_fields(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    plan = ctx.get("plan")
    band = _get(plan, "freq_band")
    return {
        "channels": [str(c) for c in (_get(plan, "channels") or [])],
        "condition": _get(plan, "condition"),
        "metrics": [metric_label(m) for m in (_get(plan, "metrics") or [])],
        "band_name": _get(band, "name"),
        "fmin": _get(band, "fmin"),
        "fmax": _get(band, "fmax"),
    }


def _fmt_hz(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _answer_channels(ctx: Mapping[str, Any]) -> Optional[str]:
    p = _plan_fields(ctx)
    if not p["channels"]:
        return None
    summary = ctx.get("data_prep_summary")
    dropped = list(ctx.get("bad_channels_dropped") or _get(summary, "dropped_channels") or [])
    retained = [c for c in p["channels"] if c not in dropped]
    orig = _get(summary, "original_channel_count")
    out = f"**Selected channels ({len(p['channels'])}):** {', '.join(display_channel(c) for c in p['channels'])}"
    if orig:
        out += f" (from a {orig}-channel recording)"
    out += "."
    if dropped:
        out += (
            f" **Dropped as bad:** {', '.join(display_channel(c) for c in dropped)}."
            f" **Retained:** {', '.join(display_channel(c) for c in retained) or 'none'}."
        )
    else:
        out += " None were dropped, so all of them were retained."
    return out


def _answer_band(ctx: Mapping[str, Any]) -> Optional[str]:
    p = _plan_fields(ctx)
    if p["band_name"] is None and p["fmin"] is None:
        return None
    out = f"**Frequency band:** {p['band_name']} ({_fmt_hz(p['fmin'])}-{_fmt_hz(p['fmax'])} Hz)."
    summary = ctx.get("data_prep_summary")
    lo, hi = _get(summary, "filter_l_freq"), _get(summary, "filter_h_freq")
    if lo is not None and hi is not None:
        out += f" The signal was band-pass filtered to {_fmt_hz(lo)}-{_fmt_hz(hi)} Hz before epoching."
    return out


def _answer_condition(ctx: Mapping[str, Any]) -> Optional[str]:
    p = _plan_fields(ctx)
    if not p["condition"]:
        return None
    return (
        f"**Condition:** {p['condition']} (a dataset-specific event label; only epochs of "
        "this condition were used)."
    )


def _answer_metrics(ctx: Mapping[str, Any]) -> Optional[str]:
    p = _plan_fields(ctx)
    if not p["metrics"]:
        return None
    out = f"**Requested metrics:** {', '.join(p['metrics'])}."
    if not ctx.get("connectivity"):
        out += (
            " Connectivity has not been computed for this run: the pipeline currently "
            "finishes after Data Preparation, so there are no connectivity values yet."
        )
    return out


def _answer_reference(ctx: Mapping[str, Any]) -> Optional[str]:
    summary = ctx.get("data_prep_summary")
    applied = _get(summary, "reference_applied")
    detected = _get(summary, "reference_detected")
    if applied is None and detected is None and summary is None:
        return None
    out = f"**Reference applied:** {applied or 'none (left as recorded)'}."
    if detected:
        out += f" Reference detected in the file: {detected}."
    return out


def _answer_epochs(ctx: Mapping[str, Any]) -> Optional[str]:
    summary = ctx.get("data_prep_summary")
    count = _get(summary, "epoch_count")
    if count is None:
        return None
    dur = _get(summary, "epoch_duration_seconds")
    cond = _get(summary, "condition") or _plan_fields(ctx)["condition"]
    out = f"**Epochs:** {count}"
    if dur is not None:
        out += f" of {float(dur):.3f} s"
    out += f" for condition {cond}."
    if isinstance(count, int) and count < 2:
        out += (
            " That is too few for reliable connectivity: phase- and coherence-based "
            "measures are estimated across epochs."
        )
    return out


def _answer_sampling(ctx: Mapping[str, Any]) -> Optional[str]:
    sfreq = _get(ctx.get("data_prep_summary"), "sampling_frequency")
    return None if sfreq is None else f"**Sampling frequency:** {_fmt_hz(sfreq)} Hz."


def _answer_filter(ctx: Mapping[str, Any]) -> Optional[str]:
    summary = ctx.get("data_prep_summary")
    lo, hi = _get(summary, "filter_l_freq"), _get(summary, "filter_h_freq")
    if lo is None or hi is None:
        return None
    return f"**Filter:** band-pass {_fmt_hz(lo)}-{_fmt_hz(hi)} Hz, applied to the continuous recording before epoching."


def _answer_dropped(ctx: Mapping[str, Any]) -> Optional[str]:
    summary = ctx.get("data_prep_summary")
    dropped = list(ctx.get("bad_channels_dropped") or _get(summary, "dropped_channels") or [])
    if summary is None and not dropped:
        return None
    if dropped:
        return f"**Dropped as bad channels:** {', '.join(dropped)}."
    return "**Dropped channels:** none. Bad-channel screening removed no channel."


def _answer_output(ctx: Mapping[str, Any]) -> Optional[str]:
    path = ctx.get("preprocessed_data_path")
    plots = list((ctx.get("channel_plot_paths") or {}).keys())
    if not path and not plots:
        return None
    parts = []
    if path:
        parts.append(f"**Preprocessed epochs file:** `{Path(str(path)).name}`")
    if plots:
        parts.append("**Diagnostic plots:** " + ", ".join(p.replace("_", " ") for p in plots))
    return ". ".join(parts) + "."


def _answer_run_id(ctx: Mapping[str, Any]) -> Optional[str]:
    return f"**Run id:** `{ctx.get('run_id')}`" if ctx.get("run_id") else None


def _answer_summary(ctx: Mapping[str, Any]) -> Optional[str]:
    pieces = [
        _answer_run_id(ctx),
        _answer_band(ctx),
        _answer_channels(ctx),
        _answer_condition(ctx),
        _answer_metrics(ctx),
        _answer_reference(ctx),
        _answer_epochs(ctx),
        _answer_sampling(ctx),
    ]
    pieces = [p for p in pieces if p]
    return "\n".join(f"- {p}" for p in pieces) if pieces else None


_ANSWERERS = {
    "summary": _answer_summary,
    "run_id": _answer_run_id,
    "channels": _answer_channels,
    "band": _answer_band,
    "condition": _answer_condition,
    "metrics": _answer_metrics,
    "reference": _answer_reference,
    "epochs": _answer_epochs,
    "sampling": _answer_sampling,
    "filter": _answer_filter,
    "dropped": _answer_dropped,
    "output": _answer_output,
}


def answer_run_question(text: str, context: Mapping[str, Any]) -> Optional[str]:
    """Answer a question about the completed run from its stored values.

    Returns ``None`` when the message is not such a question (it asks for a
    plot/signal/comparison, or matches no known topic) so the caller can fall
    back to the artifact handler.  Never invents a value: a topic whose data is
    missing from the context is skipped.
    """
    if not text or not context or _ARTIFACT_WORDS.search(text):
        return None

    matched = [name for name, pattern in _TOPICS.items() if pattern.search(text)]
    if not matched:
        return None
    # "Sampling rate in Hz" is about the recording, not the frequency band.
    if "sampling" in matched and "band" in matched and not re.search(r"\bband\b", text, re.I):
        matched.remove("band")
    # A full summary already contains the individual topics.
    if "summary" in matched:
        matched = ["summary"]

    answers = []
    for name in matched:
        answer = _ANSWERERS[name](context)
        if answer:
            answers.append(answer)
    if not answers:
        return None
    return "\n\n".join(answers)
