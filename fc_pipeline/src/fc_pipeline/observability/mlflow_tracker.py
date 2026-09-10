"""Passive observability layer wrapping MLflow for supervisor pipeline tracking.

Provides non-blocking tracking of execution parameters, parameter manifests,
state outcomes, and trace event counts with automatic credential and endpoint
privacy masking.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure local file-based tracking backend is permitted on MLflow >= 3.0
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow.entities import Run

from fc_pipeline.schemas.manifest import ParameterManifestEntry

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "supervisor_playground"
DEFAULT_TRACKING_DIR = Path("mlruns")


def mask_sensitive_value(key: str, val: Any) -> Any:
    """Mask credentials, tokens, internal network endpoints, and file paths before logging."""
    if val is None:
        return None
    k_lower = key.lower()
    if any(sec in k_lower for sec in ("key", "token", "secret", "password", "auth", "credential")):
        return "[MASKED - CREDENTIAL]"

    if isinstance(val, str):
        cleaned = val.strip().strip('"').strip("'")
        # Apply path masking matching chainlit_app.py's mask_text() pattern
        user = os.getenv("USERNAME") or os.getenv("USER") or ""
        if user:
            cleaned = re.sub(re.escape(user), "[MASKED_USER]", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[A-Za-z]:\\Users\\[^\\]+\\", "[LOCAL_ROOT]/", cleaned)
        cleaned = re.sub(r"/(?:home|Users)/[^/]+/", "[LOCAL_ROOT]/", cleaned)
        # Original endpoint masking
        if "groq.com" in cleaned.lower():
            return "https://api.groq.com/openai/v1"
        if cleaned.startswith("gsk_"):
            return "[MASKED - API KEY]"
        if re.search(r"https?://", cleaned, re.IGNORECASE):
            scheme = "https://" if cleaned.lower().startswith("https://") else "http://"
            return f"{scheme}[CONFIGURED - INTERNAL IP/HOST MASKED]"
        return cleaned

    return val


class MLflowTracker:
    """Thin, robust wrapper around MLflow tracking API."""

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        tracking_dir: Path = DEFAULT_TRACKING_DIR,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_dir = tracking_dir.resolve()
        self._initialized: bool = False
        self._active_run: Optional[Run] = None

    def init_tracking(self) -> None:
        """Initialize MLflow local file-based tracking URI and experiment."""
        if self._initialized:
            return
        try:
            self.tracking_dir.mkdir(parents=True, exist_ok=True)
            tracking_uri = self.tracking_dir.as_uri()
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow tracking at '{self.tracking_dir}': {e}")

    def start_run(self, run_id: str) -> Optional[Run]:
        """Start an MLflow run tagged with the scenario ID."""
        try:
            self.init_tracking()
            # If a prior run was left open, end it gracefully
            if mlflow.active_run() is not None:
                mlflow.end_run()

            run = mlflow.start_run(run_name=run_id, tags={"scenario_id": run_id})
            self._active_run = run
            return run
        except Exception as e:
            logger.warning(f"MLflow start_run('{run_id}') failed: {e}")
            return None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log static configuration parameters with privacy masking."""
        try:
            if mlflow.active_run() is None:
                return
            masked_params = {k: mask_sensitive_value(k, v) for k, v in params.items()}
            # Format and enforce length limits
            clean_params = {}
            for k, v in masked_params.items():
                val_str = str(v)
                if len(val_str) > 500:
                    val_str = val_str[:497] + "..."
                clean_params[k] = val_str

            mlflow.log_params(clean_params)
        except Exception as e:
            logger.warning(f"MLflow log_params failed: {e}")

    def log_manifest(self, manifest: Optional[List[ParameterManifestEntry]]) -> None:
        """Log parameter manifest rows as a structured JSON artifact and summary params."""
        try:
            if mlflow.active_run() is None:
                return

            if manifest is None:
                mlflow.log_param("manifest_present", "false")
                return

            mlflow.log_param("manifest_present", "true")
            mlflow.log_param("manifest_entry_count", str(len(manifest)))

            # Log clean structured dictionary artifact
            manifest_dicts = []
            for entry in manifest:
                m_dict = entry.model_dump() if hasattr(entry, "model_dump") else entry.dict()
                manifest_dicts.append(m_dict)

            mlflow.log_dict(manifest_dicts, "parameter_manifest.json")

            # High-level summary metrics/flags to avoid parameter clutter
            human_input_count = sum(1 for m in manifest if m.needs_human_input)
            has_advisory = any(m.name == "cross_metric_synthesis" for m in manifest)

            mlflow.log_params({
                "manifest_human_input_count": str(human_input_count),
                "has_synthesis_advisory": str(has_advisory),
            })
        except Exception as e:
            logger.warning(f"MLflow log_manifest failed: {e}")

    def log_result(
        self,
        routed_node: str,
        plan: Any = None,
        clarification_question: Optional[str] = None,
        pipeline_error: Optional[str] = None,
    ) -> None:
        """Log execution outcome tags and indicators."""
        try:
            if mlflow.active_run() is None:
                return

            tags = {
                "routed_node": str(routed_node),
                "has_plan": str(plan is not None),
                "has_clarification": str(clarification_question is not None),
                "has_pipeline_error": str(pipeline_error is not None),
            }

            if plan is not None:
                tags["plan_condition"] = str(getattr(plan, "condition", ""))
                band = getattr(plan, "freq_band", None)
                tags["plan_freq_band"] = getattr(band, "name", "") if band else ""
                channels = getattr(plan, "channels", [])
                tags["plan_channels"] = ", ".join(channels)
                metrics = getattr(plan, "metrics", [])
                tags["plan_metrics"] = ", ".join(
                    getattr(m, "value", str(m)) for m in metrics
                )

            if clarification_question:
                tags["clarification_question"] = clarification_question[:1000]

            if pipeline_error:
                tags["pipeline_error"] = str(pipeline_error)[:1000]

            mlflow.set_tags(tags)

            # Metrics for filtering in MLflow UI / search_runs
            is_success = 1.0 if (routed_node in ("gate_1_review", "clarification_pause") and not pipeline_error) else 0.0
            mlflow.log_metric("is_success", is_success)

        except Exception as e:
            logger.warning(f"MLflow log_result failed: {e}")

    def log_trace_events(self, event_count: int) -> None:
        """Log passive tracer event count as a metric."""
        try:
            if mlflow.active_run() is None:
                return
            mlflow.log_metric("trace_event_count", float(event_count))
        except Exception as e:
            logger.warning(f"MLflow log_trace_events failed: {e}")

    def end_run(self, status: Optional[str] = None) -> None:
        """End the active MLflow run."""
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run(status=status)
            self._active_run = None
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")


# Singleton instance and module-level functional API
_default_tracker = MLflowTracker()


def get_tracker() -> MLflowTracker:
    return _default_tracker


def start_run(run_id: str) -> Optional[Run]:
    return _default_tracker.start_run(run_id)


def log_params(params: Dict[str, Any]) -> None:
    _default_tracker.log_params(params)


def log_manifest(manifest: Optional[List[ParameterManifestEntry]]) -> None:
    _default_tracker.log_manifest(manifest)


def log_result(
    routed_node: str,
    plan: Any = None,
    clarification_question: Optional[str] = None,
    pipeline_error: Optional[str] = None,
) -> None:
    _default_tracker.log_result(
        routed_node=routed_node,
        plan=plan,
        clarification_question=clarification_question,
        pipeline_error=pipeline_error,
    )


def log_trace_events(event_count: int) -> None:
    _default_tracker.log_trace_events(event_count)


def end_run(status: Optional[str] = None) -> None:
    _default_tracker.end_run(status=status)