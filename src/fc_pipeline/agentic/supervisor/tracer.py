"""File-based logger writing ReAct scratchpad to logs/trace_<run_id>.json."""

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


_EVENT_SINK: ContextVar[Optional[Callable[[Dict[str, Any]], None]]] = ContextVar(
    "supervisor_event_sink", default=None
)


@contextmanager
def supervisor_event_sink(sink: Optional[Callable[[Dict[str, Any]], None]]):
    """Temporarily mirror tracer events to a UI sink without polluting GraphState."""
    token = _EVENT_SINK.set(sink)
    try:
        yield
    finally:
        _EVENT_SINK.reset(token)


class SupervisorTracer:
    """Passive audit trace logger recording the Supervisor's ReAct interaction history.
    
    Adheres to Constraint 8: writes directly to disk under logs/, causing zero GraphState pollution.
    """

    def __init__(self, run_id: str, log_dir: str = "logs"):
        self.run_id = run_id
        self.log_dir = log_dir
        self.file_path = os.path.join(self.log_dir, f"trace_{self.run_id}.json")
        self.events: List[Dict[str, Any]] = []

        # Reuse an existing run trace when a downstream stage (e.g. Data Prep)
        # emits additional events after the Supervisor has already logged.
        os.makedirs(self.log_dir, exist_ok=True)
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                self.events = list(existing.get("events") or [])
            except Exception:
                self.events = []
        else:
            self._flush_to_disk()

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Records a single reasoning step, tool call, observation, or decision and flushes to disk."""
        event_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "payload": payload,
        }
        self.events.append(event_entry)
        self._flush_to_disk()
        sink = _EVENT_SINK.get()
        if sink is not None:
            try:
                sink(event_entry)
            except Exception:
                # UI streaming must never break the scientific pipeline.
                pass

    def _flush_to_disk(self) -> None:
        """Serializes current trace history to logs/trace_<run_id>.json."""
        trace_data = {
            "run_id": self.run_id,
            "total_events": len(self.events),
            "events": self.events,
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)
