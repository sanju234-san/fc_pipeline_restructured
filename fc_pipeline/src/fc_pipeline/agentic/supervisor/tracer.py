"""File-based logger writing ReAct scratchpad to logs/trace_<run_id>.json."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List


class SupervisorTracer:
    """Passive audit trace logger recording the Supervisor's ReAct interaction history.
    
    Adheres to Constraint 8: writes directly to disk under logs/, causing zero GraphState pollution.
    """

    def __init__(self, run_id: str, log_dir: str = "logs"):
        self.run_id = run_id
        self.log_dir = log_dir
        self.file_path = os.path.join(self.log_dir, f"trace_{self.run_id}.json")
        self.events: List[Dict[str, Any]] = []
        
        # Ensure log directory exists immediately
        os.makedirs(self.log_dir, exist_ok=True)
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

    def _flush_to_disk(self) -> None:
        """Serializes current trace history to logs/trace_<run_id>.json."""
        trace_data = {
            "run_id": self.run_id,
            "total_events": len(self.events),
            "events": self.events,
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)
