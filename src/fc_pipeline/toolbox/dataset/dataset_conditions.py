"""Toolbox tool: get_dataset_conditions (epoch event triggers and condition labels)."""

from collections import Counter
from typing import Any, Dict, List
import mne
from langchain_core.tools import tool


@tool
def get_dataset_conditions(data_path: str) -> Dict[str, Any]:
    """Inspects dataset annotations and trigger channels to retrieve valid experimental conditions and trial counts.
    
    Allows the Supervisor to validate user-requested conditions against actual dataset triggers without guessing.
    """
    try:
        raw = mne.io.read_raw(data_path, preload=False, verbose=False)
        condition_counts: Dict[str, int] = {}
        
        # 1. Inspect embedded annotations (e.g., BIDS/EDF+ annotations)
        if raw.annotations is not None and len(raw.annotations) > 0:
            descriptions = [str(desc) for desc in raw.annotations.description]
            condition_counts = dict(Counter(descriptions))
        
        # 2. If no annotations, check for stimulus/trigger channel events
        if not condition_counts:
            try:
                events = mne.find_events(raw, verbose=False)
                if len(events) > 0:
                    event_ids = [int(event[2]) for event in events]
                    raw_counts = Counter(event_ids)
                    condition_counts = {f"event_{eid}": count for eid, count in raw_counts.items()}
            except Exception:
                # No stimulus channel present or events could not be parsed
                pass
        
        total_trials = sum(condition_counts.values())
        return {
            "conditions": condition_counts,
            "total_trials": total_trials,
            "has_events": total_trials > 0,
            "error": None,
        }
    except Exception as e:
        return {
            "conditions": {},
            "total_trials": 0,
            "has_events": False,
            "error": f"Failed to extract conditions from '{data_path}': {str(e)}",
        }
