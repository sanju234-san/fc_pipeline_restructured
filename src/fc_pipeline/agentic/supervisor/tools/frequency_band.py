"""Tool: resolve_frequency_band (standard band boundaries from query)."""

import re
from typing import Any, Dict, Optional
from langchain_core.tools import tool

from fc_pipeline.config.thresholds import SUPERVISOR_CONFIDENCE_THRESHOLD, N_CYCLES_MIN


# Canonical physiological EEG frequency bands (Section 6.1)
CANONICAL_BANDS: Dict[str, Dict[str, float]] = {
    "delta": {"fmin": 1.0, "fmax": 4.0},
    "theta": {"fmin": 4.0, "fmax": 8.0},
    "alpha": {"fmin": 8.0, "fmax": 12.0},
    "beta": {"fmin": 13.0, "fmax": 30.0},
    "gamma": {"fmin": 30.0, "fmax": 45.0},
}


@tool
def resolve_frequency_band(
    band_name_or_range: str, sfreq: float, duration_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """Resolves natural-language band names or numeric ranges to explicit fmin and fmax boundaries with confidence scoring.
    
    Rejects bands exceeding the Nyquist limit (sfreq / 2) and flags needs_human_input when confidence is below threshold.
    If recording duration is available, checks if epoch duration accommodates at least 3 cycles of fmin.
    """
    cleaned = band_name_or_range.strip().lower()
    nyquist = sfreq / 2.0 if sfreq > 0 else 0.0
    
    resolved_name: Optional[str] = band_name_or_range  # Echo input by default for unknown bands
    fmin: float = 0.0
    fmax: float = 0.0
    confidence: float = 0.0
    error: Optional[str] = None
    cycle_check_status: str = "not_evaluated"

    # 1. First resolve canonical band values regardless of sfreq
    if cleaned in CANONICAL_BANDS:
        resolved_name = cleaned
        fmin = CANONICAL_BANDS[cleaned]["fmin"]
        fmax = CANONICAL_BANDS[cleaned]["fmax"]
        confidence = 1.0

    # 2. Explicit numeric range match (e.g., "8-12", "8.0 to 13", "8 - 12 Hz")
    else:
        num_pattern = r"^(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)\s*(?:hz)?$"
        match = re.search(num_pattern, cleaned)
        if not match:
            # Also allow embedded range in phrases like "in range 8-12 hz"
            num_pattern_loose = r"(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)\s*hz"
            match = re.search(num_pattern_loose, cleaned)

        if match:
            fmin = float(match.group(1))
            fmax = float(match.group(2))
            resolved_name = "custom"
            if fmin <= 0:
                error = f"Lower frequency bound ({fmin} Hz) must be strictly greater than 0.0 Hz."
                confidence = 0.0
            elif fmax <= 0:
                error = f"Upper frequency bound ({fmax} Hz) must be strictly greater than 0.0 Hz."
                confidence = 0.0
            elif fmin >= fmax:
                error = f"Lower frequency ({fmin} Hz) must be strictly less than upper frequency ({fmax} Hz)."
                confidence = 0.0
            else:
                confidence = 1.0

        # 3. Fuzzy / partial band match (e.g., "fast alpha", "low theta", "alpha band")
        else:
            for band_name, bounds in CANONICAL_BANDS.items():
                if band_name in cleaned:
                    resolved_name = band_name
                    fmin = bounds["fmin"]
                    fmax = bounds["fmax"]
                    confidence = 0.70  # Below 0.80 threshold: trips needs_human_input
                    break
            
            if resolved_name == band_name_or_range:  # Still the original input - unrecognized
                error = f"Unrecognized frequency band or range: '{band_name_or_range}'."
                confidence = 0.0

    # 4. sfreq validation (run after canonical resolution to preserve band values)
    if sfreq <= 0:
        error = f"Invalid sampling frequency (sfreq={sfreq} Hz): must be strictly greater than 0."
        confidence = 0.0
    # 5. Nyquist validation gate
    elif error is None and fmax >= nyquist:
        error = f"Upper frequency limit ({fmax} Hz) meets or exceeds the Nyquist limit ({nyquist} Hz at sfreq={sfreq} Hz)."
        confidence = 0.0
    # 6. Epoch length cycle check (only when actual duration is deterministically available)
    elif error is None:
        if duration_seconds is not None and duration_seconds > 0 and fmin > 0:
            min_duration = N_CYCLES_MIN / fmin
            if duration_seconds < min_duration:
                error = f"Epoch duration ({duration_seconds:.2f}s) is too short for reliable phase estimation at {fmin} Hz. Minimum recommended duration is {min_duration:.2f}s for {N_CYCLES_MIN} cycles."
                confidence = 0.0
                cycle_check_status = "failed"
            else:
                cycle_check_status = "passed"
        else:
            cycle_check_status = "skipped (duration unavailable)"

    needs_human_input = (confidence < SUPERVISOR_CONFIDENCE_THRESHOLD) or (error is not None)

    return {
        "name": resolved_name,
        "fmin": fmin,
        "fmax": fmax,
        "confidence": confidence,
        "needs_human_input": needs_human_input,
        "cycle_check": cycle_check_status,
        "error": error,
    }