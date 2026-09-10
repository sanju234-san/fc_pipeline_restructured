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
    "beta": {"fmin": 12.0, "fmax": 30.0},
    "gamma": {"fmin": 30.0, "fmax": 100.0},
}


@tool
def resolve_frequency_band(
    band_name_or_range: str, sfreq: float, duration_seconds: float
) -> Dict[str, Any]:
    """Resolves natural-language band names or numeric ranges to explicit fmin and fmax boundaries with confidence scoring.
    
    Rejects bands exceeding the Nyquist limit (sfreq / 2) and flags needs_human_input when confidence is below threshold.
    Also checks if the epoch duration is sufficient for the requested band's lowest frequency.
    """
    cleaned = band_name_or_range.strip().lower()
    nyquist = sfreq / 2.0 if sfreq > 0 else 0.0
    
    resolved_name: Optional[str] = band_name_or_range  # Echo input by default for unknown bands
    fmin: float = 0.0
    fmax: float = 0.0
    confidence: float = 0.0
    error: Optional[str] = None

    # 1. First resolve canonical band values regardless of sfreq
    if cleaned in CANONICAL_BANDS:
        resolved_name = cleaned
        fmin = CANONICAL_BANDS[cleaned]["fmin"]
        fmax = CANONICAL_BANDS[cleaned]["fmax"]
        confidence = 1.0

    # 2. Explicit numeric range match (e.g., "8-12", "8.0 to 13", "8 - 12 Hz")
    else:
        num_pattern = r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)"
        match = re.search(num_pattern, cleaned)
        if match:
            fmin = float(match.group(1))
            fmax = float(match.group(2))
            resolved_name = "custom"
            if fmin <= 0:
                error = f"Lower frequency bound ({fmin} Hz) must be strictly greater than 0.0 Hz."
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
    # 6. Epoch length cycle check
    elif error is None and duration_seconds > 0 and fmin > 0:
        min_duration = N_CYCLES_MIN / fmin
        if duration_seconds < min_duration:
            error = f"Epoch duration ({duration_seconds:.2f}s) is too short for reliable phase estimation at {fmin} Hz. Minimum recommended duration is {min_duration:.2f}s for {N_CYCLES_MIN} cycles."
            confidence = 0.0

    needs_human_input = (confidence < SUPERVISOR_CONFIDENCE_THRESHOLD) or (error is not None)

    return {
        "name": resolved_name,
        "fmin": fmin,
        "fmax": fmax,
        "confidence": confidence,
        "needs_human_input": needs_human_input,
        "error": error,
    }