"""Synthetic EEG scenario generator for GRPO training.

Produces a ``datasets.Dataset`` of (prompt, chosen_response) pairs that
teach the Supervisor LLM to:

1. Correctly classify a connectivity value into one of four categories using
   the canonical thresholds from ``fc_pipeline.config.thresholds``.
2. Enumerate the correct metric set (all 5, or a valid user-specified subset).

No LLM calls are made here — all scenarios are generated deterministically
from NumPy random draws, using the same threshold constants as the live
pipeline.

Typical usage::

    from fc_pipeline.research.grpo_dataset import build_grpo_dataset
    ds = build_grpo_dataset(n_samples=500, seed=42)
    # ds is a HuggingFace Dataset with columns: prompt, chosen
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

import numpy as np

# Canonical thresholds — never re-defined (AGENTS.md §1, rule 4)
from fc_pipeline.config.thresholds import TAU_PHASE, TAU_ZEROLAG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_METRICS: List[str] = ["PLI", "wPLI", "ImCoh", "PLV", "Coherence"]

# Metric subsets frequently requested by users (besides the full set)
_COMMON_SUBSETS: List[List[str]] = [
    ["PLI", "wPLI"],
    ["PLI", "Coherence"],
    ["wPLI", "ImCoh"],
    ["PLV"],
    ["PLI", "wPLI", "ImCoh"],
    VALID_METRICS,  # full set
]

# Frequency band presets (name -> (fmin, fmax))
_FREQ_BANDS: Dict[str, tuple] = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 80),
}

_CONDITIONS: List[str] = ["rest", "task", "T0", "T1", "baseline", "active"]

_REGION_TO_CHANNELS: Dict[str, List[str]] = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "Fz"],
    "temporal": ["T7", "T8", "TP9", "TP10"],
    "occipital": ["O1", "O2", "Oz"],
    "parietal": ["P3", "P4", "Pz"],
    "central": ["C3", "C4", "Cz"],
}


# ---------------------------------------------------------------------------
# Classification ground-truth (mirrors live pipeline + reward fn)
# ---------------------------------------------------------------------------

def _classify(value: float) -> str:
    if value >= TAU_ZEROLAG:
        return "zero_lag"
    elif value >= TAU_PHASE:
        return "phase_coupled"
    else:
        return "weak"


# ---------------------------------------------------------------------------
# Prompt / response templates
# ---------------------------------------------------------------------------

def _build_prompt(
    metric: str,
    value: float,
    band: str,
    fmin: float,
    fmax: float,
    channels: List[str],
    condition: str,
    requested_metrics: List[str],
) -> str:
    ch_str = ", ".join(channels)
    req_str = ", ".join(requested_metrics) if requested_metrics != VALID_METRICS else "all"
    return (
        f"You are the Supervisor Agent for an EEG Functional Connectivity pipeline.\n\n"
        f"Dataset: EEG recording.\n"
        f"Analysis condition: {condition}\n"
        f"Frequency band: {band} ({fmin}-{fmax} Hz)\n"
        f"Channels: {ch_str}\n"
        f"Requested metrics: {req_str}\n"
        f"Metric: {metric}\n"
        f"Connectivity value: {value:.4f}\n\n"
        f"Classify this connectivity value and list the metrics that should be computed.\n"
        f"Respond ONLY with a JSON object with keys 'classification' and 'metrics'.\n"
        f"Valid classification labels: phase_coupled, zero_lag, weak, ambiguous.\n"
        f"Valid metric names: PLI, wPLI, ImCoh, PLV, Coherence.\n"
    )


def _build_chosen_response(
    value: float,
    requested_metrics: List[str],
) -> str:
    classification = _classify(value)
    return json.dumps(
        {"classification": classification, "metrics": requested_metrics},
        indent=None,
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_grpo_dataset(
    n_samples: int = 500,
    seed: int = 42,
    use_hf_datasets: bool = True,
) -> Any:
    """Generate a synthetic EEG scenario dataset for GRPO training.

    Parameters
    ----------
    n_samples : int
        Number of (prompt, chosen) pairs to generate.
    seed : int
        NumPy + random seed for reproducibility.
    use_hf_datasets : bool
        If ``True`` (default), returns a ``datasets.Dataset``.
        If ``False`` (e.g. when HuggingFace datasets is not installed),
        returns a plain ``list[dict]``.

    Returns
    -------
    datasets.Dataset or list[dict]
        Each record has keys: ``prompt``, ``chosen``.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    records: List[Dict[str, str]] = []

    bands = list(_FREQ_BANDS.items())
    regions = list(_REGION_TO_CHANNELS.items())

    for _ in range(n_samples):
        # Sample a random connectivity value across the full [0, 1] range,
        # with slight over-sampling of boundary regions for better gradient signal
        bucket = rng.integers(0, 3)  # 0=weak, 1=phase_coupled, 2=zero_lag
        if bucket == 0:
            value = float(rng.uniform(0.0, TAU_PHASE))
        elif bucket == 1:
            value = float(rng.uniform(TAU_PHASE, TAU_ZEROLAG))
        else:
            value = float(rng.uniform(TAU_ZEROLAG, 1.0))

        metric = random.choice(VALID_METRICS)
        band_name, (fmin, fmax) = random.choice(bands)
        _, channels_pool = random.choice(regions)
        n_ch = rng.integers(2, min(5, len(channels_pool)) + 1)
        channels = random.sample(channels_pool, int(n_ch))
        condition = random.choice(_CONDITIONS)
        requested_metrics = random.choice(_COMMON_SUBSETS)

        prompt = _build_prompt(
            metric=metric,
            value=value,
            band=band_name,
            fmin=fmin,
            fmax=fmax,
            channels=channels,
            condition=condition,
            requested_metrics=requested_metrics,
        )
        chosen = _build_chosen_response(
            value=value,
            requested_metrics=requested_metrics,
        )
        records.append({"prompt": prompt, "chosen": chosen})

    if not use_hf_datasets:
        return records

    try:
        from datasets import Dataset  # type: ignore[import]
        return Dataset.from_list(records)
    except ImportError:
        return records


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a synthetic GRPO training dataset for the EEG pipeline."
    )
    parser.add_argument("--n", type=int, default=200, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path")
    args = parser.parse_args()

    ds = build_grpo_dataset(n_samples=args.n, seed=args.seed, use_hf_datasets=False)
    if args.output:
        import pathlib
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for rec in ds:
                fh.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(ds)} records to {out}")
    else:
        print(f"Generated {len(ds)} records (dry run, no --output specified).")
        print("Sample prompt:\n" + ds[0]["prompt"])
        print("Sample chosen:\n" + ds[0]["chosen"])
