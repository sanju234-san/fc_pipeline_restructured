"""GRPO reward functions for EEG Supervisor fine-tuning.

These functions are **pure Python + NumPy** — no LLM calls, no LangChain.
They read the canonical thresholds from ``fc_pipeline.config.thresholds``
so the same boundary values govern both the live pipeline and GRPO training.

Reward design
-------------
Each reward function takes a ``(prompt, completion)`` string pair and returns
a ``float`` in [0.0, 1.0].

Three reward signals are combined by ``composite_reward``:

1. **metric_classification_reward**: Does the completion correctly classify a
   connectivity value as ``phase_coupled``, ``zero_lag``, ``weak``, or
   ``ambiguous`` given the metric type and value embedded in the prompt?
   (Uses TAU_PHASE and TAU_ZEROLAG from config.)

2. **metric_set_reward**: Does the completion name a valid subset of the 5
   canonical metrics?  Full set → 1.0; any valid subset → 0.5; invalid name
   in list → 0.0.

3. **format_compliance_reward**: Does the completion follow the expected
   JSON-output format (has ``classification`` and ``metrics`` keys)?
   Structural check only — 0.0 or 0.5 partial credit or 1.0.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

import numpy as np

# Read canonical thresholds — single source of truth (AGENTS.md §1, rule 4)
from fc_pipeline.config.thresholds import TAU_PHASE, TAU_ZEROLAG

# Canonical metric names (AGENTS.md §1, rule 3)
VALID_METRICS: frozenset[str] = frozenset({"PLI", "wPLI", "ImCoh", "PLV", "Coherence"})

# Classification labels that the Supervisor is trained to produce
_VALID_CLASSIFICATIONS = frozenset(
    {"phase_coupled", "zero_lag", "weak", "ambiguous"}
)

# Weights for the composite reward
_W_CLASSIFICATION = 0.50
_W_METRIC_SET = 0.30
_W_FORMAT = 0.20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_completion_json(completion: str) -> Dict[str, Any] | None:
    """Try to extract a JSON object from the completion string.

    Returns the parsed dict or None if no valid JSON is found.
    """
    # First try the whole string
    try:
        return json.loads(completion.strip())
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block
    m = re.search(r"\{.*\}", completion, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _extract_connectivity_value(prompt: str) -> float | None:
    """Extract the numeric connectivity value embedded in the prompt.

    Prompts are expected to contain a line like:
        ``Connectivity value: 0.42``
    """
    m = re.search(r"connectivity[_ ]value[:\s]+([0-9.eE+\-]+)", prompt, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_metric_name(prompt: str) -> str | None:
    """Extract the metric name from the prompt (e.g. 'Metric: PLI')."""
    m = re.search(r"metric[:\s]+(\w+)", prompt, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        # Normalise wPLI capitalisation
        for v in VALID_METRICS:
            if v.lower() == candidate.lower():
                return v
    return None


def _expected_classification(metric: str | None, value: float) -> str:
    """Deterministic ground-truth classification using pipeline thresholds.

    Mirrors ``fc_pipeline.deterministic.connectivity.classification`` logic.
    """
    # Phase-sensitive metrics: PLI, wPLI, ImCoh, PLV
    # Volume-conduction-sensitive metrics: Coherence, ImCoh (zero-lag component)
    # For classification training we use the simpler rule from thresholds.py:
    #   value >= TAU_ZEROLAG  -> zero_lag (high value, likely volume-conducted)
    #   TAU_PHASE <= value < TAU_ZEROLAG -> phase_coupled
    #   value < TAU_PHASE -> weak
    if value >= TAU_ZEROLAG:
        return "zero_lag"
    elif value >= TAU_PHASE:
        return "phase_coupled"
    else:
        return "weak"


# ---------------------------------------------------------------------------
# Individual reward functions
# ---------------------------------------------------------------------------

def metric_classification_reward(prompt: str, completion: str) -> float:
    """Reward for correctly classifying the connectivity value.

    Returns
    -------
    float
        1.0 — correct classification
        0.5 — completion contains a valid label but it is wrong
        0.0 — no valid label found or parse failure
    """
    value = _extract_connectivity_value(prompt)
    if value is None:
        return 0.0  # Prompt malformed; reward is meaningless

    metric = _extract_metric_name(prompt)
    expected = _expected_classification(metric, value)

    parsed = _parse_completion_json(completion)
    if parsed is None:
        # Try plain text match
        found = None
        for label in _VALID_CLASSIFICATIONS:
            if label in completion:
                found = label
                break
        if found is None:
            return 0.0
        return 1.0 if found == expected else 0.5

    label = parsed.get("classification", "")
    if label not in _VALID_CLASSIFICATIONS:
        return 0.0
    return 1.0 if label == expected else 0.5


def metric_set_reward(prompt: str, completion: str) -> float:
    """Reward for producing a valid (sub)set of the 5 canonical metrics.

    Returns
    -------
    float
        1.0 — full canonical set named (all 5)
        0.5 — valid non-empty subset
        0.0 — at least one invalid metric name found
    """
    parsed = _parse_completion_json(completion)
    if parsed is None:
        # Scan free text for metric names
        found = [m for m in VALID_METRICS if m in completion]
        invalid = bool(re.search(
            r"\b(?!PLI\b|wPLI\b|ImCoh\b|PLV\b|Coherence\b)[A-Z][a-zA-Z]{2,}\b",
            completion,
        ))
        if invalid:
            return 0.0
        if not found:
            return 0.0
        return 1.0 if set(found) == VALID_METRICS else 0.5

    raw_metrics = parsed.get("metrics", [])
    if not isinstance(raw_metrics, list):
        return 0.0

    named = {str(m).strip() for m in raw_metrics}
    if not named:
        return 0.0
    invalid = named - VALID_METRICS
    if invalid:
        return 0.0  # Invented metric name
    return 1.0 if named == VALID_METRICS else 0.5


def format_compliance_reward(prompt: str, completion: str) -> float:  # noqa: ARG001
    """Reward for producing well-formed JSON with the expected keys.

    Returns
    -------
    float
        1.0 — valid JSON with both ``classification`` and ``metrics`` keys
        0.5 — valid JSON but missing one key
        0.0 — not parseable as JSON
    """
    parsed = _parse_completion_json(completion)
    if parsed is None:
        return 0.0
    has_cls = "classification" in parsed
    has_met = "metrics" in parsed
    if has_cls and has_met:
        return 1.0
    if has_cls or has_met:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Composite reward (used by GRPOTrainer)
# ---------------------------------------------------------------------------

def composite_reward(prompt: str, completion: str) -> float:
    """Weighted combination of all three reward signals.

    Parameters
    ----------
    prompt : str
        The input prompt shown to the model.
    completion : str
        The model-generated completion.

    Returns
    -------
    float
        Weighted reward in [0.0, 1.0].
    """
    r_cls = metric_classification_reward(prompt, completion)
    r_met = metric_set_reward(prompt, completion)
    r_fmt = format_compliance_reward(prompt, completion)
    return (
        _W_CLASSIFICATION * r_cls
        + _W_METRIC_SET * r_met
        + _W_FORMAT * r_fmt
    )


def batch_composite_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
) -> List[float]:
    """Vectorised composite reward for a batch (used by trl GRPOTrainer).

    Parameters
    ----------
    prompts : Sequence[str]
    completions : Sequence[str]

    Returns
    -------
    List[float]
        One reward per (prompt, completion) pair.
    """
    return [composite_reward(p, c) for p, c in zip(prompts, completions)]
