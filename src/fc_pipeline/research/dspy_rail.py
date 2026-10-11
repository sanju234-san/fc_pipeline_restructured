"""DSPy backend for the INPUT guardrail (opt-in: ``INPUT_RAIL_BACKEND=dspy``).

Replaces the NeMo ``self check input`` LLM question with the DSPy
``InputScopeClassifier`` program, whose prompt can be measured and optimised
(``scripts/optimize_input_scope.py``) instead of hand-edited. The output
rail is unchanged (NeMo ``self check facts``).

Contract: ``check_input_with_dspy`` returns the same dict shape as the NeMo
check in ``pipeline/nodes.py`` -- ``{"status", "content", "rail", "error"}``
with status ``passed`` / ``blocked`` / ``error`` -- so the existing action
policy and the Chainlit Allow / Block review card work unchanged. It never
raises, and any failure is ``error`` (fail closed -> human review).

It uses the same ``SUPERVISOR_LLM_*`` endpoint as the rest of the pipeline.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

RAIL_NAME = "dspy input scope"
MAX_INPUT_CHARS = 2000
DEFAULT_PROGRAM_PATH = "outputs/dspy/input_scope.json"

_lock = threading.Lock()
_cache: Dict[str, Any] = {}


def dspy_rail_enabled() -> bool:
    return os.getenv("INPUT_RAIL_BACKEND", "nemo").strip().lower() == "dspy"


def _clean(value: Optional[str]) -> str:
    return (value or "").strip().strip("\"'")


def _build_lm() -> Any:
    import dspy

    endpoint = _clean(os.getenv("SUPERVISOR_LLM_ENDPOINT"))
    model = _clean(os.getenv("SUPERVISOR_LLM_MODEL"))
    key = _clean(os.getenv("SUPERVISOR_LLM_API_KEY") or os.getenv("GROQ_API_KEY")) or "not-required"
    if not endpoint or not model:
        raise EnvironmentError("SUPERVISOR_LLM_ENDPOINT and SUPERVISOR_LLM_MODEL must be set.")
    return dspy.LM(f"openai/{model}", api_base=endpoint, api_key=key, temperature=0.0, max_tokens=300)


def _build_program() -> Any:
    from fc_pipeline.research.dspy_scope import InputScopeClassifier

    program = InputScopeClassifier()
    path = Path(_clean(os.getenv("INPUT_RAIL_DSPY_PROGRAM")) or DEFAULT_PROGRAM_PATH)
    if path.is_file():
        program.load(str(path))
        logger.info("DSPy input rail: loaded optimised program from %s", path.name)
    return program


def _get_lm_and_program() -> Any:
    with _lock:
        if "program" not in _cache:
            _cache["lm"] = _build_lm()
            _cache["program"] = _build_program()
        return _cache["lm"], _cache["program"]


def reset_cache() -> None:
    """Forget the cached LM / program (used by tests and after re-optimising)."""
    with _lock:
        _cache.clear()


def check_input_with_dspy(text: str, *, lm: Any = None, program: Any = None) -> Dict[str, Any]:
    """Classify one user message. Never raises."""
    try:
        import dspy

        if lm is None or program is None:
            cached_lm, cached_program = _get_lm_and_program()
            lm = lm or cached_lm
            program = program or cached_program
        with dspy.context(lm=lm):
            pred = program(message=str(text or "")[:MAX_INPUT_CHARS])
        verdict = str(getattr(pred, "verdict", "")).strip().lower()
        reason = str(getattr(pred, "reason", "") or "").strip()
        logger.info("DSPy input rail result: verdict=%s", verdict)
        if verdict == "allow":
            return {"status": "passed", "content": str(text), "rail": RAIL_NAME, "error": None}
        if verdict == "block":
            return {"status": "blocked", "content": reason, "rail": RAIL_NAME, "error": None}
        raise ValueError(f"unrecognised verdict {verdict!r}")
    except Exception as exc:  # fail closed -> the action policy asks a human
        logger.warning("DSPy input rail could not run: %s: %s", type(exc).__name__, exc)
        return {"status": "error", "content": "", "rail": None, "error": f"{type(exc).__name__}: {exc}"}
