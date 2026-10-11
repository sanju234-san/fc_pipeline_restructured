"""Deterministic pre-filter in front of the NeMo input rail.

Why this exists
---------------
The NeMo ``self check input`` rail asks the Supervisor LLM a Yes/No question.
Small Groq / vLLM models answer that unreliably, and any rail failure is
treated as fail-closed, so plain requests such as "analyze this EEG" were
blocked.  This module lets *obviously benign, in-scope* requests skip the
LLM classifier.  It never approves anything on its own authority beyond
"do not spend an LLM call screening this": the Query Transformer's intent
classification and the Supervisor's tool scoping still run afterwards.

Policy (conservative):
  * pass only short messages that contain EEG-analysis vocabulary, AND
  * contain none of the injection / code-execution / URL markers below.
Everything else still goes to the LLM rail, unchanged.

Disable with ``NEMO_INPUT_FASTPATH=false``.
"""

from __future__ import annotations

import os
import re

MAX_FASTPATH_CHARS = 300

_EEG_VOCAB = re.compile(
    r"\b("
    r"eeg|electrode|electrodes|channel|channels|connectivity|coherence|"
    r"plv|pli|wpli|imcoh|alpha|beta|theta|delta|gamma|"
    r"epoch|epochs|frontal|parietal|occipital"
    r")\b",
    re.IGNORECASE,
)

# Any of these sends the message to the full LLM rail.
_SUSPICIOUS = re.compile(
    r"("
    r"ignore|disregard|forget|override|bypass|jailbreak|developer mode|"
    r"system prompt|previous instruction|prior instruction|your instruction|"
    r"pretend|role-?play|act as|you are now|"
    r"approve (the )?(gate|plan)|gate_1|set .*approved|"
    r"base64|exec\b|eval\b|subprocess|os\.system|shell|terminal|sudo|rm -|"
    r"https?://|www\.|<\||\|>|```|\{\{|\}\}|<script|<system|</"
    r")",
    re.IGNORECASE,
)


def fastpath_enabled() -> bool:
    return os.getenv("NEMO_INPUT_FASTPATH", "true").strip().lower() not in {"0", "false", "no", "off"}


def is_obviously_in_scope(text: str) -> bool:
    """True only for short, plainly EEG-related text with no suspicious markers."""
    if not fastpath_enabled():
        return False
    if not text or not text.strip():
        return False
    cleaned = text.strip()
    if len(cleaned) > MAX_FASTPATH_CHARS:
        return False
    if _SUSPICIOUS.search(cleaned):
        return False
    return bool(_EEG_VOCAB.search(cleaned))
