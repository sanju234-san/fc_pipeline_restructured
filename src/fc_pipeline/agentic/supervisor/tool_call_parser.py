"""
Fallback parser for raw text-embedded tool calls (e.g., Mistral/Ministral syntax).

When local inference servers (e.g., vLLM without `--tool-call-parser mistral`) output
raw tokenizer format `[TOOL_CALLS]tool_name[ARGS]{...}` into `message.content` instead
of populating the OpenAI `tool_calls` schema, this module extracts and normalizes them
into standard LangChain tool-call dictionaries.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


def _extract_json_block(text: str, start_pos: int) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Extract a balanced JSON object starting at or after `start_pos`.

    Uses `json.JSONDecoder.raw_decode` to robustly parse arbitrary nested JSON objects
    without relying on fragile regex matching.

    Returns:
        (parsed_dict, end_pos) if successful, or (None, start_pos) if invalid.
    """
    decoder = json.JSONDecoder()
    # Find the opening '{'
    brace_idx = text.find("{", start_pos)
    if brace_idx == -1:
        return None, start_pos

    try:
        obj, end_idx = decoder.raw_decode(text[brace_idx:])
        if isinstance(obj, dict):
            return obj, brace_idx + end_idx
        return None, start_pos
    except json.JSONDecodeError:
        return None, start_pos


def extract_mistral_style_tool_calls(response_content: str) -> List[Dict[str, Any]]:
    """
    Extracts all Mistral-formatted `[TOOL_CALLS]name[ARGS]{...}` tool calls from raw response text.

    Args:
        response_content: Raw string content from the LLM AIMessage.

    Returns:
        List of standard LangChain tool-call dictionaries:
        [
            {
                "name": str,
                "args": Dict[str, Any],
                "id": str,
                "type": "tool_call"
            },
            ...
        ]

    Examples:
        Single tool call:
        >>> text = '[TOOL_CALLS]get_dataset_info[ARGS]{"data_path": "outputs/eeg.fif"}'
        >>> extract_mistral_style_tool_calls(text)
        [{'name': 'get_dataset_info', 'args': {'data_path': 'outputs/eeg.fif'}, 'id': '...', 'type': 'tool_call'}]

        Multiple tool calls with nested JSON arguments:
        >>> text = '[TOOL_CALLS]resolve_frequency_band[ARGS]{"query": "alpha", "opts": {"loose": true}}[TOOL_CALLS]get_dataset_conditions[ARGS]{"data_path": "data.fif"}'
        >>> calls = extract_mistral_style_tool_calls(text)
        >>> len(calls)
        2
        >>> calls[0]['name']
        'resolve_frequency_band'
        >>> calls[0]['args']['opts']['loose']
        True

        Malformed JSON (skipped gracefully):
        >>> text = '[TOOL_CALLS]broken_tool[ARGS]{invalid_json}'
        >>> extract_mistral_style_tool_calls(text)
        []
    """
    if not response_content or "[TOOL_CALLS]" not in response_content:
        return []

    tool_calls: List[Dict[str, Any]] = []

    # Find all occurrences of [TOOL_CALLS]<tool_name>[ARGS]
    pattern = re.compile(r"\[TOOL_CALLS\]\s*([A-Za-z0-9_]+)\s*\[ARGS\]", re.IGNORECASE)

    for match in pattern.finditer(response_content):
        tool_name = match.group(1).strip()
        args_start_pos = match.end()

        # Parse balanced JSON object immediately following [ARGS]
        args_dict, _ = _extract_json_block(response_content, args_start_pos)
        if args_dict is not None:
            tool_calls.append({
                "name": tool_name,
                "args": args_dict,
                "id": uuid.uuid4().hex[:9],
                "type": "tool_call",
            })

    return tool_calls
