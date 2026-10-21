"""Deep Agents context-management engine for the EEG Functional Connectivity Pipeline.

Implements the official LangChain Deep Agents context-management architecture:
1. Large tool result and input offloading to the filesystem with lightweight reference stubs.
2. Automatic sliding-window conversation history summarization when token limits approach.
3. Separation of persistent long-term memory (user preferences) from transient conversation state.
4. On-demand retrieval tools for inspecting previously offloaded context fragments.
5. Thread isolation and preservation of LangGraph HITL approval states.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Default token thresholds adhering to Deep Agents standards
# In EEG pipelines with constrained LLM endpoints (Groq 8k, vLLM), default trigger is conservative
DEFAULT_CONTEXT_TOKEN_LIMIT = int(os.getenv("DEEPAGENTS_CONTEXT_TOKEN_LIMIT", "8000"))
DEFAULT_TOOL_OFFLOAD_CHAR_THRESHOLD = int(os.getenv("DEEPAGENTS_TOOL_OFFLOAD_CHAR_THRESHOLD", "1500"))
DEFAULT_SUMMARIZATION_RATIO = float(os.getenv("DEEPAGENTS_SUMMARIZATION_RATIO", "0.85"))
DEFAULT_PRESERVED_TAIL_MESSAGES = int(os.getenv("DEEPAGENTS_PRESERVED_TAIL_MESSAGES", "4"))

DEEPAGENTS_SUMMARY_PROMPT = """You are a context-management assistant compressing an ongoing EEG analysis conversation.
Produce a concise, factual summary of the conversation history so far.
Include:
1. Stated analysis objective and dataset context.
2. Resolved scientific parameters (frequency band, channels, condition, metrics) and their values.
3. Tools invoked and their key findings or generated artifacts.
4. Pending questions or human-in-the-loop decisions currently awaiting resolution.

Do NOT invent information. Keep the summary under 300 words."""


def estimate_tokens(text_or_messages: Union[str, Sequence[BaseMessage]]) -> int:
    """Approximate token count (~4 characters per token) without requiring heavy dependencies."""
    if isinstance(text_or_messages, str):
        return max(1, len(text_or_messages) // 4)
    total_chars = 0
    for msg in text_or_messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    total_chars += len(part)
                elif isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
        # Account for tool call arguments
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            total_chars += len(json.dumps(tool_calls, default=str))
    return max(1, total_chars // 4)


@dataclass
class OffloadRecord:
    """Metadata for an offloaded tool result or message payload."""
    record_id: str
    file_path: Path
    char_length: int
    token_estimate: int
    tool_name: Optional[str] = None
    preview: str = ""


class FilesystemContextStore:
    """Thread-isolated filesystem storage for offloaded payloads and conversation records."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir is None:
            self.base_dir = Path("outputs") / "context_storage"
        else:
            self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _thread_dir(self, thread_id: str) -> Path:
        safe_id = re.sub(r"[^\w\-.]", "_", thread_id or "default_thread")
        td = self.base_dir / safe_id
        td.mkdir(parents=True, exist_ok=True)
        return td

    def save_payload(
        self,
        thread_id: str,
        payload_name: str,
        content: str,
    ) -> Path:
        """Write offloaded content to thread directory and return canonical path."""
        td = self._thread_dir(thread_id)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        safe_name = re.sub(r"[^\w\-.]", "_", payload_name)
        filename = f"{safe_name}_{digest}.txt"
        target_path = td / filename
        target_path.write_text(content, encoding="utf-8")
        return target_path

    def read_payload(self, file_path: Union[str, Path]) -> str:
        """Read content from offloaded file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Offloaded context file not found: {p}")
        return p.read_text(encoding="utf-8")


class DeepAgentsContextManager:
    """Orchestrates Deep Agents context management for Supervisor ReAct and Chainlit sessions."""

    def __init__(
        self,
        thread_id: str,
        token_limit: int = DEFAULT_CONTEXT_TOKEN_LIMIT,
        tool_offload_threshold: int = DEFAULT_TOOL_OFFLOAD_CHAR_THRESHOLD,
        store: Optional[FilesystemContextStore] = None,
        llm: Optional[Any] = None,
    ):
        self.thread_id = thread_id
        self.token_limit = token_limit
        self.tool_offload_threshold = tool_offload_threshold
        self.store = store or FilesystemContextStore()
        self.llm = llm
        self.offload_registry: Dict[str, OffloadRecord] = {}
        self.summaries: List[str] = []

    def maybe_offload_tool_result(
        self,
        tool_name: str,
        tool_output: Any,
        tool_call_id: Optional[str] = None,
    ) -> Tuple[Any, Optional[OffloadRecord]]:
        """If tool output exceeds threshold, offload to filesystem and return reference stub."""
        if isinstance(tool_output, (dict, list)):
            content_str = json.dumps(tool_output, indent=2, default=str)
        else:
            content_str = str(tool_output)

        if len(content_str) <= self.tool_offload_threshold:
            return tool_output, None

        # Content exceeds threshold: offload to filesystem
        record_id = f"{tool_name}_{tool_call_id or 'res'}"
        saved_path = self.store.save_payload(self.thread_id, record_id, content_str)
        tokens_est = estimate_tokens(content_str)

        lines = content_str.splitlines()
        preview = "\n".join(lines[:10])
        if len(lines) > 10:
            preview += f"\n... [{len(lines) - 10} additional lines offloaded]"

        record = OffloadRecord(
            record_id=record_id,
            file_path=saved_path,
            char_length=len(content_str),
            token_estimate=tokens_est,
            tool_name=tool_name,
            preview=preview,
        )
        self.offload_registry[record_id] = record

        stub_message = (
            f"[DEEP AGENTS OFFLOAD: Output of tool '{tool_name}' ({len(content_str)} chars / ~{tokens_est} tokens) "
            f"offloaded to: {saved_path.as_posix()}]\n\n"
            f"Preview (first 10 lines):\n```\n{preview}\n```\n\n"
            f"Use 'read_offloaded_context(key=\"{record_id}\")' to inspect full content if needed."
        )
        return stub_message, record

    def should_summarize(self, messages: Sequence[BaseMessage]) -> bool:
        """Check if message sequence approaches token budget limit."""
        current_tokens = estimate_tokens(messages)
        return current_tokens >= int(self.token_limit * DEFAULT_SUMMARIZATION_RATIO)

    def summarize_messages(
        self,
        messages: List[BaseMessage],
        preserved_tail: int = DEFAULT_PRESERVED_TAIL_MESSAGES,
    ) -> List[BaseMessage]:
        """Compress older messages into a structured summary message while preserving recent turns."""
        if len(messages) <= preserved_tail + 1:
            return list(messages)

        # Retain system prompt at index 0 if present
        has_system = isinstance(messages[0], SystemMessage)
        system_msg = messages[0] if has_system else None
        body_messages = messages[1:] if has_system else messages

        if len(body_messages) <= preserved_tail:
            return list(messages)

        cutoff = len(body_messages) - preserved_tail
        to_summarize = body_messages[:cutoff]
        tail_messages = body_messages[cutoff:]

        # Format text to summarize
        history_buffer = get_buffer_string(to_summarize)

        # Offload full historical text to filesystem as canonical record
        record_path = self.store.save_payload(
            self.thread_id,
            f"conversation_history_turn_{cutoff}",
            history_buffer,
        )

        # Generate summary (via LLM if available, otherwise deterministic structured extraction)
        summary_text = self._generate_summary(to_summarize, history_buffer)
        self.summaries.append(summary_text)

        summary_content = (
            f"[DEEP AGENTS CONTEXT SUMMARY: Previous {len(to_summarize)} messages compressed]\n\n"
            f"{summary_text}\n\n"
            f"(Full verbatim history preserved at: {record_path.as_posix()})"
        )
        summary_msg = HumanMessage(content=summary_content)

        new_messages: List[BaseMessage] = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append(summary_msg)
        new_messages.extend(tail_messages)

        logger.info(
            "Deep Agents context summarization: compressed %d messages into summary, preserved %d tail messages.",
            len(to_summarize),
            len(tail_messages),
        )
        return new_messages

    def _generate_summary(self, messages_to_summarize: Sequence[BaseMessage], buffer_text: str) -> str:
        """Generate structured summary of historical messages."""
        if self.llm is not None:
            try:
                prompt_msgs = [
                    SystemMessage(content=DEEPAGENTS_SUMMARY_PROMPT),
                    HumanMessage(content=f"Conversation to summarize:\n\n{buffer_text[:6000]}"),
                ]
                resp = self.llm.invoke(prompt_msgs)
                content = getattr(resp, "content", "")
                if content and isinstance(content, str) and content.strip():
                    return content.strip()
            except Exception as exc:
                logger.warning("LLM summarization failed (%s); falling back to deterministic extraction.", exc)

        # Deterministic extraction fallback
        lines: List[str] = ["**Key Context Summary:**"]
        tools_called = set()
        user_queries = []
        for msg in messages_to_summarize:
            if isinstance(msg, HumanMessage):
                text = str(msg.content).strip()
                if text and not text.startswith("[DEEP AGENTS"):
                    user_queries.append(text.splitlines()[0][:100])
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_called.add(tc.get("name", "unknown_tool"))

        if user_queries:
            lines.append(f"- User request intent: {user_queries[0]}")
            if len(user_queries) > 1:
                lines.append(f"- Subsequent refinements: {', '.join(user_queries[1:])}")
        if tools_called:
            lines.append(f"- Tools executed: {', '.join(sorted(tools_called))}")
        lines.append("- Prior stages completed; reasoning continues on latest parameters.")
        return "\n".join(lines)

    def build_retrieval_tools(self) -> List[Any]:
        """Create LangChain tools allowing agents to read or search previously offloaded context."""
        registry = self.offload_registry
        store = self.store

        @tool
        def read_offloaded_context(key: str) -> str:
            """Read the full body of a previously offloaded tool result or message payload.

            Args:
                key: The record_id shown in the offload reference (e.g. 'get_dataset_info_res').
            """
            rec = registry.get(key.strip())
            if not rec:
                available = ", ".join(sorted(registry.keys())) or "none"
                return f"Offload key '{key}' not found. Available keys: {available}"
            try:
                return store.read_payload(rec.file_path)
            except Exception as exc:
                return f"Error reading offloaded payload '{key}': {exc}"

        @tool
        def search_offloaded_context(query: str) -> str:
            """Search through all offloaded context files for lines matching query.

            Args:
                query: Search string or parameter name to look for across offloaded results.
            """
            matches = []
            q_lower = query.strip().lower()
            for key, rec in registry.items():
                try:
                    text = store.read_payload(rec.file_path)
                    for line_no, line in enumerate(text.splitlines(), start=1):
                        if q_lower in line.lower():
                            snippet = line.strip()
                            if len(snippet) > 160:
                                snippet = snippet[:157] + "..."
                            matches.append(f"- [{key}:{line_no}] {snippet}")
                            if len(matches) >= 25:
                                break
                except Exception:
                    continue
                if len(matches) >= 25:
                    break
            if not matches:
                return f"No matches found for '{query}' across offloaded context."
            return "Matches across offloaded context:\n" + "\n".join(matches)

        return [read_offloaded_context, search_offloaded_context]
