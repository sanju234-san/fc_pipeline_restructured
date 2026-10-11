"""Context window manager for post-run Deep Agents follow-ups.

Implements the Deep Agents context-management strategy at the application
layer:
  * Progressive disclosure — the entire run context JSON is NOT loaded into
    every prompt.  Only the slice needed for the current query is injected.
  * Sliding window with summarisation of older turns so a long chat does not
    overflow the model's token budget.
  * Offloading of large artefact payloads (plot image descriptions, full
    evidence tables) to a "virtual file" registry that is only read on demand
    via the harness's read_file / grep tools when the subagent actually needs
    the full content.

This module is deliberately free of LangChain / Deep Agents runtime imports
so the compression and selection logic can be unit-tested offline without an
LLM endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import textwrap
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from fc_pipeline.pipeline.run_context import (
    _TOPICS,
    answer_run_question,
    looks_like_new_analysis,
)

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKEN_WINDOW: int = 8000
"""Maximum tokens allowed inside the LLM prompt for run-context history.

Older turns are summarised / offloaded once the cumulative estimate exceeds
this budget.  Budget is a rough token estimate (~4 chars per token) because
an exact tiktoken count is not available offline without pulling in a token
library that may not be installed.
"""

DEFAULT_CONTEXT_TOKEN_BUDGET: int = DEFAULT_CONTEXT_TOKEN_WINDOW

_ARTIFACT_WORDS = re.compile(
    r"\b(plots?|graphs?|figures?|visuali[sz]\w*|psd|spectr\w*|signals?|compare|comparison|"
    r"images?|pictures?|heatmap|network|before\s+(and|vs\.?|versus)\s+after)\b",
    re.IGNORECASE,
)

_NEW_ANALYSIS_VERB = re.compile(
    r"^\s*(please\s+|now\s+|can you\s+|could you\s+)?(re-?run|re-?analy[sz]e?|redo|repeat|analy[sz]e|compute|"
    r"calculate|run|try|do|use|switch to|change)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OffloadedTurn:
    """An older chat turn that has been summarised and moved out of the active window."""

    turn_index: int
    summary: str
    role: str
    """user | assistant."""
    tokens_estimate: int
    full_text_ref: str
    """Lookup key into the offload registry if the full text is ever needed."""


@dataclass
class ContextSelection:
    """The exact run-context slice + chat window the orchestrator should inject.

    Returned by :meth:`PostRunContextWindowManager.prepare_for_query` and
    consumed by the agent builder to build the Deep Agents system prompt and
    the ``messages`` list.
    """

    # --- Active prompt content ------------------------------------------------
    system_prompt_fragment: str
    """Inject this into the Deep Agents harness ``system_prompt``."""

    active_messages: List[Dict[str, str]]
    """Messages to append to the harness call: {"role": "user"|"assistant", "content": ...}."""

    # --- Offloaded / summarised content ---------------------------------------
    offloaded_turns: List[OffloadedTurn] = field(default_factory=list)
    """Old turns that were summarised and moved out of the active window."""

    offloaded_context_files: Dict[str, str] = field(default_factory=dict)
    """key -> markdown text that a subagent can read by asking for the key.

    Populated with items like ``"run_plan"``, ``"data_prep_details"``,
    ``"manifest_elevated_risk"`` when those topics are not needed right now.
    """

    # --- Routing metadata ----------------------------------------------------
    routed_intent: str = "run_context_qa"
    """One of: ``run_context_qa`` | ``plot_interpreter`` | ``new_analysis``."""

    estimated_active_tokens: int = 0
    """Rough estimate of how many tokens the active_messages + system fragment
    consume.  Used by tests and by the caller to sanity-check the window."""


# ---------------------------------------------------------------------------
# Core manager
# ---------------------------------------------------------------------------


class PostRunContextWindowManager:
    """Builds the context window for each follow-up query.

    Usage
    -----
    ::

        manager = PostRunContextWindowManager(run_context=stored_ctx_dict)
        for each user message:
            manager.push_user(message)
            selection = manager.prepare_for_query(message)
            result = run_followup_agent(user=message, selection=selection, ...)
            manager.push_assistant(result.assistant_text)
            manager.set_delegation_used(result.delegated_to)

    Offline mode (no LLM):
        :meth:`answer_deterministically` reproduces the exact
        :func:`answer_run_question` behaviour that currently powers the UI, so
        the same contract is preserved before the Deep Agents harness takes
        over for non-trivial follow-ups.
    """

    def __init__(
        self,
        run_context: Mapping[str, Any],
        *,
        token_window: int = DEFAULT_CONTEXT_TOKEN_WINDOW,
        summariser: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> None:
        self.run_context: Dict[str, Any] = dict(run_context) if run_context else {}
        self.token_window = token_window
        self._summariser = summariser or self._default_summariser
        self._lock = threading.Lock()

        self._active_turns: List[Dict[str, Any]] = []
        """Elements: {"role": "user"|"assistant", "content": str, "tokens": int, "turn": int}."""

        self._offloaded_turns: List[OffloadedTurn] = []
        self._turn_counter = 0

        # Registry of "large" context chunks that are only referenced by name
        # inside the active prompt.  Each key can be returned to the agent via
        # its tools when it asks for them.
        self._offload_registry: Dict[str, str] = self._build_offload_registry(self.run_context)

        # Track which specialist handled the previous turn, so context
        # selection can prefer to keep the last-delegate's topic slice loaded.
        self._last_delegate: Optional[str] = None

    # ------------------------------------------------------------------
    # Offload registry construction
    # ------------------------------------------------------------------

    def _build_offload_registry(self, ctx: Mapping[str, Any]) -> Dict[str, str]:
        """Slice the stored run context into named, offload-able documents.

        Only the INDEX (topic -> key mapping) lives in the active window; the
        full body of each key is only loaded when the delegated subagent asks
        for it explicitly via a tool call.
        """
        reg: Dict[str, str] = {}

        plan = ctx.get("plan") or {}
        band = plan.get("freq_band") if isinstance(plan, Mapping) else {}
        dp = ctx.get("data_prep_summary") or {}
        manifest = ctx.get("parameter_manifest") or []

        reg["run_plan_summary"] = textwrap.dedent(
            f"""
            # Run Plan

            - run_id: {ctx.get("run_id", "unknown")}
            - Frequency band: {band.get("name", "?")} {band.get("fmin", "?")}-{band.get("fmax", "?")} Hz
            - Condition: {plan.get("condition", "?")}
            - Channels ({len(plan.get("channels") or [])}): {", ".join(str(c) for c in (plan.get("channels") or []))}
            - Metrics: {", ".join(str(m) for m in (plan.get("metrics") or []))}
            """
        ).strip()

        reg["data_prep_details"] = textwrap.dedent(
            f"""
            # Data Preparation Details

            - Original channels: {dp.get("original_channel_count", "?")}
            - Retained channels: {dp.get("retained_channel_count", "?")}
            - Dropped channels: {", ".join(str(c) for c in (ctx.get("bad_channels_dropped") or dp.get("dropped_channels") or [])) or "none"}
            - Reference applied: {dp.get("reference_applied", "?")}
            - Reference detected: {dp.get("reference_detected", "?")}
            - Filter: {dp.get("filter_l_freq", "?")}-{dp.get("filter_h_freq", "?")} Hz
            - Sampling frequency: {dp.get("sampling_frequency", "?")} Hz
            - Epochs: {dp.get("epoch_count", "?")} of {dp.get("epoch_duration_seconds", "?")} s
            - Condition epochs extracted for: {dp.get("condition", plan.get("condition", "?"))}
            """
        ).strip()

        elevated = [
            f"- {row.get('name')}: proposed={row.get('proposed_value')} approved={row.get('human_approved_value')}"
            for row in manifest
            if str(row.get("risk_tier", "")).lower() == "elevated"
        ]
        reg["manifest_elevated_risk"] = textwrap.dedent(
            """
            # Elevated-Risk Manifest Values

            These parameters directly change scientific classification outcomes.
            """ + ("\n".join(elevated) if elevated else "\n(No elevated-risk rows recorded for this run.)")
        ).strip()

        connectivity = ctx.get("connectivity")
        if connectivity:
            reg["connectivity_summary"] = (
                "# Connectivity Stage Summary\n"
                + json.dumps(connectivity, indent=2, ensure_ascii=False, default=str)
            )

        # Before / after plot metadata (NOT the binary PNGs — those stay on disk)
        plots = ctx.get("channel_plot_paths") or {}
        reg["plot_index"] = textwrap.dedent(
            f"""
            # Available Plots

            - channels_before: {plots.get("before", "not generated")}
            - channels_after:  {plots.get("after", "not generated")}
            - preprocessed epochs file: {ctx.get("preprocessed_data_path", "none")}
            """
        ).strip()

        return reg

    # ------------------------------------------------------------------
    # Turn accounting
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    @staticmethod
    def _default_summariser(turns: Sequence[Dict[str, str]]) -> str:
        """Fallback summariser used when the caller did not provide an LLM one.

        Produces a deterministic, rule-based bullet summary so offline tests
        and the no-LLM path have a well-defined contraction.
        """
        bullets: List[str] = []
        for t in turns:
            role = str(t.get("role", "")).capitalize()
            content = str(t.get("content", ""))
            stripped = re.sub(r"\s+", " ", content).strip()
            if len(stripped) > 160:
                stripped = stripped[:157] + "…"
            bullets.append(f"{role}: {stripped}")
        return "\n".join(bullets)

    def push_user(self, content: str) -> None:
        with self._lock:
            self._turn_counter += 1
            self._active_turns.append(
                {
                    "role": "user",
                    "content": content,
                    "tokens": self._estimate_tokens(content),
                    "turn": self._turn_counter,
                }
            )

    def push_assistant(self, content: str) -> None:
        with self._lock:
            self._turn_counter += 1
            self._active_turns.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tokens": self._estimate_tokens(content),
                    "turn": self._turn_counter,
                }
            )

    def set_delegation_used(self, delegate_name: Optional[str]) -> None:
        self._last_delegate = delegate_name

    # ------------------------------------------------------------------
    # Intent routing (context layer)
    # ------------------------------------------------------------------

    @staticmethod
    def route_intent(query: str) -> str:
        """Classify the user's follow-up without running an LLM.

        Returns one of ``run_context_qa`` | ``plot_interpreter`` | ``new_analysis``.
        The declarative subagent declarations use the same string as their
        ``trigger`` field so the orchestrator can send the task straight to
        the specialist.
        """
        if not query or not query.strip():
            return "run_context_qa"
        q = query.strip()
        if looks_like_new_analysis(q) or _NEW_ANALYSIS_VERB.match(q):
            return "new_analysis"
        if _ARTIFACT_WORDS.search(q):
            return "plot_interpreter"
        return "run_context_qa"

    # ------------------------------------------------------------------
    # Topic -> offload key mapping (progressive disclosure)
    # ------------------------------------------------------------------

    _TOPIC_OFFLOAD_MAP: Dict[str, List[str]] = {
        "summary": ["run_plan_summary", "data_prep_details", "manifest_elevated_risk"],
        "run_id": ["run_plan_summary"],
        "channels": ["run_plan_summary", "data_prep_details"],
        "band": ["run_plan_summary"],
        "condition": ["run_plan_summary"],
        "metrics": ["run_plan_summary", "connectivity_summary"],
        "reference": ["data_prep_details"],
        "epochs": ["data_prep_details"],
        "sampling": ["data_prep_details"],
        "filter": ["data_prep_details"],
        "dropped": ["data_prep_details"],
        "output": ["data_prep_details", "plot_index"],
    }

    def _select_offload_keys_for_query(self, query: str, intent: str) -> List[str]:
        """Return the offload-registry keys whose content should be *directly*
        promoted into the active window for this query.  Everything else
        remains available only via the agent's tools.
        """
        keys: List[str] = []
        if intent == "plot_interpreter":
            keys.append("plot_index")
            keys.append("data_prep_details")
        elif intent == "new_analysis":
            # New analysis only needs the plan for context of what was already
            # done — do NOT dump elevated risk or data prep details.
            keys.append("run_plan_summary")
        else:  # run_context_qa
            matched = [name for name, pattern in _TOPICS.items() if pattern.search(query)]
            if "summary" in matched:
                matched = ["summary"]
            if not matched:
                # Catch-all: bring in the short plan summary for free so the
                # agent has a skeleton answer if the user asks something that
                # the topic detector missed.
                keys.append("run_plan_summary")
            else:
                for name in matched:
                    keys.extend(self._TOPIC_OFFLOAD_MAP.get(name, ["run_plan_summary"]))

        seen = set()
        ordered: List[str] = []
        for k in keys:
            if k not in seen and k in self._offload_registry:
                seen.add(k)
                ordered.append(k)
        return ordered

    # ------------------------------------------------------------------
    # Window compression (sliding + summarise)
    # ------------------------------------------------------------------

    def _compress_active_window_if_needed(self) -> None:
        """Evict old turns from the active window once it exceeds the budget.

        Implements the Deep Agents "summarisation and context offloading"
        contract: old turns are collapsed into a single ``OffloadedTurn`` via
        the summariser and removed from ``_active_turns``.  A reference is
        kept so the harness can still read the full text on demand.
        """
        total = sum(int(t.get("tokens", 0)) for t in self._active_turns)
        if total <= self.token_window:
            return

        keep_fraction = 0.4  # Keep the newest 40 % of turns verbatim
        total_turns = len(self._active_turns)
        keep_count = max(2, int(total_turns * keep_fraction))
        evict_count = total_turns - keep_count
        # Always evict an even number (user + assistant pairs) to avoid
        # dangling half-turns in the active sequence.
        evict_count = evict_count - (evict_count % 2)
        if evict_count <= 0:
            return

        to_evict = self._active_turns[:evict_count]
        evicted_messages = [{"role": t["role"], "content": t["content"]} for t in to_evict]
        summary_text = self._summariser(evicted_messages)
        ref_hash = hashlib.sha1(
            json.dumps(evicted_messages, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:10]
        self._offloaded_turns.append(
            OffloadedTurn(
                turn_index=to_evict[0]["turn"],
                summary=summary_text,
                role="assistant",
                tokens_estimate=self._estimate_tokens(summary_text),
                full_text_ref=f"chat_turns_evicted_{ref_hash}",
            )
        )
        self._offload_registry[self._offloaded_turns[-1].full_text_ref] = "\n\n".join(
            f"[{t['turn']:03d}] {t['role'].upper()}: {t['content']}" for t in to_evict
        )
        self._active_turns = self._active_turns[evict_count:]

    # ------------------------------------------------------------------
    # Public: build the selection for a query
    # ------------------------------------------------------------------

    def prepare_for_query(self, query: str) -> ContextSelection:
        """Build the exact prompt payload the orchestrator should feed the harness.

        * Routes the intent.
        * Promotes only the needed offload-registry documents into the system
          fragment (progressive disclosure).
        * Runs sliding-window compression on older chat turns.
        * Returns a :class:`ContextSelection`.
        """
        with self._lock:
            self._compress_active_window_if_needed()
            intent = self.route_intent(query)
            promoted_keys = self._select_offload_keys_for_query(query, intent)

            promoted_docs: List[str] = []
            offload_files: Dict[str, str] = {}
            for key, body in self._offload_registry.items():
                if key in promoted_keys:
                    promoted_docs.append(f"## {key}\n{body}")
                else:
                    offload_files[key] = body

            # Index of everything that is offloaded, so the agent knows which
            # keys exist and can ask for them via its tools.
            offload_index_lines = ["## Offloaded context registry (use read_file tool to retrieve full body)"]
            for key in sorted(self._offload_registry.keys()):
                if key in promoted_keys:
                    status = "✅ loaded above"
                else:
                    n_chars = len(self._offload_registry[key])
                    status = f"⏳ offloaded ({n_chars} chars)"
                offload_index_lines.append(f"- `{key}` — {status}")
            offload_index = "\n".join(offload_index_lines)

            # Build the system fragment: it is concatenated ON TOP of the
            # harness's own system_prompt.  Declarative subagents each have
            # their own instructions, so this fragment only carries the
            # grounded run-context slice.
            system_fragment = textwrap.dedent(
                """
                # Grounded Run Context (Post-run follow-up)

                You are answering questions about a SINGLE COMPLETED EEG
                functional-connectivity pipeline run.  All your answers must
                be STRICTLY GROUNDED in the material provided below and in
                the offload registry.  NEVER invent numbers, channel names,
                epoch counts, or thresholds.  When a value is not available
                say so explicitly instead of guessing.

                If the user asks for a NEW analysis (different band, channels,
                metrics, etc.) do NOT try to answer from the current context.
                Delegate to the NewAnalysisDispatcher specialist which will
                guide the user to press 🆕 New query.

                --- Context currently loaded into this window ---
                {promoted_docs}

                {offload_index}

                --- Older chat (summarised; full text available via read_file) ---
                {offloaded_chat}
                """
            ).strip()
            promoted_section = "\n\n".join(promoted_docs) if promoted_docs else "_(Nothing promoted — use the registry tools.)_"
            if self._offloaded_turns:
                offloaded_chat = "\n".join(
                    f"- Turn {o.turn_index}…{self._offloaded_turns[-1].turn_index if i == len(self._offloaded_turns)-1 else o.turn_index + max(1, (len(self._active_turns) // max(1,len(self._active_turns))))} summary: {o.summary}"
                    for i, o in enumerate(self._offloaded_turns)
                )
            else:
                offloaded_chat = "_(No older turns.)_"

            system_fragment = system_fragment.format(
                promoted_docs=promoted_section,
                offload_index=offload_index,
                offloaded_chat=offloaded_chat,
            )

            active_messages = [
                {"role": str(t["role"]), "content": str(t["content"])} for t in self._active_turns
            ]

            estimate = self._estimate_tokens(system_fragment) + sum(
                int(t.get("tokens", 0)) for t in self._active_turns
            )

            return ContextSelection(
                system_prompt_fragment=system_fragment,
                active_messages=active_messages,
                offloaded_turns=list(self._offloaded_turns),
                offloaded_context_files=offload_files,
                routed_intent=intent,
                estimated_active_tokens=estimate,
            )

    # ------------------------------------------------------------------
    # Deterministic fallback (no LLM / offline tests)
    # ------------------------------------------------------------------

    def answer_deterministically(self, query: str) -> Optional[str]:
        """Reproduce the existing :func:`answer_run_question` contract.

        Returns ``None`` when the question needs the plot interpreter or a
        new analysis dispatch — those MUST go through the Deep Agents harness
        because they involve multimodal reasoning or UI navigation.
        """
        intent = self.route_intent(query)
        if intent != "run_context_qa":
            return None
        return answer_run_question(query, self.run_context)
