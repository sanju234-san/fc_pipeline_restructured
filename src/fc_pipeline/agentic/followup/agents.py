"""Deep Agents orchestrator for post-run follow-ups.

Architecture (what actually runs):
  * New-analysis requests short-circuit to a "press New query" reply (no LLM).
  * Plain run-context questions are answered deterministically when a scripted
    topic matches.
  * Questions about the dataset / recording itself are answered from the file's
    own header and annotations (:mod:`fc_pipeline.toolbox.dataset.recording_facts`).
  * Plot questions go to the vision model in one direct multimodal call, because
    the Deep Agents ``task`` tool can only hand text to a subagent.
  * Everything else goes to ONE coordinator built with ``create_deep_agent`` that
    delegates to a single ``RunContextQAAgent`` subagent (a Deep Agents
    ``SubAgent`` dict). Custom tools expose the offload registry
    (``read_offloaded_context``, ``search_run_context``) so large context is
    fetched on demand.
  * If the harness cannot run, the deterministic answerer is used and the
    result says which fallback answered.

The context manager in :mod:`context_window` has already shrunk the active
window by the time this module runs.  The Deep Agents harness then takes over:

* **Summarisation middleware** (built into ``create_deep_agent`` by default)
  further compresses message history inside each subagent's isolated context
  window when long follow-ups push against the token budget.

* **Subagent isolation** (``delegation``) means each specialist sees ONLY the
  task-specific instructions + the grounded run-context slice — never the
  routing plumbing or the other specialist's prompts.

All LLM invocations in this module use the SAME provider as the Supervisor —
see :func:`fc_pipeline.agentic.supervisor.llm_provider.get_supervisor_llm`.
"""

from __future__ import annotations

import base64
import logging
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from fc_pipeline.agentic.followup.context_window import (
    ContextSelection,
    PostRunContextWindowManager,
)
from fc_pipeline.pipeline.run_context import answer_run_question

logger = logging.getLogger(__name__)


class FollowUpResultKind(str, Enum):
    """Possible outcomes of a follow-up invocation."""

    ANSWERED = "answered"
    """A grounded answer was produced; show to the user."""

    DELEGATED = "delegated"
    """Task was delegated to a specialist (informational — still has answer text)."""

    DISPATCH_NEW_QUERY = "dispatch_new_query"
    """User asked for a NEW analysis; UI should prompt the "new query" confirmation."""

    NEEDS_PLOT = "needs_plot"
    """Plot interpreter returned an image + caption."""

    ERROR = "error"
    """Harness or tool failed; a fallback answer is attached."""


@dataclass
class FollowUpResult:
    """Value-object returned to the Chainlit UI handler."""

    kind: FollowUpResultKind
    assistant_text: str
    delegated_to: Optional[str] = None
    """Name of the subagent that produced the answer, for the context manager."""
    image_paths: List[str] = field(default_factory=list)
    """Inline image paths (only for ``NEEDS_PLOT``)."""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Deep Agents model-string adapter
# ---------------------------------------------------------------------------


def _build_deepagents_model_string() -> str:
    """Translate the env-configured Supervisor LLM endpoint to a deepagents
    ``model="provider:name"`` string.

    The Deep Agents harness uses OpenAI, Anthropic, etc. prefix strings.  The
    pipeline's own LLM is always OpenAI-compatible REST, so we use
    ``"openai:gpt-4o-mini"`` as a well-supported default when deepagents is
    the active harness.  The caller can override with ``DEEPAGENTS_MODEL``.

    NOTE: Deep Agents accepts a ``model=`` string in ``create_deep_agent`` but
    also accepts a full LangChain ``ChatModel`` via ``chat_model=`` in later
    versions.  We prefer passing the LangChain instance because it reuses the
    user's already-configured endpoint.
    """
    import os

    override = os.getenv("DEEPAGENTS_MODEL", "").strip()
    if override:
        return override
    # Sensible fallback.  Callers that want strict reuse of the Supervisor
    # provider should use ``chat_model=get_supervisor_llm()`` directly.
    return "openai:gpt-4o-mini"


# ---------------------------------------------------------------------------
# Custom tools that expose the offload registry to the subagents
# ---------------------------------------------------------------------------


def make_offload_tools(
    manager: PostRunContextWindowManager,
    selection: ContextSelection,
):
    """Build the two harness tools every subagent gets for progressive disclosure.

    ``read_offloaded_context(key)`` → full body of one offload registry doc.
    ``search_run_context(pattern)`` → grep-like matches across all offloaded docs.
    """

    from langchain_core.tools import tool

    registry_body: Dict[str, str] = dict(selection.offloaded_context_files)
    # Also attach any offloaded turns so the subagent can read old full-text chat.
    for off in selection.offloaded_turns:
        registry_body[off.full_text_ref] = (
            f"# Offloaded chat turns (summary: {off.summary})\n"
            + manager._offload_registry.get(off.full_text_ref, "")
        )

    @tool
    def read_offloaded_context(key: str) -> str:
        """Read the full body of a document from the grounded run-context registry.

        Args:
            key: One of the registry keys listed in the system prompt under
                 "Offloaded context registry" (e.g. "data_prep_details",
                 "manifest_elevated_risk", "connectivity_summary").
        """
        body = registry_body.get(str(key).strip())
        if body is None:
            available = ", ".join(sorted(registry_body.keys())) or "(registry empty)"
            return f"Key '{key}' not found. Available keys: {available}"
        return body

    @tool
    def search_run_context(pattern: str) -> str:
        """Search the offloaded run-context registry for lines matching ``pattern``.

        Case-insensitive substring search.  Args:
            pattern: the string to look for (e.g. "dropped", "Fp1", "alpha").
        """
        import re as _re

        hits: List[str] = []
        needle = str(pattern).lower()
        for key, body in registry_body.items():
            for line_no, line in enumerate(body.splitlines(), start=1):
                if needle in line.lower():
                    snippet = line.strip()
                    if len(snippet) > 200:
                        snippet = snippet[:197] + "…"
                    hits.append(f"- {key}:{line_no}: {snippet}")
            if _re.search(r"(?<!\w)" + _re.escape(needle), key, _re.IGNORECASE):
                hits.append(f"- [key match] '{key}'")
        if not hits:
            return f"No matches for '{pattern}' in the grounded run-context registry."
        return "\n".join(hits[:40])

    return [read_offloaded_context, search_run_context]


# ---------------------------------------------------------------------------
# Plot interpreter tooling (also builds the inline PNG → base64 → langchain
# multimodal message bridge so the subagent can SEE a plot the user asked about).
# ---------------------------------------------------------------------------


def _load_plot_images(ctx: Mapping[str, Any]) -> Dict[str, str]:
    """Return ``{plot_key: base64_data_uri}`` for the stored diagnostic PNGs.

    The keys are exactly those in the run context's ``channel_plot_paths``
    (for example ``channels_before``, ``channels_after``, ``channel_variance``,
    ``psd_overview``). Missing or unreadable files are skipped.
    """
    plots = (ctx.get("channel_plot_paths") or {}) if isinstance(ctx, Mapping) else {}
    result: Dict[str, str] = {}
    for label, raw_path in plots.items():
        if not raw_path:
            continue
        p = Path(str(raw_path))
        if p.suffix.lower() != ".png" or not p.exists():
            continue
        try:
            data = p.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s plot at %s: %s", label, p, exc)
            continue
        b64 = base64.b64encode(data).decode("ascii")
        result[str(label)] = f"data:image/png;base64,{b64}"
    return result


_RECORDING_SUBJECT = re.compile(r"\b(dataset|data set|recording|this file|the file|the data)\b", re.IGNORECASE)
_RECORDING_INQUIRY = re.compile(
    r"\b(what|which|describe|tell me|about|contain|contains|overview|details?|information|info)\b",
    re.IGNORECASE,
)
_RECORDING_EXCLUDE = re.compile(
    r"\b(prep|preprocess|clean|filter|epoch|reference|plot|figure|psd|variance|metric|band)\w*",
    re.IGNORECASE,
)


def _is_recording_question(text: str) -> bool:
    """True for "what is this dataset / recording about" style questions."""
    t = str(text or "")
    return bool(
        _RECORDING_SUBJECT.search(t)
        and _RECORDING_INQUIRY.search(t)
        and not _RECORDING_EXCLUDE.search(t)
    )


def _recording_answer(manager: PostRunContextWindowManager) -> Optional[str]:
    """Answer from the stored recording's header + annotations (no LLM)."""
    from fc_pipeline.toolbox.dataset.recording_facts import (
        format_recording_facts,
        get_recording_facts,
    )

    ctx = manager.run_context
    facts = get_recording_facts(ctx.get("raw_data_path") if isinstance(ctx, Mapping) else None)
    parts = ["**From the recording file** (header and annotations only):", format_recording_facts(facts)]
    try:
        summary = answer_run_question("summary of the run", ctx)
    except Exception:  # pragma: no cover - defensive
        summary = None
    if summary:
        parts += ["", "**What this run did:**", summary]
    parts += [
        "",
        "_The file itself does not say which experiment or study it comes from, and I did not "
        "look up any external dataset documentation._",
    ]
    return "\n".join(parts)


def _content_to_text(content: Any) -> str:
    """Flatten a LangChain message ``content`` (str or list of blocks) to text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out: List[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, Mapping) and block.get("type") == "text":
                out.append(str(block.get("text", "")))
        return "\n".join(t for t in out if t).strip()
    return ""


def _message_type(m: Any) -> Optional[str]:
    if isinstance(m, Mapping):
        return m.get("type") or m.get("role")
    return getattr(m, "type", None) or getattr(m, "role", None)


def _extract_final_text(response: Any) -> str:
    """Last assistant text in a harness response (LangChain ``ai`` messages)."""
    msgs = response.get("messages") if isinstance(response, Mapping) else None
    if not msgs:
        return ""
    fallback = ""
    for m in reversed(list(msgs)):
        if _message_type(m) not in ("ai", "assistant"):
            continue
        content = m.get("content") if isinstance(m, Mapping) else getattr(m, "content", None)
        text = _content_to_text(content)
        if not text:
            continue
        calls = m.get("tool_calls") if isinstance(m, Mapping) else getattr(m, "tool_calls", None)
        if not calls:
            return text
        fallback = fallback or text
    return fallback


def _delegated_name(msgs: Any) -> str:
    """Name of the subagent the coordinator delegated to, from ``task`` tool calls."""
    for m in msgs or []:
        calls = (m.get("tool_calls") if isinstance(m, Mapping) else getattr(m, "tool_calls", None)) or []
        for tc in calls:
            if isinstance(tc, Mapping) and tc.get("name") == "task":
                st = (tc.get("args") or {}).get("subagent_type")
                return str(st) if st else "RunContextQAAgent"
    return "Coordinator"


_PLOT_SYSTEM = textwrap.dedent(
    """
    You describe the stored diagnostic plots of ONE completed EEG pipeline run
    for a researcher.

    RULES:
      1. Describe only what is visible in the images you are given, plus the
         "Run facts" in the message. Never invent a number, channel, band, or
         result that is in neither.
      2. Keep claims visual and cautious ("appears", "visually"). You are not a
         clinical tool: never infer a diagnosis or a medical conclusion.
      3. Say which plot you are talking about (the image order is stated).
      4. If the question cannot be answered from these plots, say so plainly.
      5. Keep it short and in plain language.
    """
).strip()


def _run_plot_interpreter(
    user_message: str,
    manager: PostRunContextWindowManager,
    chat_model: Any,
) -> "FollowUpResult":
    """Answer a plot question by showing the stored PNGs to the vision model.

    The Deep Agents ``task`` tool can only pass text to a subagent, so images
    are sent here with one direct multimodal call.
    """
    ctx = manager.run_context
    plots = _load_plot_images(ctx)
    if not plots:
        return FollowUpResult(
            kind=FollowUpResultKind.ANSWERED,
            assistant_text="No diagnostic plots were generated for this run.",
            delegated_to="PlotInterpreterAgent",
        )
    if chat_model is None:
        return FollowUpResult(
            kind=FollowUpResultKind.ANSWERED,
            assistant_text=_fallback_answer(user_message, "plot_interpreter"),
            delegated_to="DeterministicFallback",
            error="No LLM is configured for follow-ups.",
        )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            facts = answer_run_question("summary of the run", ctx) or ""
        except Exception:  # pragma: no cover - defensive
            facts = ""
        content: List[Any] = [
            {
                "type": "text",
                "text": (
                    f"Question: {user_message}\n\nRun facts (authoritative):\n{facts}\n\n"
                    f"The images that follow are, in order: {', '.join(plots)}."
                ),
            }
        ]
        for uri in plots.values():
            content.append({"type": "image_url", "image_url": {"url": uri}})
        reply = chat_model.invoke([SystemMessage(content=_PLOT_SYSTEM), HumanMessage(content=content)])
        text = _content_to_text(getattr(reply, "content", reply))
        if not text:
            raise ValueError("The plot interpreter returned no text.")
    except Exception as exc:
        logger.warning("Plot interpreter failed: %s: %s", type(exc).__name__, exc)
        return FollowUpResult(
            kind=FollowUpResultKind.ERROR,
            assistant_text=_fallback_answer(user_message, "plot_interpreter"),
            delegated_to="ErrorFallback",
            error=f"{type(exc).__name__}: {exc}",
        )
    return FollowUpResult(
        kind=FollowUpResultKind.NEEDS_PLOT,
        assistant_text=text,
        delegated_to="PlotInterpreterAgent",
        image_paths=list(plots.keys()),
    )


# ---------------------------------------------------------------------------
# Deep Agents orchestrator
# ---------------------------------------------------------------------------


def run_followup_agent(
    user_message: str,
    manager: PostRunContextWindowManager,
    *,
    selection: Optional[ContextSelection] = None,
    chat_model: Any = None,
    allow_harness_import_errors: bool = True,
) -> FollowUpResult:
    """Run a single post-run follow-up through the Deep Agents harness.

    Args:
        user_message: Current user input (newest turn only).
        manager:      Active context window manager — holds run context +
                      chat history.  push_user/push_assistant are called by
                      the UI layer around this function (so the manager can
                      compress the window BEFORE the harness runs).
        selection:    Optional pre-built selection; if omitted it is built
                      from manager.prepare_for_query(user_message).
        chat_model:   Optional explicit LangChain ChatModel override.  Defaults
                      to the provider-agnostic Supervisor LLM.
        allow_harness_import_errors: When ``deepagents`` is not installed or
                      fails to import, fall back to the context manager's
                      deterministic answerer and return kind=ANSWERED with a
                      notice attached instead of crashing.
    """
    if selection is None:
        selection = manager.prepare_for_query(user_message)
    intent = selection.routed_intent

    # -------------------- Precondition: LLM chat model --------------------
    if chat_model is None:
        try:
            from fc_pipeline.agentic.supervisor.llm_provider import get_supervisor_llm

            chat_model = get_supervisor_llm()
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("Supervisor LLM not available for follow-up: %s", exc)

    # -------------------- Intent short-circuits ---------------------------
    # 1) New analysis → the harness CANNOT "compute new connectivity".  The
    #    only correct behaviour is to guide the user to click the explicit
    #    🆕 New query button.  Return this immediately (no LLM call needed) so
    #    the Chainlit handler can render the confirmation prompt.
    if intent == "new_analysis":
        msg = textwrap.dedent(
            """
            It looks like you want to run a **different analysis** — new
            frequency band, different channels, condition, metrics, etc.

            The completed run's context is frozen and cannot be re-analysed
            here (that would silently invalidate the grounded audit trail).
            Press the **🆕 New query** button in the header to start the
            Supervisor again with your new parameters.
            """
        ).strip()
        return FollowUpResult(
            kind=FollowUpResultKind.DISPATCH_NEW_QUERY,
            assistant_text=msg,
            delegated_to="NewAnalysisDispatcher",
        )

    # 2) Deterministic shortcut for plain run-context QA — when the topic
    #    detector already has a scripted answer, we prefer it so the answer
    #    is guaranteed identical to what the ad-hoc UI used to show.  The LLM
    #    still runs for anything not covered by the deterministic topic set.
    if intent == "run_context_qa":
        det_answer = manager.answer_deterministically(user_message)
        if det_answer:
            return FollowUpResult(
                kind=FollowUpResultKind.ANSWERED,
                assistant_text=det_answer,
                delegated_to="RunContextQAAgent",
            )

    # -------------------- Attempt harness call ---------------------------
    # 3) Questions about the dataset / recording itself are answered from the
    #    FILE (header + annotations), never from the language model.
    if intent == "run_context_qa" and _is_recording_question(user_message):
        recording_text = _recording_answer(manager)
        if recording_text:
            return FollowUpResult(
                kind=FollowUpResultKind.ANSWERED,
                assistant_text=recording_text,
                delegated_to="RecordingFacts",
            )

    # 4) Plot questions: the vision model sees the stored plot images itself.
    if intent == "plot_interpreter":
        return _run_plot_interpreter(user_message, manager, chat_model)

    # -------------------- Harness for grounded run-context QA --------------
    if chat_model is None:
        return FollowUpResult(
            kind=FollowUpResultKind.ANSWERED,
            assistant_text=_fallback_answer(user_message, intent),
            delegated_to="DeterministicFallback",
            error="No LLM is configured for follow-ups.",
        )

    try:
        from deepagents import create_deep_agent
    except Exception as exc:
        if not allow_harness_import_errors:
            raise
        logger.warning("deepagents import failed — falling back to deterministic QA: %s", exc)
        fallback = (
            det_answer
            if (det_answer := manager.answer_deterministically(user_message))
            else _fallback_answer(user_message, intent)
        )
        return FollowUpResult(
            kind=FollowUpResultKind.ANSWERED,
            assistant_text=fallback,
            delegated_to="DeterministicFallback",
            error=f"deepagents unavailable: {exc}",
        )

    offload_tools = make_offload_tools(manager, selection)

    # Deep Agents 0.7 subagents are plain dicts (``SubAgent``): ``description``
    # tells the coordinator when to delegate, ``system_prompt`` is the role.
    run_ctx_qa_subagent: Dict[str, Any] = {
        "name": "RunContextQAAgent",
        "description": textwrap.dedent(
            """
            Delegate here when the user asks a factual question about the
            completed run's parameters, filtering, channels, epochs, metrics,
            manifest, or summary. Typical queries: which band, which channels,
            any dropped channels, how many epochs, which condition, what
            reference or filter was applied, a summary of the run.
            """
        ).strip(),
        "system_prompt": textwrap.dedent(
            """
            You are RunContextQAAgent: a grounded answerer for questions about a
            SINGLE COMPLETED EEG FC pipeline run.

            RULES (never break them):
              1. NEVER invent a number, channel label, frequency, or epoch count.
                 If a value is not in the loaded context, use the
                 ``read_offloaded_context`` tool to retrieve it, then say
                 explicitly when it still cannot be found.
              2. Prefer the format:  **<Topic>:** <grounded value>.
              3. For multi-topic questions, build a bulleted list.
              4. Do NOT suggest running new connectivity or re-doing Data Prep
                 inside a follow-up. If the user wants a change, reply
                 "please press 🆕 New query".
              5. Use ``search_run_context`` when a value may be offloaded under
                 a different key.
            """
        ).strip(),
        "tools": offload_tools,
    }

    # -------------------- System prompt for coordinator ------------------
    orchestrator_system = textwrap.dedent(
        """
        You are the Post-run Follow-up Coordinator for an EEG functional
        connectivity pipeline.

        YOUR ONLY JOB is to:
          (a) treat the user's question as a grounded factual question about
              the completed run;
          (b) DELEGATE it to the RunContextQAAgent subagent using the ``task``
              tool;
          (c) return the subagent's answer as your final answer verbatim —
              do NOT paraphrase, do NOT add disclaimers the specialist did
              not write, do NOT invent values.

        {system_fragment}

        COORDINATOR RULES:
          * NEVER answer a factual question directly — ALWAYS delegate.
          * NEVER re-run the pipeline, never compute new connectivity values.
          * If anything is missing, reply with the exact string:
            "I don't have grounded information about that in this completed
            run's context — please check the report or press 🆕 New query
            if you want a different analysis."
        """
    ).strip().format(system_fragment=selection.system_prompt_fragment)

    # ------ (M4 + M6) Load project memory + skills progressive-disclosure index
    memory_bundle_text: str = ""
    memory_file_paths: List[str] = []
    skills_index: str = ""
    matched_skill_bodies: str = ""
    try:
        from fc_pipeline.pipeline.memory import load_full_memory_bundle, find_repo_root
        from fc_pipeline.pipeline.skills import SkillRegistry, get_skill_registry

        bundle = load_full_memory_bundle()
        memory_bundle_text = bundle.text or ""
        repo_root = find_repo_root()
        if repo_root is not None:
            for rel in ("AGENTS.md", ".fc_pipeline/AGENTS.md"):
                cand = repo_root / rel
                if cand.exists():
                    memory_file_paths.append(str(cand))
        registry: SkillRegistry = get_skill_registry()
        skills_index = registry.index_summary()
        matched = registry.matching_for_query(user_message, limit=2)
        if matched:
            matched_skill_bodies = "\n\n".join(
                s.system_prompt_fragment for s in matched if s.body.strip()
            )
    except Exception as exc:  # pragma: no cover - env + path dependent
        logger.warning("follow-up: memory/skill load failed (continuing without): %s", exc)
        memory_bundle_text = ""
        memory_file_paths = []
        skills_index = ""
        matched_skill_bodies = ""

    preamble_parts: List[str] = []
    if memory_bundle_text.strip():
        preamble_parts.append(
            "## Project + user memory (always loaded)\n\n" + memory_bundle_text.strip()
        )
    if skills_index.strip():
        # Always-load FRONTMATTER only (the index) — skill BODIES load only when
        # matched (progressive disclosure). This is layer 1 of the Deep Agents
        # Skills design.
        preamble_parts.append(
            "## Installed skills (progressive disclosure index)\n\n"
            + skills_index.strip()
            + "\n\n"
            + "Rule: ONLY pull a skill body into context when the user request "
            "matches a trigger above.  Do NOT describe or invoke a skill whose "
            "triggers are unrelated."
        )
    if matched_skill_bodies.strip():
        preamble_parts.append(
            "## Skills — bodies loaded for this specific query (progressive-disclosure hits)\n\n"
            + matched_skill_bodies.strip()
        )
    if preamble_parts:
        orchestrator_system = (
            "\n\n---\n\n".join(preamble_parts) + "\n\n---\n\n" + orchestrator_system
        )

    # -------------------- Build + invoke the harness ----------------------
    try:
        agent = create_deep_agent(
            model=chat_model,
            tools=offload_tools,
            system_prompt=orchestrator_system,
            subagents=[run_ctx_qa_subagent],
            name="fc_followup_agent",
        )

        # selection.active_messages already includes every turn pushed into the
        # manager (including the latest user turn).
        msg_payload: List[Dict[str, Any]] = list(selection.active_messages)
        if not msg_payload or msg_payload[-1].get("role") != "user" or (
            isinstance(msg_payload[-1].get("content"), str)
            and msg_payload[-1]["content"].strip() != str(user_message).strip()
        ):
            msg_payload.append({"role": "user", "content": str(user_message)})

        response = agent.invoke({"messages": msg_payload})

        answer_text = _extract_final_text(response)
        if not answer_text:
            raise ValueError("The follow-up harness returned no answer text.")
        raw_msgs = response.get("messages") if isinstance(response, Mapping) else None
        return FollowUpResult(
            kind=FollowUpResultKind.ANSWERED,
            assistant_text=answer_text,
            delegated_to=_delegated_name(raw_msgs),
        )

    except Exception as exc:
        logger.exception("Deep Agents follow-up harness failed: %s", exc)
        fallback = (
            det_answer
            if "det_answer" in locals()
            and (det_answer := manager.answer_deterministically(user_message))
            else _fallback_answer(user_message, intent)
        )
        return FollowUpResult(
            kind=FollowUpResultKind.ERROR,
            assistant_text=fallback,
            delegated_to="ErrorFallback",
            error=f"{type(exc).__name__}: {exc}",
        )


def _fallback_answer(user_message: str, intent: str) -> str:
    if intent == "plot_interpreter":
        return (
            "I couldn't interpret the plots right now. The diagnostic plots from "
            "this run are shown in the chat above, and nothing was re-run."
        )
    return (
        "I don't have grounded information about that in this completed run's "
        "context — please check the report, or press 🆕 New query if you want "
        "to run a different analysis."
    )
