"""Project + user memory loader.

Implements Deep Agents "Memory" layer: static instruction files loaded at
agent startup (always in context, not progressive-disclosure). The three
concepts here mirror the Deep Agents + Claude Code taxonomies:

  * **M4 – Project memory** (``AGENTS.md`` at project root) — committed to
    repo, team-shared, loaded for every LLM node regardless of user.

  * **M5 – User preference memory** (JSON under ``user_memory/``) — per-user,
    persisted to disk, loaded on top of project memory so user defaults
    overwrite / complement project-wide rules.  Content here is produced by
    the auto-memory heuristic in :mod:`fc_pipeline.pipeline.user_memory` or
    by an explicit "remember this" command.

  * **M4b – User instructions** (``~/.fc_pipeline_instructions.md``) —
    optional personal file outside the repo (like ``~/.claude/CLAUDE.md``).
    Loaded last so it beats project rules on conflicts.

All loaders in this module are **OFFLINE-TESTABLE**: they only touch the
filesystem and return plain strings / small dicts.  No LLM, no networking.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution order matches Claude Code (broad → narrow) but is simpler:
#   1. repo AGENTS.md  (broad, committed, team)
#   2. repo .fc_pipeline/AGENTS.md  (alternative location)
#   3. repo user_memory/{user_id}_instructions.md  (per-user, optional, committed rarely)
#   4. ~/.fc_pipeline_instructions.md  (per-user, personal, outside repo)
# ---------------------------------------------------------------------------

_REPO_ROOT_HINTS = (
    Path(__file__).resolve().parent.parent.parent.parent,   # fc_pipeline/ (repo clone root)
    Path(__file__).resolve().parent.parent.parent,          # src/ parent (when installed editable)
)


def _candidate_repo_roots() -> List[Path]:
    seen = set()
    out: List[Path] = []
    for hint in _REPO_ROOT_HINTS:
        p = hint
        if str(p) in seen:
            continue
        seen.add(str(p))
        out.append(p)
        # Walk up to find a folder that contains AGENTS.md / chainlit_app.py.
        for parent in p.parents:
            if str(parent) in seen:
                continue
            seen.add(str(parent))
            out.append(parent)
    return out


def find_repo_root() -> Optional[Path]:
    """Return the first ancestor that holds either ``AGENTS.md`` or
    ``chainlit_app.py`` (two unambiguous markers for the project root)."""
    for root in _candidate_repo_roots():
        try:
            if not root.exists():
                continue
        except OSError:
            continue
        markers = ("AGENTS.md", "chainlit_app.py", "pyproject.toml")
        if any((root / m).exists() for m in markers):
            return root
    return None


@dataclass
class MemoryBundle:
    """Concatenated memory ready to be pasted into a system prompt.

    Attributes:
        text:            Raw Markdown string (concatenated with `---` dividers).
        sources:         Files actually read, in load order (for debugging / UI).
        user_defaults:   Key/value snapshot from M5 preference memory
                         (``{axis: value}``) — separate so callers like the
                         Supervisor can **pre-populate GraphState defaults**
                         instead of relying on the LLM to notice the prompt.
    """

    text: str = ""
    sources: List[str] = field(default_factory=list)
    user_defaults: Dict[str, object] = field(default_factory=dict)


def _read_if_exists(path: Path) -> Optional[str]:
    try:
        if not path.exists() or not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("memory: could not read %s: %s", path, exc)
        return None


def load_project_memory(repo_root: Optional[Path] = None) -> MemoryBundle:
    """M4 + M4b: load project + optional personal static memory files.

    Returns a :class:`MemoryBundle` ready for:
      * ``memory=`` kwarg in Deep Agents ``create_deep_agent``.
      * concatenation into Supervisor / Synthesis / Evaluator system prompts.
    """
    if repo_root is None:
        repo_root = find_repo_root()

    parts: List[str] = []
    sources: List[str] = []

    def _add(label: str, body: str, source_path: Path) -> None:
        parts.append(f"<!-- BEGIN MEMORY: {label} (source: {source_path.name}) -->\n\n{body.rstrip()}\n\n<!-- END MEMORY: {label} -->")
        sources.append(str(source_path))

    if repo_root is not None:
        for rel in ("AGENTS.md", ".fc_pipeline/AGENTS.md"):
            path = repo_root / rel
            body = _read_if_exists(path)
            if body:
                _add("project", body, path)

        # Per-user committed instructions (rare) — {user_id}_instructions.md.
        user_id = _current_user_id()
        if user_id:
            user_md = repo_root / "user_memory" / f"{user_id}_instructions.md"
            body = _read_if_exists(user_md)
            if body:
                _add("per-user instructions", body, user_md)

    # Personal file in home (like ~/.claude/CLAUDE.md).
    home = Path(os.path.expanduser("~"))
    personal = home / ".fc_pipeline_instructions.md"
    body = _read_if_exists(personal)
    if body:
        _add("personal home instructions", body, personal)

    return MemoryBundle(text="\n\n".join(parts), sources=sources)


def _current_user_id() -> str:
    """Stable identifier for M5 preference memory storage.

    Priority: env ``FC_PIPELINE_USER_ID`` → login name → ``"default_user"``.
    Callers (Chainlit UI, tests) can override by setting
    ``FC_PIPELINE_USER_ID`` before a session starts.
    """
    override = (os.getenv("FC_PIPELINE_USER_ID") or "").strip()
    if override:
        return override
    login = (os.getenv("USERNAME") or os.getenv("USER") or "").strip()
    if login:
        return login.lower().replace(" ", "_")
    return "default_user"


def load_full_memory_bundle(
    user_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> MemoryBundle:
    """M4 + M4b + M5 snapshot (user preference defaults dict).

    This is the **single entry point** callers should use:
      * Supervisor calls this at session start → uses ``user_defaults`` to
        pre-populate resolved axes in GraphState (skips HITL on that axis).
      * Follow-up Deep Agents harness calls this → appends ``text`` to the
        coordinator system prompt and passes ``memory=`` when supported.
      * Chainlit UI startup → caches in ``cl.user_session`` to avoid re-reads.
    """
    bundle = load_project_memory(repo_root=repo_root)
    user_id = user_id or _current_user_id()

    # M5 preference snapshot (JSON) lives next to the JSONL history.
    root = repo_root or find_repo_root()
    if root is not None:
        snap_path = root / "user_memory" / f"{user_id}_preferences.json"
        snap = _read_if_exists(snap_path)
        if snap:
            import json as _json
            try:
                parsed = _json.loads(snap)
                if isinstance(parsed, dict):
                    bundle.user_defaults = dict(parsed.get("confirmed_defaults") or {})
                    if str(snap_path) not in bundle.sources:
                        bundle.sources.append(str(snap_path))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("memory: could not parse %s: %s", snap_path, exc)

    return bundle
