"""Skills (progressive-disclosure domain workflows) — Deep Agents "Skills" layer.

Architecture (mirrors docs.langchain.com/oss/python/deepagents/overview
  "Skills" section + SKILL.md agent skills spec):

  * Each skill lives under ``repo_root / skills / <slug> /``.
  * Mandatory file: ``SKILL.md`` with YAML frontmatter between ``---`` lines:

        ---
        name: synthesis-viva
        description: Write a viva-ready FC analysis report (5 sections).
        version: 1.0.0
        author: fc-pipeline-team
        triggers:
          - viva
          - defence
          - viva-ready
          - oral exam
          - university report
        tags: [synthesis, writing, evaluation]
        inputs:
          - plan
          - data_prep_summary
          - connectivity_summary
          - evaluation_verdict
        outputs:
          - report.md
        requires_model: false
        ---

        # Body of the skill
        Multi-step instructions for the LLM here. Companion files (templates.md,
        heuristics_ref.md) in the same folder are referenced with @filename.md
        syntax — they are loaded lazily when the skill body is loaded.

  * Loading contract:
      - **At startup**: Only the frontmatter of every ``SKILL.md`` is scanned.
        Bodies and companion files are NEVER read until a trigger matches.
      - **On demand**: :meth:`SkillRegistry.matching_for_query` returns a list of
        :class:`Skill` objects whose ``triggers`` regex matched the user query
        or task description.  Each returned skill lazily reads its full body on
        the first ``.body`` access, plus any ``@filename`` references from the
        same skill folder.

  * Consumers:
      - Synthesis node (future): before invoking the synthesis LLM, ask
        ``skills.matching_for_query(user_request + " synthesis report")`` and
        append skill bodies to the system prompt fragment.
      - FollowUp coordinator (M6 wiring): added to
        :func:`fc_pipeline.agentic.followup.agents.run_followup_agent` so the
        subagents can pull in a skill when the user asks for a viva report or
        artifact-scoring review.

Offline-testable: no LLM, no network, pure file IO + regex. Zero deepagents
imports (callers wire the loaded bodies into harness prompts).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fc_pipeline.pipeline.memory import find_repo_root

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """One loaded skill.  Frontmatter is always populated; ``body`` is lazy."""

    slug: str
    folder: Path
    frontmatter: Dict[str, Any]
    triggers: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # --- lazy body --------------------------------------------------------
    _body: Optional[str] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return str(self.frontmatter.get("name") or self.slug)

    @property
    def description(self) -> str:
        return str(self.frontmatter.get("description") or "")

    def _load_companions(self, body: str) -> str:
        """Expand @relative/path references from the skill folder into their
        markdown body, fenced with BEGIN/END markers.

        Mirrors Claude Code AGENTS.md import syntax but applies ONLY within an
        already-loaded skill body (so we don't pull random files).
        """
        pat = re.compile(r"(?<!`)@([A-Za-z0-9_\-./][A-Za-z0-9_\-./ ]*\.md)(?!`)")

        def _sub(match: re.Match) -> str:
            rel = match.group(1).strip()
            candidate = (self.folder / rel).resolve()
            try:
                if self.folder.resolve() not in candidate.parents and candidate != self.folder.resolve():
                    return match.group(0)
                if not candidate.exists() or not candidate.is_file():
                    return match.group(0)
                included = candidate.read_text(encoding="utf-8")
                return (
                    f"\n\n<!-- BEGIN SKILL INCLUDE: {candidate.name} -->\n"
                    + included.rstrip()
                    + f"\n<!-- END SKILL INCLUDE: {candidate.name} -->\n\n"
                )
            except OSError as exc:
                logger.warning("skills: could not include %s: %s", candidate, exc)
                return match.group(0)

        return pat.sub(_sub, body)

    @property
    def body(self) -> str:
        if self._body is None:
            p = self.folder / "SKILL.md"
            try:
                raw = p.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("skills: could not read %s: %s", p, exc)
                self._body = ""
                return self._body
            body_only = _strip_frontmatter(raw)
            self._body = self._load_companions(body_only)
        return self._body

    @property
    def system_prompt_fragment(self) -> str:
        """The skill rendered as a system-prompt fragment for an LLM agent."""
        head = (
            f"<!-- BEGIN SKILL: {self.name} (slug: {self.slug}) -->\n\n"
            f"## Skill: {self.name}\n\n"
        )
        if self.description:
            head += f"_{self.description}_\n\n"
        body = self.body.strip()
        if not body:
            head += "_(no body)_"
        else:
            head += body
        head += f"\n\n<!-- END SKILL: {self.name} -->\n"
        return head


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _strip_frontmatter(raw: str) -> str:
    """Return only the body portion of a SKILL.md file, stripping the YAML
    frontmatter block (if present).  First call should always use the full
    file content; second call on already-stripped text is a no-op."""
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return raw
    return raw[m.end():]


def _parse_frontmatter(raw: str) -> Dict[str, Any]:
    """Minimal YAML frontmatter parser for the small subset we rely on.

    We don't add a PyYAML dependency just to read 6 keys from SKILL.md.
    Supports:
      * ``key: scalar value``
      * ``key:`` followed by indented ``  - item`` bulleted list
    Unknown nested structures produce a warning and are skipped.
    """
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}
    block = m.group(1)
    out: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_list: Optional[List[str]] = None
    for line in block.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        stripped = line.rstrip()
        if stripped.startswith(" ") or stripped.startswith("\t"):
            content = stripped.lstrip()
            if content.startswith("- ") and current_key is not None and current_list is not None:
                item = content[2:].strip().strip("'\"")
                if item:
                    current_list.append(item)
            continue
        if ":" in stripped:
            # Flush previous list entry.
            if current_list is not None:
                out[str(current_key)] = current_list
                current_list = None
                current_key = None
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip("'\"")
            if not val:
                current_key = key
                current_list = []
                # List may be on next lines (handled above), OR inline []:
                continue
            # Inline list: [a, b, "c d"].
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                items = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                out[key] = items
            else:
                # Numeric coercion so `version: 1.0.0` stays string but simple
                # numbers become int/float for downstream callers.
                try:
                    if re.fullmatch(r"\d+", val):
                        parsed: Any = int(val)
                    elif re.fullmatch(r"\d+\.\d+", val):
                        parsed = float(val)
                    else:
                        parsed = val
                except Exception:
                    parsed = val
                out[key] = parsed
    if current_list is not None and current_key is not None:
        out[current_key] = current_list
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DEFAULT_SKILL_DIR_NAME = "skills"


def _default_skill_dirs() -> List[Path]:
    out: List[Path] = []
    root = find_repo_root()
    if root is not None:
        for rel in (_DEFAULT_SKILL_DIR_NAME, ".fc_pipeline/skills", "src/fc_pipeline/skills"):
            cand = root / rel
            if cand.exists() and cand.is_dir():
                out.append(cand)
        if not out and (root / _DEFAULT_SKILL_DIR_NAME).parent == root:
            # Accept a non-existent repo/skills dir so callers can create it.
            out.append(root / _DEFAULT_SKILL_DIR_NAME)
    return out


class SkillRegistry:
    """Scan one or more skill folders; match user queries against triggers."""

    def __init__(self, dirs: Optional[Iterable[Path]] = None):
        self.dirs: List[Path] = [Path(d) for d in (dirs or _default_skill_dirs())]
        self._skills: Dict[str, Skill] = {}
        self._scan()

    # ------------------------------------------------------------------
    def _scan(self) -> None:
        seen: Dict[str, Skill] = {}
        for d in self.dirs:
            try:
                if not d.exists():
                    continue
                for entry in sorted(d.iterdir()):
                    if not entry.is_dir():
                        continue
                    skill_md = entry / "SKILL.md"
                    if not skill_md.exists():
                        continue
                    try:
                        raw = skill_md.read_text(encoding="utf-8")
                    except OSError as exc:
                        logger.warning("skills: skip %s: %s", skill_md, exc)
                        continue
                    fm = _parse_frontmatter(raw)
                    triggers_raw = fm.get("triggers") or fm.get("keywords") or []
                    if isinstance(triggers_raw, str):
                        triggers_raw = [triggers_raw]
                    triggers = [str(t).strip() for t in triggers_raw if str(t).strip()]
                    tags_raw = fm.get("tags") or []
                    if isinstance(tags_raw, str):
                        tags_raw = [tags_raw]
                    tags = [str(t).strip() for t in tags_raw if str(t).strip()]
                    slug = entry.name
                    if slug in seen:
                        logger.warning("skills: duplicate slug '%s' — first wins", slug)
                        continue
                    seen[slug] = Skill(
                        slug=slug,
                        folder=entry,
                        frontmatter=fm,
                        triggers=triggers,
                        tags=tags,
                    )
            except OSError as exc:
                logger.warning("skills: could not scan %s: %s", d, exc)
                continue
        self._skills = seen

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self):
        return iter(self._skills.values())

    def get(self, slug: str) -> Optional[Skill]:
        return self._skills.get(slug)

    def matching_for_query(self, query: str, limit: int = 3) -> List[Skill]:
        """Return skills whose triggers regex-matched anywhere in the query.

        Matching order: exact-trigger-first -> longest trigger first so a more
        specific phrase beats a single keyword.
        """
        q = str(query or "")
        if not q:
            return []
        scored: List[tuple[int, int, Skill]] = []
        for skill in self._skills.values():
            matched_len = 0
            matches = 0
            for trigger in skill.triggers:
                try:
                    if re.search(rf"(?<!\w){re.escape(trigger)}(?!\w)", q, flags=re.IGNORECASE) or trigger.lower() in q.lower():
                        matched_len = max(matched_len, len(trigger))
                        matches += 1
                except re.error:
                    if trigger.lower() in q.lower():
                        matched_len = max(matched_len, len(trigger))
                        matches += 1
            if matches > 0:
                scored.append((matches, matched_len, skill))
        scored.sort(key=lambda s: (-s[0], -s[1], s[2].slug))
        return [s[2] for s in scored[: max(1, int(limit))]]

    def index_summary(self) -> str:
        """Human-readable table of discovered skills (names, descriptions,
        triggers).  Used as a ALWAYS-loaded system-prompt fragment so the LLM
        can decide which skills to request via `Skill.get(slug).body` call."""
        if not self._skills:
            return "_No skills installed._"
        lines = ["## Installed skills (progressive disclosure)", ""]
        lines.append("Load a skill body ONLY when its trigger phrase matches the user request.")
        lines.append("")
        lines.append("| Skill (slug) | Description | Triggers |")
        lines.append("|:---|:---|:---|")
        _pipe_esc = "|"
        for skill in sorted(self._skills.values(), key=lambda s: s.slug):
            triggers = ", ".join(f"`{t}`" for t in skill.triggers[:5]) or "_none_"
            if len(skill.triggers) > 5:
                triggers += f", … (+{len(skill.triggers)-5})"
            desc_safe = skill.description.replace(_pipe_esc, "\\|")
            lines.append(f"| **{skill.name}** (`{skill.slug}`) | {desc_safe} | {triggers} |")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton (avoid repeating disk scans per follow-up turn).
# ---------------------------------------------------------------------------

_GLOBAL_REGISTRY: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Return (lazily creating) the process-wide :class:`SkillRegistry`."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = SkillRegistry()
    return _GLOBAL_REGISTRY


def reload_skill_registry() -> SkillRegistry:
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = SkillRegistry()
    return _GLOBAL_REGISTRY
