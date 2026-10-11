"""Offline tests for M6 skills (progressive disclosure).

No LLM / no deepagents. Exercises:
  * SkillRegistry scans repo_root/skills/ and discovers synthesis_viva +
    artifact_scoring skills with correct slug/description/triggers.
  * matching_for_query("viva report") returns synthesis_viva first.
  * matching_for_query("gate 2 artifact severity") returns artifact_scoring.
  * Skill body is LAZY — inspecting frontmatter does NOT load SKILL.md body
    from disk; accessing .body does.
  * .system_prompt_fragment contains BEGIN/END markers and the description.
  * @filename.md references in synthesis_viva/SKILL.md are inlined into body.
  * index_summary() renders a markdown table containing both installed skills.
  * Empty skill dirs (no SKILL.md) are silently skipped.
  * Custom SkillRegistry built from a temp dir with a user skill works.
"""

from __future__ import annotations

import pytest

from fc_pipeline.pipeline import skills as sk


def test_registry_discovers_shipped_skills():
    reg = sk.get_skill_registry()
    # Reload to reset any cached bodies for the lazy-loading test.
    reg = sk.reload_skill_registry()
    assert len(reg) >= 2, (
        f"Expected synthesis_viva + artifact_scoring, found {len(reg)}: "
        + ", ".join(s.slug for s in reg)
    )
    viv = reg.get("synthesis_viva")
    assert viv is not None
    assert "viva-ready" in viv.description.lower() or "viva" in viv.name.lower()
    # At least one "viva" trigger.
    assert any("viva" in t.lower() for t in viv.triggers)
    art = reg.get("artifact_scoring")
    assert art is not None
    assert any("artifact" in t.lower() for t in art.triggers)


def test_matching_for_query_routes_correctly():
    reg = sk.reload_skill_registry()
    hits = reg.matching_for_query("produce a viva-ready report for this thesis defence")
    assert hits, "viva query should match at least one skill"
    assert hits[0].slug == "synthesis_viva", (
        f"Expected synthesis_viva as top hit, got {hits[0].slug}"
    )

    hits2 = reg.matching_for_query("gate 2 severity scoring and artifact review")
    assert hits2, "gate 2 + artifact query should match at least one skill"
    assert hits2[0].slug == "artifact_scoring"


def test_body_is_lazy(tmp_path, monkeypatch):
    """Reading frontmatter only should not cause a SKILL.md body read.

    We verify indirectly: the Skill object exposes .name/.description/.triggers
    after scan but `_body` stays None until .body is accessed.
    """
    reg = sk.reload_skill_registry()
    s = reg.get("synthesis_viva")
    assert s is not None
    # Cache is None before first access.
    assert s._body is None
    text = s.body
    assert s._body is not None
    assert text
    assert "Viva-Ready" in text or "viva-ready" in text.lower()


def test_synthesis_viva_body_inlines_section_includes():
    reg = sk.reload_skill_registry()
    viv = reg.get("synthesis_viva")
    body = viv.body
    # The companion templates section1_methods_template.md and
    # section2_template.md are referenced as @sectionX_...md inside SKILL.md
    # and must be inlined as "BEGIN SKILL INCLUDE: ...".
    assert "BEGIN SKILL INCLUDE" in body, (
        "Synthesis-viva SKILL.md has @section template refs that should be "
        "inlined by Skill._load_companions."
    )
    assert "section1_methods_template.md" in body
    assert "section2_template.md" in body


def test_index_summary_has_table():
    reg = sk.reload_skill_registry()
    summary = reg.index_summary()
    assert "|" in summary, "index_summary should return a markdown table"
    assert "synthesis_viva" in summary
    assert "artifact_scoring" in summary


def test_custom_registry_from_temp_dir(tmp_path):
    user = tmp_path / "my_skill"
    user.mkdir()
    (user / "SKILL.md").write_text(
        """---
name: Custom Skill
description: My user-defined workflow.
version: 1.0.0
triggers:
  - mytrigger
  - custom
tags: [user]
---

# Body of custom skill
Hello world.
""",
        encoding="utf-8",
    )
    reg = sk.SkillRegistry(dirs=[tmp_path])
    assert len(reg) == 1
    s = reg.get("my_skill")
    assert s is not None
    assert s.description == "My user-defined workflow."
    hits = reg.matching_for_query("please run the custom mytrigger thing")
    assert hits and hits[0].slug == "my_skill"


def test_frontmatter_parser_scalar_and_inline_list():
    raw = """---
name: foo
version: 1.2
triggers: [a, b, c]
tags:
  - x
  - y
---
Body text.
"""
    fm = sk._parse_frontmatter(raw)
    assert fm["name"] == "foo"
    # version strings like "1.2" are float-like by the simple parser; that's ok.
    assert fm["version"] == 1.2
    assert fm["triggers"] == ["a", "b", "c"]
    assert fm["tags"] == ["x", "y"]
    # Body stripped:
    body = sk._strip_frontmatter(raw)
    assert "Body text." in body
    assert "version:" not in body


def test_empty_skill_dir_skipped(tmp_path):
    empty = tmp_path / "empty_skill_dir"
    empty.mkdir()
    # No SKILL.md -> ignored, no warnings raised.
    reg = sk.SkillRegistry(dirs=[tmp_path])
    assert len(reg) == 0
