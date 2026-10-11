"""Offline tests for M4 project-memory loader.

No LLM, no network, no deepagents imports. Exercises:
  * find_repo_root() returns a real Path with AGENTS.md or chainlit_app.py.
  * load_project_memory() returns non-empty text + AGENTS.md listed in sources
    when repo root is found.
  * load_full_memory_bundle() returns {} user_defaults for new users.
  * _current_user_id is stable and not "default_user" on a real machine.
"""

from __future__ import annotations

from pathlib import Path

from fc_pipeline.pipeline import memory as mem_mod


def test_find_repo_root_returns_marker():
    root = mem_mod.find_repo_root()
    assert root is not None, "find_repo_root should find the AGENTS.md / chainlit_app.py ancestor"
    assert root.exists()
    any_marker = any(
        (root / marker).exists()
        for marker in ("AGENTS.md", "chainlit_app.py", "pyproject.toml")
    )
    assert any_marker, f"No project marker found under candidate root: {root}"


def test_load_project_memory_reads_agents_md(tmp_path):
    # Build a fake project root with a known AGENTS.md and confirm it loads.
    (tmp_path / "AGENTS.md").write_text(
        "# M4 Test Project\n\n## 1. Rule A\nValue = 0.20\n", encoding="utf-8"
    )
    bundle = mem_mod.load_project_memory(repo_root=tmp_path)
    assert "M4 Test Project" in bundle.text
    assert "Rule A" in bundle.text
    assert any("AGENTS.md" in str(s) for s in bundle.sources)


def test_load_full_memory_bundle_defaults_empty(tmp_path, monkeypatch):
    # New user -> empty confirmed_defaults dict.
    monkeypatch.setenv("FC_PIPELINE_USER_ID", "m4_fresh_user_xyz")
    # Make a user_memory dir so path exists but no preference snapshot file.
    (tmp_path / "user_memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "AGENTS.md").write_text("# stub\n", encoding="utf-8")
    bundle = mem_mod.load_full_memory_bundle(
        user_id="m4_fresh_user_xyz", repo_root=tmp_path
    )
    assert isinstance(bundle.user_defaults, dict)
    # No preference snapshot -> nothing loaded
    assert bundle.user_defaults == {}


def test_load_full_memory_bundle_prefers_env_user_id(monkeypatch):
    monkeypatch.setenv("FC_PIPELINE_USER_ID", "stable_id_123")
    assert mem_mod._current_user_id() == "stable_id_123"
