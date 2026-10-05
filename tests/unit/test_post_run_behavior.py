"""Regression checks for the Chainlit post-run routing contract.

These are source-level checks because the sandbox used for packaging does not
have Chainlit/LangGraph installed. The runtime behavior is implemented in
chainlit_app.py and should be exercised in the project's normal environment.
"""
from pathlib import Path


APP = Path(__file__).resolve().parents[2] / "chainlit_app.py"


def test_completed_run_intercepts_plain_followups_before_pipeline():
    text = APP.read_text(encoding="utf-8")
    marker = "if cl.user_session.get(\"completed_run_active\", False) and not is_explicit_new:"
    assert marker in text
    intercept = text.index(marker)
    pipeline = text.index("# --- Invoke Pipeline ---")
    assert intercept < pipeline


def test_new_query_is_the_explicit_unlock_for_a_fresh_pipeline():
    text = APP.read_text(encoding="utf-8")
    assert 'label="🆕 New query"' in text
    assert 'cl.user_session.set("completed_run_active", False)' in text


def test_post_run_before_after_plot_path_uses_inline_images():
    text = APP.read_text(encoding="utf-8")
    assert "Before Data Prep — Raw EEG overview" in text
    assert "After Data Prep — Diagnostic plots" in text
    assert 'cl.Image(path=str(before_path)' in text
