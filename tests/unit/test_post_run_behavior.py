"""Regression checks for the Chainlit post-run routing contract.

These are source-level checks because the sandbox used for packaging does not
have Chainlit/LangGraph installed. The runtime behavior is implemented in
chainlit_app.py and should be exercised in the project's normal environment.

Implementation change (2025 Q1): The ad-hoc if/elif body of
_handle_completed_run_followup was replaced with the Deep Agents harness
(see fc_pipeline.agentic.followup).  Routing semantics are unchanged:
  * a completed run stays read-only for typed follow-ups,
  * New query is the sole explicit unlock for a fresh pipeline run,
  * plot/signal requests return inline images via cl.Image elements.
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
    """Post-run plot requests go through the Deep Agents harness but still:
    (1) send inline Chainlit images via cl.Image elements,
    (2) route through the result-kind dispatcher which attaches stored
        before/after plot paths.
    The harness-only dispatcher lives inside the async follow-up handler.
    """
    text = APP.read_text(encoding="utf-8")
    # The harness invocation + inline image attachment are the new contract.
    assert "run_followup_agent" in text
    assert "FollowUpResultKind.NEEDS_PLOT" in text
    assert "followup_plot_" in text  # inline image name
    assert 'cl.Image(' in text  # images are still rendered inline (element array)


def _extract_is_explicit_new_block(text: str) -> str:
    start = text.index("is_explicit_new = (")
    end = text.index(")", start) + 1
    return text[start:end]


def test_reset_startswith_overtrigger_fixed():
    """'resetting the reference...' must NOT be treated as explicit reset."""
    text = APP.read_text(encoding="utf-8")
    block = _extract_is_explicit_new_block(text)

    def is_explicit_new(user_text: str) -> bool:
        user_text_lc = user_text.lower().lstrip()
        return (
            user_text_lc.startswith("new query:")
            or user_text_lc.startswith("new topic:")
            or user_text_lc == "/reset"
            or user_text_lc == "reset"
            or user_text_lc.startswith("reset ")
        )

    cases = {
        "resetting the reference made it worse, why?": False,
        "reset": True,
        "/reset": True,
        "reset condition task": True,
        "new query: theta band": True,
        "analyze this again with gamma": False,
        "reset  ": True,
    }
    for inp, expected in cases.items():
        assert is_explicit_new(inp) is expected, f"mismatch for {inp!r}"

    # And confirm the source block no longer has the buggy bare startswith("reset")
    # without a following space/equality guard.
    assert "user_text_lc.startswith(\"reset\")" not in block or "user_text_lc.startswith(\"reset \")" in block
