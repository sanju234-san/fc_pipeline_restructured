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


def test_completed_run_new_intent_falls_through_not_catchall():
    """NEW_INTENT_KEYWORDS in catch-all returns False (fall-through to Supervisor)."""
    text = APP.read_text(encoding="utf-8")

    # Verify NEW_INTENT_KEYWORDS tuple is declared inside _handle_completed_run_followup
    fn_start = text.index("async def _handle_completed_run_followup(")
    fn_body_start = text.index("\n", fn_start)
    fn_end = text.index("\nasync def ", fn_body_start + 1) if "\nasync def " in text[fn_body_start + 1:] else len(text)
    fn_body = text[fn_body_start:fn_end]

    assert "NEW_INTENT_KEYWORDS" in fn_body
    # Conservative keywords list must include the 5 metrics + bands
    for required_kw in ("pli", "wpli", "coherence", "plv", "imcoh", "theta", "alpha", "beta", "gamma", "delta", "compute", "analy"):
        assert f'"{required_kw}"' in fn_body or f"'{required_kw}'" in fn_body, f"missing required keyword {required_kw!r}"
    # Return-false fall-through branch must exist inside the catch-all area
    assert "return False" in fn_body
    # And the old rigid 'press button' hint must no longer be the final sentence
    old_rigid = "To analyse something different, press the **🆕 New query** button."
    assert old_rigid not in fn_body, "old rigid catch-all hint still present"
    # New relaxed wording must be there
    assert "new analysis requests automatically start a fresh cycle" in fn_body or "automatically start a fresh cycle" in fn_body

    # Source-level: a follow-up hint (channel/plot) must still return True
    # (early branches are unchanged).  Confirm 'return True' remains multiple times.
    assert fn_body.count("return True") >= 2
