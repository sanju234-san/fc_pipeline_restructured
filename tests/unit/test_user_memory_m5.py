"""Offline tests for M5 user-preference (auto-memory) module.

No LLM, no deepagents. Exercises:
  * load_preference_store() returns empty defaults for a fresh user.
  * record_pick() tracks history; does NOT propose until count >= 3.
  * At count==3, record_pick returns a PreferenceProposal.
  * confirm_default() persists into {user}_preferences.json and is then
    returned by load_preference_store() on the next call.
  * confirm_default() + apply_confirmed_defaults_to_state() produce the
    expected GraphState updates.
  * remember_explicit() recognises plain-text commands like "remember that
    I prefer alpha band".
  * reject_proposal(forever=True) prevents proposals for the same value.
  * clear_default() correctly removes a stored default.
  * /remember ... command paths via remember_explicit produce a confirmation
    string for "average reference".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fc_pipeline.pipeline import user_memory as um


# ---- fixtures ------------------------------------------------------------

@pytest.fixture
def fresh_user(tmp_path, monkeypatch):
    """Route user_memory into a tmp_path + fixed user_id so tests are isolated.

    Uses the module's built-in ``FC_PIPELINE_USER_MEMORY_DIR`` override rather
    than monkey-patching a function, so both ``_pref_path`` and any inlined
    use of ``_user_memory_dir()`` resolve to the same temp directory.
    """
    user_id = "m5_automem_test_user"
    monkeypatch.setenv("FC_PIPELINE_USER_ID", user_id)
    target_dir = tmp_path / "user_memory"
    target_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FC_PIPELINE_USER_MEMORY_DIR", str(tmp_path))
    # Force clear any cached registry/store state that might outlive the env var
    # changes between isolated test runs.
    return user_id, target_dir


def _three_picks(user_id: str, axis: str, value):
    for i in range(3):
        prop = um.record_pick(
            user_id, axis, value, run_id=f"run_{i}", threshold=3
        )
    return prop


# ---- tests ---------------------------------------------------------------

def test_fresh_store_empty(fresh_user):
    user_id, _ = fresh_user
    store = um.load_preference_store(user_id)
    assert store.confirmed_defaults == {}
    assert store.rejected_proposals == {}


def test_record_pick_below_threshold(fresh_user):
    user_id, _ = fresh_user
    prop = um.record_pick(user_id, "frequency_band", "alpha", "run_0", threshold=5)
    # 1 pick -> None
    assert prop is None


def test_record_pick_at_threshold_proposes(fresh_user):
    user_id, _ = fresh_user
    prop = _three_picks(user_id, "frequency_band", "alpha")
    assert prop is not None
    assert prop.axis == "frequency_band"
    assert prop.count >= 3
    assert "frequency band" in prop.axis_label
    assert prop.raw_value_repr  # non-empty


def test_confirm_default_roundtrip(fresh_user):
    user_id, _ = fresh_user
    defaults = um.confirm_default(
        user_id,
        "frequency_band",
        {"name": "alpha", "fmin": 8.0, "fmax": 12.0},
    )
    assert defaults["frequency_band"]["name"] == "alpha"
    store = um.load_preference_store(user_id)
    assert store.confirmed_defaults["frequency_band"]["fmin"] == 8.0


def test_confirmed_default_skips_proposal(fresh_user):
    user_id, _ = fresh_user
    um.confirm_default(
        user_id,
        "channels",
        ["Fp1", "Fp2", "F3", "F4"],
    )
    # Even after 3 identical picks, no proposal is emitted because axis is
    # already a confirmed default.
    prop = _three_picks(user_id, "channels", ["Fp1", "Fp2", "F3", "F4"])
    assert prop is None


def test_reject_proposal_forever_suppresses(fresh_user):
    user_id, _ = fresh_user
    # 3 picks -> proposal. Then reject(forever=True). Then 3 more -> no proposal.
    prop = _three_picks(user_id, "condition", "rest")
    assert prop is not None
    um.reject_proposal(user_id, prop.axis, prop.normalised_value, forever=True)
    # 3 more identical picks should stay silent
    for i in range(5, 8):
        prop2 = um.record_pick(user_id, "condition", "rest", run_id=f"run_{i}", threshold=3)
        assert prop2 is None


def test_apply_confirmed_defaults_to_state_populates_fields(fresh_user):
    user_id, _ = fresh_user
    um.confirm_default(
        user_id,
        "frequency_band",
        {"name": "theta", "fmin": 4.0, "fmax": 8.0},
    )
    um.confirm_default(user_id, "condition", "T0")
    um.confirm_default(
        user_id,
        "channels",
        ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    )
    um.confirm_default(user_id, "reference", "average (CAR)")

    empty_state = {
        "resolved_frequency_band_info": None,
        "resolved_channel_info": None,
        "resolved_condition_value": None,
        "dataset_reference": None,
    }
    store = um.load_preference_store(user_id)
    updates = um.apply_confirmed_defaults_to_state(empty_state, store=store)
    assert updates["resolved_frequency_band_info"]["name"] == "theta"
    assert updates["resolved_frequency_band_info"]["source"] == "user_default"
    assert updates["resolved_condition_value"] == "T0"
    assert len(updates["resolved_channel_info"]["resolved_channels"]) == 7
    assert updates["dataset_reference"] == "average (CAR)"


def test_apply_confirmed_defaults_respects_existing(fresh_user):
    """Pre-existing resolved info should NOT be overwritten by a default."""
    user_id, _ = fresh_user
    um.confirm_default(
        user_id, "frequency_band", {"name": "alpha", "fmin": 8, "fmax": 12}
    )
    state_with_band = {
        "resolved_frequency_band_info": {
            "name": "gamma", "fmin": 30, "fmax": 45, "source": "user_hitl"
        }
    }
    store = um.load_preference_store(user_id)
    updates = um.apply_confirmed_defaults_to_state(state_with_band, store=store)
    # frequency_band already set -> no update for it.
    assert "resolved_frequency_band_info" not in updates


def test_remember_explicit_alpha_band(fresh_user):
    user_id, _ = fresh_user
    out = um.remember_explicit(user_id, "remember that I prefer alpha band", run_id="cmd_1")
    assert out is not None
    assert "alpha" in out.lower()
    store = um.load_preference_store(user_id)
    assert store.confirmed_defaults["frequency_band"]["name"] == "alpha"


def test_remember_explicit_average_reference(fresh_user):
    user_id, _ = fresh_user
    out = um.remember_explicit(user_id, "/remember average reference", run_id="cmd_2")
    assert out is not None
    store = um.load_preference_store(user_id)
    assert "average" in store.confirmed_defaults["reference"].lower()


def test_clear_default(fresh_user):
    user_id, _ = fresh_user
    um.confirm_default(
        user_id, "frequency_band", {"name": "beta", "fmin": 12, "fmax": 30}
    )
    assert um.load_preference_store(user_id).confirmed_defaults.get("frequency_band")
    um.clear_default(user_id, "frequency_band")
    assert "frequency_band" not in um.load_preference_store(user_id).confirmed_defaults


def test_unknown_axis_errors_on_confirm(fresh_user):
    user_id, _ = fresh_user
    with pytest.raises(ValueError):
        um.confirm_default(user_id, "nonexistent_axis", "x")


def test_record_pick_ignores_empty_value(fresh_user):
    user_id, _ = fresh_user
    assert um.record_pick(user_id, "frequency_band", "", "run_e") is None
    assert um.record_pick(user_id, "channels", [], "run_e") is None


def test_gitignore_created(fresh_user):
    _user_id, target_dir = fresh_user
    # Force dir creation + gitignore write by reading/writing a store.
    um.load_preference_store("x")
    gi = target_dir / ".gitignore"
    assert gi.exists()
    content = gi.read_text(encoding="utf-8")
    assert "*_history.jsonl" in content
    assert "*_preferences.json" in content
