"""Tests verifying optional SqliteSaver persistence per thread_id in compile_pipeline_app."""

from __future__ import annotations

from pathlib import Path

from fc_pipeline.pipeline.graph import compile_pipeline_app


def test_compile_pipeline_app_sqlite_persistence(tmp_path: Path):
    db_file = str(tmp_path / "checkpoints.sqlite")
    app1 = compile_pipeline_app(sqlite_path=db_file)
    cfg_a = {"configurable": {"thread_id": "run_persist_a"}}
    cfg_b = {"configurable": {"thread_id": "run_persist_b"}}

    app1.update_state(cfg_a, {"run_id": "run_persist_a", "gate_1_approved": False})
    app1.update_state(cfg_b, {"run_id": "run_persist_b", "gate_1_approved": True})

    # Recompile a fresh app instance pointing to the same SQLite file
    app2 = compile_pipeline_app(sqlite_path=db_file)
    snap_a = app2.get_state(cfg_a)
    snap_b = app2.get_state(cfg_b)

    assert snap_a.values["run_id"] == "run_persist_a"
    assert snap_a.values["gate_1_approved"] is False
    assert snap_b.values["run_id"] == "run_persist_b"
    assert snap_b.values["gate_1_approved"] is True
