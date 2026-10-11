# EEG Functional Connectivity Pipeline — Project Memory

## 1. Hard Constraints (never break these)

1. **5-Node DAG order is fixed**: Supervisor -> Data Prep -> Connectivity -> Evaluator -> Synthesis.
   You may NOT reorder, merge, or bypass a node. Interrupts / HITL pauses only occur
   inside Supervisor (HITL axis clarification) or at Gate 1 (preflight manifest review)
   or Gate 2 (prepublication verification).

2. **Deterministic modules are offline-testable with ZERO LLM imports**.
   Affected modules (do NOT import langchain / deepagents / OpenAI SDK):
   - `fc_pipeline.deterministic.data_prep.*`
   - `fc_pipeline.deterministic.connectivity.*`
   - `fc_pipeline.deterministic.sanity_checks.*`
   - `fc_pipeline.config.*`
   - `fc_pipeline.schemas.*`
   - `fc_pipeline.gates.cli.*` (CLI gate review is offline)
   - `fc_pipeline.pipeline.run_context` (deterministic QA path)
   - `fc_pipeline.agentic.followup.context_window` (window manager is offline-testable;
     summariser LLM is optional and must degrade to a pure-tokenizer compression)

3. **Connectivity metrics are computed via MNE-Connectivity / NumPy only**.
   The canonical supported metric set is exactly:
   - `PLI`        (Phase Lag Index)
   - `wPLI`       (weighted Phase Lag Index)
   - `ImCoh`      (Imaginary Coherence)
   - `PLV`        (Phase Locking Value)
   - `Coherence`  (Magnitude-squared Coherence)
   Any other metric name must be rejected as unsupported, NEVER invented.

4. **Centralised thresholds live ONLY in `config/thresholds.py`**.
   Canonical values:
   - `TAU_PHASE = 0.20`       (classification threshold for phase-coupled links)
   - `TAU_ZEROLAG = 0.35`     (classification threshold for zero-lag / volume-conducted links)
   - `SUPERVISOR_CONFIDENCE_THRESHOLD = 0.80`
   All other code READS these constants; never re-defines them inline.

5. **Gate 1 (preflight) manifest review triggers for confidence-based needs
   OR any elevated-risk row**. If `needs_human_input == True` on ANY row, or if
   `risk_tier == "elevated"` on ANY row, the Supervisor node MUST interrupt at
   `gate_1_review` and the UI pauses until the human approves.

6. **Data sources are real EEG only (Chainlit UI refuses demo/synthetic fallbacks)**.
   Only `.fif` and `.edf` (or a `.zip` containing exactly one of them) are accepted.
   Unit tests may use synthetic fixtures ONLY for signal-processing correctness,
   never for the actual UX path.

## 2. Architecture map

```
fc_pipeline/
  agentic/
    supervisor/            ReAct agent + HITL resolver. Uses get_supervisor_llm().
      agent.py             Supervisor node body + turn policy.
      query_transformer.py Pre-processes user text before the ReAct loop.
      hitl_resolution.py   Deterministic apply_clarification_reply(freq/ch/cond).
      llm_provider.py      Shared LangChain ChatModel builder.
      tools/               Dataset info / conditions / freq band / channel / plotting.
    evaluator/             VLM visual topology verifier (stub — body TBD, schema in state).
    synthesis/             Report writer (stub — body TBD, AGENTS.md-viva SKILL.md driven).
    followup/              Post-run Deep Agents harness.
      context_window.py    PostRunContextWindowManager (offline-testable).
      agents.py            run_followup_agent + 3 DeclarativeSubagents.
  deterministic/
    data_prep/             Cleaning, CAR reference, epoching. NO LLM.
    connectivity/          PLI/wPLI/ImCoh/PLV/Coherence + classification + plotting. NO LLM.
    sanity_checks/         Matrix symmetry / diagonal / finite-value checks. NO LLM.
  gates/
    cli/gate1_preflight.py Gate 1 manifest review CLI.
    cli/gate2_prepublication.py Gate 2 report-signoff CLI.
  config/thresholds.py     Single source of truth for all numeric thresholds.
  schemas/                 manifest, state, clarification. Pure Pydantic/dataclass. NO LLM.
  pipeline/
    graph.py / nodes.py    LangGraph 5-node DAG + edges.
    run_context.py         save/load/restore + deterministic answer_run_question().
  observability/mlflow_tracker.py  start_run / log_params / log_manifest / log_result.
  guardrails/              NeMo Guardrails Colang 1.0: config.yml, rails.co.
  chainlit_app.py          UI + session lifecycle. Starts LangGraph, routes HITL, follow-ups.
  AGENTS.md                THIS FILE. Loaded via memory.py as project memory.
  skills/                  Progressive-domain SKILL.md files (see skills/spec).
  user_memory/             Per-user preference JSONL (gitignored). Created on demand.
```

## 3. HITL + Gate rules (operational)

- **Supervisor HITL axes**: exactly 3 scientific axes are asked if unresolved after
  ReAct loop + tools:
  1. Frequency band (name or numeric Hz range)
  2. Channels (labels OR brain-region keyword with canonical mapping)
  3. Condition / event label
  4. Metric set is OPTIONAL; defaults to all 5 when the user does not specify.

- **Gate 1 pauses** when GraphState shows a pending manifest with any
  `needs_human_input` or `risk_tier=="elevated"`. The only valid resume actions are
  `"approve"` (full manifest), `"revise"`, or per-row edits routed back through
  `apply_clarification_reply` + Supervisor re-validation.

- **Gate 2 (prepublication)** is CLI-only today (`gate2_prepublication.py`).
  Future Synthesis report body must ALWAYS pass the 4 checks (PHI, thresholds,
  numerical consistency, references) before final_report_path is set.

## 4. Observability / reproducibility

- Every run writes a `run_context_{run_id}.json` next to its preprocessed epochs
  (or under `outputs/`), containing plan + data_prep_summary + manifest +
  connectivity / evaluation / synthesis sections as they are populated.
- `mlflow_tracker.start_run` logs `(model_name, endpoint, confidence_threshold,
  tau_phase, tau_zerolag, query, data_path, is_real_data)` as params.
- `log_manifest(manifest)` + `log_result(...)` are ALWAYS called, even on failure,
  so a failed run is still queryable in MLflow.

## 5. Post-run follow-up contract (Deep Agents harness)

- A completed run is **frozen read-only**. Ordinary chat messages route through
  `_handle_completed_run_followup` -> `run_followup_agent`, NEVER to Supervisor.
- The ONLY way to discard a completed run and start a new analysis is the explicit
  `new query:` prefix, `/reset`, or clicking the 🆕 "New query" action button.
- `PostRunContextWindowManager` always compresses the chat window and promotes only
  the query-relevant run-context slice into the active prompt; other sections live
  in the offload registry and are accessible only via `read_offloaded_context` /
  `search_run_context` tools (progressive disclosure).
- When `deepagents` is not installed, `allow_harness_import_errors=True` in
  `run_followup_agent` ensures a deterministic fallback still answers grounded
  topics identical to `answer_run_question`.

## 6. Commands

```bash
# install (from fc_pipeline/ subdir containing pyproject.toml / requirements.txt)
pip install -r requirements.txt

# unit tests — offline bucket (no LLM):
pytest tests/unit/deterministic tests/unit/config tests/unit/schemas \
       tests/unit/gates tests/unit/test_toolbox_registry.py \
       tests/unit/test_run_context.py tests/unit/pipeline -v

# unit tests — full:
pytest -v

# web UI:
chainlit run chainlit_app.py
```

## 7. Style / naming

- Agent system prompts are plain English; "RULE:" prefixes are used for hard
  constraints, and every agent must include a "never invent / never run a new
  analysis without NewQuery dispatch" clause.
- Pydantic schemas live under `schemas/` and are the only authoritative shapes.
  Never add an "ad-hoc dict" shape into GraphState without a schema field first.
- Error messages from a HITL / Gate validator must include the exact available
  values (e.g. "Available: rest, task, T0") — never a generic "invalid input".
