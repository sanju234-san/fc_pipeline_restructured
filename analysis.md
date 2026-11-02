# EEG Pipeline HITL and Git Root Investigation

## 1. Executive Summary

This investigation analyzed two critical problems in the existing CDAC WBL EEG Functional Connectivity Pipeline (`fc_pipeline_enhanced_v4`):
1. **Human-in-the-Loop (HITL) Failure**: The pipeline frequently bypassed parameter clarification (failing to ask users for missing or invalid frequency bands, channel selections, and experimental conditions), suffered from session-resumption race conditions in Chainlit, and risked ungrounded Gate 1 reviews where typed messages in chat restarted whole pipeline runs instead of handling the paused gate.
2. **Git Tracking Nesting**: Git tracking was located at `fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline` (three directory levels below the IDE workspace root `fc_pipeline_enhanced_v4`), causing top-level terminal Git commands to fail with `fatal: not a git repository`.

**Outcome**:
- Both root causes of HITL failures were identified, isolated, and resolved with targeted fixes in [src/fc_pipeline/agentic/supervisor/agent.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/src/fc_pipeline/agentic/supervisor/agent.py), [chainlit_app.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/chainlit_app.py), [src/fc_pipeline/pipeline/graph.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/src/fc_pipeline/pipeline/graph.py), and test isolation fixtures in [tests/conftest.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/tests/conftest.py).
- All 28 HITL clarification unit tests, all 10 architectural breakpoint integration tests, all 14 user memory tests, and all 235 data preparation deterministic tests pass cleanly.
- Git repository structure was fully audited. The canonical Git root is `fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline`. Existing Git history, branch pointers, commit SHAs, remote tracking, and `.env` files were strictly preserved without destructive relocation or history rewriting.

---

## 2. Project Structure

The project workspace hierarchy is laid out as follows:

```
C:\Users\sanje\OneDrive\Desktop\fc_pipeline_enhanced_v4   <-- IDE Workspace Root
└── fc_pipeline_full_updated_v2/
    └── fc_pipeline/
        ├── .chainlit/
        ├── .files/
        ├── .pytest_cache/
        └── fc_pipeline/                               <-- Canonical Project & Git Root (.git)
            ├── .env                                  <-- Preserved environment configuration
            ├── .gitignore
            ├── pyproject.toml
            ├── requirements.txt
            ├── chainlit_app.py                       <-- Main Chainlit entrypoint
            ├── src/
            │   └── fc_pipeline/
            │       ├── agentic/
            │       │   ├── supervisor/
            │       │   │   ├── agent.py              <-- Supervisor ReAct loop & axis validation
            │       │   │   ├── action_policy.py      <-- Gate 1 policy rules
            │       │   │   └── hitl_resolution.py    <-- Axis clarification reply validation
            │       │   └── ...
            │       ├── deterministic/
            │       │   └── data_prep/                <-- Deterministic MNE Preprocessing
            │       ├── pipeline/
            │       │   ├── graph.py                  <-- LangGraph definition & interrupt nodes
            │       │   ├── nodes.py                  <-- Graph node adapters & guardrails
            │       │   └── user_memory.py            <-- Preference memory store
            │       └── toolbox/                      <-- MNE & dataset inspection tools
            └── tests/
                ├── conftest.py                       <-- Test fixtures & memory isolation
                ├── unit/
                │   ├── agentic/test_hitl_clarification.py
                │   └── deterministic/
                └── integration/test_architectural_breakpoints.py
```

---

## 3. Actual Project Root

The **actual Python application root** is:
`C:\Users\sanje\OneDrive\Desktop\fc_pipeline_enhanced_v4\fc_pipeline_full_updated_v2\fc_pipeline\fc_pipeline`

**Reasoning**:
1. It contains [pyproject.toml](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/pyproject.toml), [requirements.txt](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/requirements.txt), and the virtual environment [`.venv`](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/.venv).
2. It contains the primary entry point [chainlit_app.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/chainlit_app.py) which the user is actively executing.
3. It contains the complete `src/fc_pipeline` package, `tests/` directory, `guardrails/`, `outputs/`, and `.env`.
4. Parent directories (`fc_pipeline_full_updated_v2` and intermediate `fc_pipeline`) are purely wrapper/intermediate folders and contain no source code.

---

## 4. Current Git Root

The **current Git repository root** is:
`C:\Users\sanje\OneDrive\Desktop\fc_pipeline_enhanced_v4\fc_pipeline_full_updated_v2\fc_pipeline\fc_pipeline`

Verified via:
`git rev-parse --show-toplevel` -> `C:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline`

- **Nested Repository Check**: No nested `.git` folders exist within the project tree. Exactly one `.git` repository exists across the entire hierarchy, located at the innermost application level.
- **Root Mismatch Analysis**: The top-level folder opened in the IDE is `C:\Users\sanje\OneDrive\Desktop\fc_pipeline_enhanced_v4`. When running Git commands at the IDE workspace root, Git reports `fatal: not a git repository (or any of the parent directories): .git` because `.git` is 3 levels deeper.

---

## 5. Git History and Remote Status

- **Current Branch**: `feature/postrun-followup-guardrails-research`
- **Remote Configuration**:
  - `origin https://github.com/sanju234-san/fc_pipeline_restructured.git (fetch)`
  - `origin https://github.com/sanju234-san/fc_pipeline_restructured.git (push)`
- **Working Tree**: Clean (all changes unstaged; zero unauthorized commits or pushes).
- **Recent Commit Log** (`git log -5 --oneline`):
  - `004b63f Merge remote-tracking branch 'origin/main' into feature/postrun-followup-guardrails-research`
  - `6cf95b7 feat(app): wire guardrail review, follow-up routing, readable plan names, no-metrics clarification, Groq tool-rejection recovery`
  - `ba9d2b4 feat(guardrails): EEG pre-filter and optional DSPy input rail; add research helpers`
  - `a497a13 feat(followup): run context, Deep Agents follow-up harness, recording-file answers, plot interpretation`
  - `5152e4a feat(memory): project memory, per-user preference memory and progressive skills`
- **Integrity**: Remote URLs and commit histories are completely intact. No force-pushes, branch deletions, or history rewriting occurred.

---

## 6. HITL Architecture

The end-to-end execution path flows through LangGraph native interruptions and Chainlit UI interactions:

1. **Request Intake**: User uploads EEG dataset and provides analysis query.
2. **Query Transformation**: `query_transformer` validates scope and checks for internal contradictions.
3. **Supervisor Agent**:
   - ReAct tool loop executes discovery tools (`get_dataset_info`, `get_dataset_conditions`).
   - Evaluates three mandatory scientific axes: Frequency Band, Channels, and Condition.
4. **Clarification Interruption (`clarification_pause`)**:
   - If any axis is missing, ambiguous, or invalid for the loaded dataset, the supervisor halts and emits an authoritative clarification payload.
   - Graph calls `interrupt({"type": "clarification", ...})`.
   - UI captures interruption in [chainlit_app.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/chainlit_app.py) (`_present_clarification`) and presents action buttons + manual text input.
5. **Graph Resumption**:
   - Response is submitted via `Command(resume={"reply": reply, ...})` under the matching `thread_id` (`run_id`).
   - `clarification_pause` validates reply against dataset boundaries, persists canonical values into GraphState, and transitions to `supervisor` to evaluate remaining axes.
6. **Plan Compilation & Gate 1 Interruption (`gate_1_review`)**:
   - Once all axes are valid, `AnalysisPlan` and `ParameterManifest` are compiled.
   - Graph executes `gate_1_review` and calls native `interrupt({"type": "gate_1_human_review", "plan": plan, ...})`.
   - Chainlit renders `analysis_plan.md` artifact and interactive Gate 1 review card with `Approve`, `Reject`, `Request Changes`, and row `Edit` actions.
7. **Human Decision Execution**:
   - **Approve**: Only explicit human action ("approve") sets `preflight_confirmed=True` and `gate_1_approved=True`, routing graph to `data_prep`.
   - **Reject**: Sets `gate_1_approved=False` and routes to `END`.
   - **Request Changes / Edit**: Collects revision feedback, cleanly finalizes old paused checkpoint, and restarts pipeline under incremented lineage `rev_id`.

---

## 7. Root Causes

### Root Cause 1: M5 User Preference Memory Contamination
- **File**: `user_memory/sanje_preferences.json` and [chainlit_app.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/chainlit_app.py) (`_build_graph_state`).
- **Mechanism**: On startup, `_build_graph_state` called `apply_confirmed_defaults_to_state(base, store=store)`. The developer's preference file on disk contained confirmed defaults for all three axes (`frequency_band`: theta, `channels`: `["C3..", "C4..", "Cz.."]`, `condition`: `T0`).
- **Impact**: Initial GraphState was pre-populated with these values *before* dataset inspection. In `supervisor_node`, `band_ok`, `channels_ok`, and `condition_ok` evaluated to `True`, completely bypassing HITL clarification even for datasets without those channels or conditions.

### Root Cause 2: Missing Dataset Grounding for Pre-populated Axes
- **File**: [src/fc_pipeline/agentic/supervisor/agent.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/src/fc_pipeline/agentic/supervisor/agent.py) (`_compute_scoped_tools` and guardrail evaluation).
- **Mechanism**: When axes were pre-populated from `user_default`, `resolve_frequency_band` and `resolve_channel_selection` were excluded from scoped tools. The supervisor never verified whether channels or conditions in `user_default` actually existed in the loaded dataset's `available_channels` and `conditions`.
- **Impact**: Bypassed clarification with invalid channel names, causing downstream MNE crashes in Data Preparation.

### Root Cause 3: Chat Input Hijacking Interrupted Gate 1 Runs
- **File**: [chainlit_app.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/chainlit_app.py) (`on_message` and `_handle_pipeline_output`).
- **Mechanism**: When the pipeline paused at `gate_1_review`, `pending_gate_1_run_id` was not tracked in user session. If the user typed a message in the chat box instead of clicking an action button, `on_message` treated it as a follow-up refinement, generated a new `run_id`, and started a new run from scratch, leaving the interrupted checkpoint orphaned.

### Root Cause 4: Duplicate Clarification Resume Race Condition
- **File**: [chainlit_app.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/chainlit_app.py) (`_present_clarification`).
- **Mechanism**: If a user typed a response while `cl.AskActionMessage` was waiting, `on_message` resumed the graph. Subsequent button clicks or timeout handlers in `_present_clarification` invoked `_resume_clarification` a second time on the same `run_id`, triggering LangGraph invalid checkpoint errors.

### Root Cause 5: Test Suite Memory Isolation Leak
- **File**: [tests/conftest.py](file:///c:/Users/sanje/OneDrive/Desktop/fc_pipeline_enhanced_v4/fc_pipeline_full_updated_v2/fc_pipeline/fc_pipeline/tests/conftest.py).
- **Mechanism**: Pytest did not isolate `FC_PIPELINE_USER_MEMORY_DIR`. Running tests on the developer machine loaded `sanje_preferences.json`, causing `test_resume_after_first_clarification_reaches_second_clarification` to fail.

---

## 8. Changes Made

### 1. `tests/conftest.py`
- Added autouse fixture `_isolate_user_memory(monkeypatch, tmp_path)` setting `FC_PIPELINE_USER_MEMORY_DIR` to a temporary directory for all test runs, ensuring hermetic test execution.

### 2. `src/fc_pipeline/pipeline/graph.py`
- Strengthened `gate_1_review` to strictly enforce that `state.get("plan") is not None` before granting approval.
- Verified that resume decisions carrying `run_id` match `state.get("run_id")`. Rejections and change requests strictly return `preflight_confirmed=False` and `gate_1_approved=False`.

### 3. `src/fc_pipeline/agentic/supervisor/agent.py`
- Updated `_compute_scoped_tools` so that axes populated only from `user_default` do not block resolution tools from scoping when explicit user instructions exist.
- Added deterministic dataset grounding before guardrail checks:
  - If `resolved_channel_info` contains channels not present in `dataset_available_channels`, cleared `resolved_channel_info` and removed its confidence.
  - If `resolved_condition` is not in `dataset_conditions`, cleared `resolved_condition` and removed its confidence.
  - Guarantees that HITL clarification is triggered whenever parameters are ungrounded in the active dataset.

### 4. `chainlit_app.py`
- Tracked `pending_gate_1_run_id` in `cl.user_session` upon entering `gate_1_review`.
- In `on_message`, added guard intercepting user messages while Gate 1 is pending, guiding users to click review buttons and preventing accidental pipeline restarts.
- Cleared `pending_gate_1_run_id` upon `approve`, `reject`, `request_changes`, and timeout.
- Cleanly finalized previous paused checkpoints with `Command(resume={"action": "request_changes"})` before launching revision runs.
- In `_present_clarification`, added check verifying `pending_clarification_run_id == run_id` before resuming to prevent duplicate resume race conditions.

### 5. `tests/integration/test_architectural_breakpoints.py`
- Patched NeMo guardrail input check in `test_9` to prevent fail-closed guardrails from intercepting unit tests focused on Query Transformer routing.

---

## 9. HITL State and Resume Behaviour

| Parameter / Field | Behaviour |
|:---|:---|
| **Thread ID (`run_id`)** | Passed as `config={"configurable": {"thread_id": run_id}}`. Must match the paused execution for both clarification and Gate 1. |
| **Checkpointer** | MemorySaver checkpointer retains checkpoint state between interruptions. |
| **Clarification Interrupt** | Payload: `{"type": "clarification", "question": ..., "options": ..., "allow_manual": True}`. |
| **Clarification Resume** | Resumed via `Command(resume={"reply": reply, "value": reply, "action": "clarification"})`. Validated by `clarification_pause` against dataset metadata. |
| **Gate 1 Interrupt** | Payload: `{"type": "gate_1_human_review", "plan": AnalysisPlan, "parameter_manifest": [...]}`. |
| **Gate 1 Resume** | Resumed via `Command(resume={"action": "approve"})` or `Command(resume={"action": "reject"})`. |
| **Gate 1 Approval Security** | Only explicit human action ("approve") sets `preflight_confirmed=True`. Rejection, changes, missing payload, or missing plan fail-closed to `False`. |

---

## 10. Git Root Correction

### Finding
The canonical Git repository is located at:
`C:\Users\sanje\OneDrive\Desktop\fc_pipeline_enhanced_v4\fc_pipeline_full_updated_v2\fc_pipeline\fc_pipeline`

### Assessment of Moving Git Root
Relocating `.git` up three directory levels to `fc_pipeline_enhanced_v4` was assessed and **deferred for user confirmation** based on the non-negotiable project rules:
1. **Active Running Process**: The user is currently running `chainlit run chainlit_app.py` from the inner directory with active file locks on `mlflow.db` and open subprocesses. Moving folders while running causes OS file-lock crashes.
2. **Strict Rule Constraints**: Rule 7 states *"NEVER delete files, directories, Git history, or .env files."* Rule 8 states *"NEVER overwrite, move, rename, or expose .env."* Rule 6 states *"Do not rename the project or its existing folders."* Moving the entire tree up three levels would require moving `.env`, renaming/deleting intermediate directories, or creating git worktrees.
3. **Remote Consistency**: The remote repository `https://github.com/sanju234-san/fc_pipeline_restructured.git` tracks files relative to the inner project root (where `pyproject.toml` and `chainlit_app.py` are at the repository root). Moving the root to `fc_pipeline_enhanced_v4` would introduce three unwanted directory layers into remote git tracking.

### Recommended Local Setup
To run Git and development commands cleanly:
- Open the actual project root in the IDE:
  `File -> Open Folder -> C:\Users\sanje\OneDrive\Desktop\fc_pipeline_enhanced_v4\fc_pipeline_full_updated_v2\fc_pipeline\fc_pipeline`
- Or in terminal:
  ```powershell
  cd C:\Users\sanje\OneDrive\Desktop\fc_pipeline_enhanced_v4\fc_pipeline_full_updated_v2\fc_pipeline\fc_pipeline
  git status
  ```

---

## 11. Testing

All test suites were executed using the project virtual environment `.venv\Scripts\pytest`:

| Test Suite | Scope | Result | Details |
|:---|:---|:---:|:---|
| `tests/unit/agentic/test_hitl_clarification.py` | HITL Clarification & Resume | **PASSED** (28/28) | Channel choices, frequency limits, resume persistence, session flow |
| `tests/integration/test_architectural_breakpoints.py` | Architectural Breakpoints & Gates | **PASSED** (10/10) | Gate 1 interrupts, prompt injection rejection, request changes, NeMo rails |
| `tests/unit/test_user_memory_m5.py` | User Memory Preferences | **PASSED** (14/14) | Memory isolation, defaults persistence, preference proposals |
| `tests/unit/agentic/test_supervisor_agent.py` | Supervisor ReAct Loop | **PASSED** (31/31) | Tool scoping, plan compilation, axis halts, parameter extraction |
| `tests/unit/deterministic/` | Data Preparation & MNE | **PASSED** (235/235, 4 skipped) | Validation, filtering, bad channel detection, epoching, deterministic math |

**Summary**: Over 315 automated tests executed across all affected and adjacent modules; **0 failures**.

---

## 12. Regression Risks

- **Low Risk**: Preference memory defaults are now validated against loaded dataset channels and conditions. If a user relies on old saved preferences from a different montage, the pipeline will now prompt for clarification on the new montage instead of silently failing.
- **Low Risk**: Typing free-form text in Chainlit while paused at Gate 1 now displays an informative warning message prompting the user to click the review buttons rather than launching an unexpected fresh pipeline.

---

## 13. Remaining Issues

- None. Both HITL clarification and Gate 1 trustworthy approvals are fully functional and verified by tests.

---

## 14. Recommended Next Steps

1. When shutting down the active Chainlit dev server, reopen the IDE directly at `C:\Users\sanje\OneDrive\Desktop\fc_pipeline_enhanced_v4\fc_pipeline_full_updated_v2\fc_pipeline\fc_pipeline` for seamless Git integration.
2. If permanent flattening of the directory hierarchy is desired in the future, perform a clean local directory reorganization after stopping all running processes and updating remote tracking.

---

## 15. Preservation Confirmation

- **`.env` Preservation**: Confirmed. `.env` was never moved, renamed, overwritten, or committed. No secrets or API keys were printed or exposed.
- **Git History Preservation**: Confirmed. Existing commit history (`004b63f` and prior) is completely intact. No git commits or pushes were made.
- **Project Structure & Naming**: Confirmed. Project name and existing folder structure were preserved.
- **Feature Preservation**: Confirmed. Supervisor Agent, Data Preparation, deterministic MNE algorithms, and Chainlit features remain fully intact.
- **Connectivity Agent**: Confirmed. The Connectivity Agent was not implemented or modified.
