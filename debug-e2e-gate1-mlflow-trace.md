# [OPEN] Debug Session: e2e-gate1-mlflow-trace
## Purpose
End-to-end verification of Task 2 Phase B (Gate 1 HITL) + MLflow tracing instrumentation. Run live Chainlit browser session, exercise all flows, inspect real MLflow trace spans.

## Hypotheses (H1-H5)
| ID | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Chainlit UI: 7-col table / edit buttons render incorrectly in real CSS | pending | |
| H2 | Real LLM: transform_query / Supervisor return unparsable output in live calls | pending | |
| H3 | MLflow: autolog + manual trace() conflict → no spans / broken nesting | pending | |
| H4 | _reapply_overrides: Supervisor returns renamed entries → overrides silently lost | pending | |
| H5 | while loop closure: manifest direct edit not reflected on re-render → stale Approved Value | pending | |

## Session Log
| Step | Action | Result |
|---|---|---|
| 1 | Explore existing MLflow + startup setup | |
| 2 | Instrument MLflow trace spans | |
| 3 | Start MLflow UI + Chainlit | |
| 4 | Browser walkthrough (clean query, contradiction, OOS) | |
| 5 | Gate 1 flows (Approve/Reject/Request Changes/advisory/scientific/multi-edit) | |
| 6 | MLflow Traces inspection | |

## Files Modified
- (none yet)
