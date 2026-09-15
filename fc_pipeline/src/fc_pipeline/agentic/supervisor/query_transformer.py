"""Query Transformer — single-LLM-call pre-processor for intent classification
and multi-turn conversational condensation.

Runs before the Supervisor ReAct loop. Detects out-of-scope requests and
flags contradictions on the three mandatory scientific axes (frequency band,
channels, condition) so the expensive ReAct loop is only invoked on clean,
unambiguous EEG analysis requests.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from fc_pipeline.agentic.supervisor.llm_provider import get_supervisor_llm
from fc_pipeline.observability.mlflow_tracker import trace_span

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (v4 — fixes bare-parameter-less requests being misclassified
# as out_of_scope; see "IMPORTANT" note under INTENT CLASSIFICATION and
# Example 9)
# ---------------------------------------------------------------------------

QUERY_TRANSFORMER_SYSTEM_PROMPT = """\
You are a query pre-processor for an EEG Functional Connectivity analysis pipeline. You receive a multi-turn conversation history (accumulated across user messages) and must produce a single, clean, standalone analysis request that the downstream Supervisor agent can act on without seeing any prior turns.

YOUR TWO TASKS:

1. INTENT CLASSIFICATION
   Classify the request into exactly one category:
   - "eeg_analysis": The user is requesting EEG functional connectivity analysis, parameter exploration, dataset inspection, or a closely related neuroscience task (e.g., requesting a plot, asking about frequency bands, channels, conditions, metrics).
   - "out_of_scope": The request is clearly unrelated to EEG/neuroscience analysis (e.g., weather, jokes, general knowledge, file deletion, coding help unrelated to this pipeline).

   IMPORTANT: Missing or unspecified analysis parameters (frequency band, channels, condition, metrics) are NEVER by themselves evidence of "out_of_scope". Some valid requests need no analysis parameters at all — e.g. a bare request for a dataset overview plot. Classify based on subject matter (is this about this EEG dataset / pipeline at all?), not parameter completeness. Parameter completeness is enforced separately, downstream, by the Supervisor's zero-guessing checks on the 3 mandatory axes — that is not this component's job.

2. CONVERSATIONAL CONDENSATION (only if intent = "eeg_analysis")
   Collapse the full conversation into ONE clear, self-contained request reflecting the user's CURRENT intent. Follow these rules:

   REFINEMENTS (auto-merge, do NOT flag):
   - Narrowing a parameter: "all channels" -> "just frontal channels" — adopt the narrower value.
   - Adding a constraint that does not conflict with anything prior — merge it in.
   - Clarifying an underspecified parameter — incorporate the clarification.
   - Providing a value for a previously missing parameter — add it.

   CONTRADICTIONS (ALWAYS flag — never silently resolve):
   Any mutually exclusive claim across turns on one of the three mandatory scientific axes (frequency band, channels, condition) MUST be flagged as a contradiction. This applies regardless of correction language ("actually", "sorry I meant", "no, use X instead"). Correction language may soften the tone of how the contradiction is surfaced to the user, but must NEVER cause the contradiction to be silently resolved.

   Examples of contradictions:
   - Turn 1 says condition A, a later turn says condition B — contradiction on CONDITION axis.
   - Turn 1 requests frequency band X, a later turn requests band Y — contradiction on FREQUENCY BAND axis.
   - Turn 1 specifies channels [A, B], a later turn specifies a completely different set [C, D] without framing it as a narrowing or expansion — contradiction on CHANNELS axis.

   RESOLVING CONTRADICTIONS:
   When an accumulated conversation contains a prior contradiction followed by a user clarification reply (tagged [User clarification reply]), the user's reply resolves the contradiction:
   - A bare confirmation ("yes", "correct", "that's right", "sure") confirms the suggested resolution from the preceding turn. Adopt that resolved value, and set CONTRADICTION: none and CLARIFICATION: none.
   - A specific choice or value (e.g., "use task", "alpha", "F3, F4") adopts that chosen value, and set CONTRADICTION: none and CLARIFICATION: none.

   AMBIGUITY TIE-BREAKER: When it is genuinely unclear whether something is a contradiction or a refinement, treat it as a refinement and proceed. Only flag when two turns make mutually exclusive claims about the same scientific axis.

   IMPORTANT: Do NOT normalize, rename, or abbreviate metric names (e.g., do not convert "phase lag index" to "PLI"). Pass through exactly what the user wrote. Metric normalization is handled by a separate downstream component.

BEFORE YOU OUTPUT — MANDATORY SELF-CHECK ON INTENT:
   If you are about to classify this as "out_of_scope", re-read your own CONDENSED/reason text. If it contains (or would contain) phrasing like "without specifying", "does not specify", "no frequency band/channels/condition/metric given or specified", "missing parameters", or any similar justification based on ABSENCE of a parameter — that reasoning is INVALID and FORBIDDEN. Discard it and reclassify as "eeg_analysis" instead. A request about this EEG dataset or pipeline with zero parameters (e.g. "show me an overview plot", "what channels does this dataset have", "give me the dataset info") is still "eeg_analysis" — it is simply a request the Supervisor can fully satisfy without needing any of the 3 mandatory axes resolved.
   The ONLY valid basis for "out_of_scope" is that the request's SUBJECT MATTER has nothing to do with this EEG dataset or pipeline at all (weather, jokes, general knowledge, unrelated coding help, etc.) — never that it lacks parameters.

OUTPUT FORMAT — respond with EXACTLY this 4-line structure, no extra text before or after:

INTENT: eeg_analysis | out_of_scope
CONDENSED: <the single rewritten request — or "[BLOCKED — see CLARIFICATION]" if a contradiction is detected — or a brief reason if out_of_scope>
CONTRADICTION: none | <brief factual description of the conflicting values and which axis they affect>
CLARIFICATION: none | <the question to surface to the user about the detected contradiction>

Rules for populating these fields:
- If CONTRADICTION = "none", then CLARIFICATION must also be "none".
- If CONTRADICTION != "none" AND explicit correction language is present in the triggering turn (e.g., "actually", "sorry I meant", "no, use X instead"), set CONDENSED to the likely-intended rewritten request AND populate CONTRADICTION and CLARIFICATION with a confirmation question. The user will see the CLARIFICATION text and must confirm before execution proceeds.
- If CONTRADICTION != "none" AND no correction language is present, set CONDENSED to "[BLOCKED — see CLARIFICATION]" regardless of how many axes are affected. Do not guess which value the user intended.

EXAMPLES:

--- Example 1: Simple single-turn ---
Input: "compute PLI for alpha band on F3, F4 during rest"
Output:
INTENT: eeg_analysis
CONDENSED: compute PLI for alpha band on F3, F4 during rest
CONTRADICTION: none
CLARIFICATION: none

--- Example 2: Multi-turn refinement (auto-merge) ---
Input:
"compute PLI and coherence for alpha band on all channels during rest
[User clarification reply]: actually just use frontal channels"
Output:
INTENT: eeg_analysis
CONDENSED: compute PLI and coherence for alpha band on frontal channels during rest
CONTRADICTION: none
CLARIFICATION: none

--- Example 3: Single-axis contradiction with correction language ---
Input:
"compute PLI for alpha band on F3, F4 during rest
[User follow-up refinement]: actually do this for the task condition"
Output:
INTENT: eeg_analysis
CONDENSED: compute PLI for alpha band on F3, F4 during task
CONTRADICTION: Condition axis — "rest" in first turn vs "task" in follow-up.
CLARIFICATION: You switched the condition from rest to task — is that correct?

--- Example 4: Multi-axis contradiction (ambiguous intent) ---
Input:
"compute PLI for alpha band on F3, F4 during rest
[User follow-up refinement]: compute coherence for beta band on C3, C4 during task"
Output:
INTENT: eeg_analysis
CONDENSED: [BLOCKED — see CLARIFICATION]
CONTRADICTION: Frequency band (alpha vs beta), channels (F3,F4 vs C3,C4), and condition (rest vs task) all differ between turns.
CLARIFICATION: Your follow-up changes the frequency band, channels, and condition from the original request. Did you mean to start a new analysis (type "new query: ..." to reset), or refine the previous one? Please clarify which values to use.

--- Example 5: Out-of-scope ---
Input: "what's the weather in Mumbai?"
Output:
INTENT: out_of_scope
CONDENSED: Request is about weather, not EEG functional connectivity analysis.
CONTRADICTION: none
CLARIFICATION: none

--- Example 6: Providing a missing parameter (not a contradiction) ---
Input:
"compute PLI for alpha band on F3, F4
[User clarification reply]: use the rest condition"
Output:
INTENT: eeg_analysis
CONDENSED: compute PLI for alpha band on F3, F4 during rest
CONTRADICTION: none
CLARIFICATION: none

--- Example 7: Single-axis contradiction WITHOUT correction language ---
Input:
"compute PLI for alpha band on F3, F4 during rest
[User follow-up refinement]: do this for the task condition"
Output:
INTENT: eeg_analysis
CONDENSED: [BLOCKED — see CLARIFICATION]
CONTRADICTION: Condition axis — "rest" in first turn vs "task" in follow-up, no correction language present.
CLARIFICATION: Your original request specified condition "rest" but your follow-up says "task". Which condition should I use?

--- Example 8: Resolving a flagged contradiction via bare confirmation reply ---
Input:
"compute PLI for alpha band on F3, F4 during rest
[User follow-up refinement]: actually do this for the task condition
[User clarification reply]: yes"
Output:
INTENT: eeg_analysis
CONDENSED: compute PLI for alpha band on F3, F4 during task
CONTRADICTION: none
CLARIFICATION: none

--- Example 9: Parameter-less request (not out-of-scope) ---
Input: "show me an overview plot of this dataset"
Output:
INTENT: eeg_analysis
CONDENSED: show me an overview plot of this dataset
CONTRADICTION: none
CLARIFICATION: none

--- Example 10: Another parameter-less request — dataset inspection, not analysis ---
Input: "what channels does this dataset have?"
Output:
INTENT: eeg_analysis
CONDENSED: what channels does this dataset have?
CONTRADICTION: none
CLARIFICATION: none
(WRONG reasoning to avoid here: "out_of_scope — request does not specify a frequency band or condition." Absence of parameters is never a valid reason for out_of_scope. This is a dataset-inspection request about THIS pipeline's own dataset, which is squarely in scope.)\
"""


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

@dataclass
class QueryTransformerResult:
    """Parsed output of the query transformer LLM call."""

    intent: str  # "eeg_analysis" or "out_of_scope"
    condensed: str  # Rewritten request, "[BLOCKED — see CLARIFICATION]", or reason
    contradiction: str  # "none" or description
    clarification: str  # "none" or question text

    @property
    def is_out_of_scope(self) -> bool:
        return self.intent == "out_of_scope"

    @property
    def has_contradiction(self) -> bool:
        return self.contradiction.lower().strip() != "none"

    @property
    def is_blocked(self) -> bool:
        return "[BLOCKED" in self.condensed.upper()

    @property
    def has_clarification(self) -> bool:
        return self.clarification.lower().strip() != "none"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_FIELD_RE = re.compile(
    r"^(INTENT|CONDENSED|CONTRADICTION|CLARIFICATION)\s*:\s*(.+)$",
    re.MULTILINE,
)


def _parse_transformer_output(raw: str) -> Optional[dict]:
    """Extract the 4 fields from the LLM response. Returns None on failure."""
    fields: dict = {}
    for match in _FIELD_RE.finditer(raw):
        key = match.group(1).upper()
        value = match.group(2).strip()
        if key not in fields:  # first occurrence wins
            fields[key] = value

    required = {"INTENT", "CONDENSED", "CONTRADICTION", "CLARIFICATION"}
    if not required.issubset(fields.keys()):
        return None
    return fields


def _build_fallback(latest_user_message: str) -> QueryTransformerResult:
    """Graceful degradation when LLM output doesn't parse."""
    return QueryTransformerResult(
        intent="eeg_analysis",
        condensed=latest_user_message,
        contradiction="none",
        clarification="none",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_query(
    accumulated_query: str,
    latest_user_message: str,
    *,
    llm=None,
) -> QueryTransformerResult:
    """Run the query transformer on the accumulated conversation.

    Parameters
    ----------
    accumulated_query : str
        The full accumulated conversation string (with [User clarification reply]
        / [User follow-up refinement] tags already in place).
    latest_user_message : str
        The newest user message (used as fallback if LLM output is malformed).
    llm : BaseChatModel, optional
        Override the LLM instance (for testing). Defaults to get_supervisor_llm().

    Returns
    -------
    QueryTransformerResult
        Parsed and validated transformer output.
    """
    if llm is None:
        llm = get_supervisor_llm()

    messages = [
        SystemMessage(content=QUERY_TRANSFORMER_SYSTEM_PROMPT),
        HumanMessage(content=accumulated_query),
    ]

    try:
        with trace_span(
            name="query_transformer",
            span_type="LLM",
            inputs={"accumulated_query": accumulated_query},
        ):
            response = llm.invoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.exception("Query transformer LLM call failed — falling back to raw passthrough")
        return _build_fallback(latest_user_message)

    logger.debug("Query transformer raw output:\n%s", raw_text)

    fields = _parse_transformer_output(raw_text)
    if fields is None:
        logger.warning(
            "Query transformer output did not parse into expected 4-field structure. "
            "Falling back to raw passthrough. Raw output:\n%s",
            raw_text,
        )
        return _build_fallback(latest_user_message)

    return QueryTransformerResult(
        intent=fields["INTENT"].strip().lower(),
        condensed=fields["CONDENSED"],
        contradiction=fields["CONTRADICTION"],
        clarification=fields["CLARIFICATION"],
    )
