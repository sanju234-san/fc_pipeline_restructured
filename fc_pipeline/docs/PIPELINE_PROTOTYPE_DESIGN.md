# Functional Connectivity Analysis at Electrode Level: Pipeline Prototype Design Specification

---

## Changelog

| Version | Date | Summary |
| :--- | :--- | :--- |
| **v4** | 2026-09-07 | **Additive: risk-tier attention layer on top of existing two gates.** `risk_tier: str` (`"low"` \| `"elevated"`) added to `ParameterManifestEntry` (Section 5.1). `"elevated"` covers every threshold/constant whose value changes scientific classification outcomes regardless of resolution confidence: reference scheme, τ_phase, τ_zerolag, ocular\_variance\_ratio, muscle\_power\_threshold, electrode\_pop\_sigma, bad\_channel\_variance\_threshold, min\_cycles. `"low"` covers scientific axes and metric selection. Assigned statically by the Supervisor — no new LLM judgment. Gate 1 checklist table gains a Risk Tier column; confidence flag and risk tier are orthogonal (Section 3.2). Gate 2 description updated: Top-5 VLM topology reads are rendered in a separate labeled subsection, distinct from deterministic PASS/FAIL rows (Sections 3.2 and 6.4). Section 7 Full Confirmed Manifest table gains a Risk Tier column; methodological caveats gain one line noting elevated-risk confirmed values for downstream auditors. No new gate, no schema field removals, no changes to PASS/WARNING/FAIL or confidence mechanics. |
| **v3** | 2026-09-07 | **Two additive extensions, no architectural reversals.** (1) Evaluator CSV: Constraint 3 relaxed from one CSV stage to exactly two — Connectivity Analysis (unchanged) and Evaluator (new `evaluation_summary_<run_id>.csv`, one row per check with columns `run_id`, `check_category`, `check_name`, `result`, `confidence`, `affected_channels`, `detail`). `evaluation_summary_csv_path` added to GraphState; Step 4 (CSV Serialization) added to Section 6.4; Section 7 report spec updated with cross-run tracking note. (2) MLflow passive observability: MLflow added to Section 4.1 tech stack as a thin wrapper reading GraphState after each node — params logged at Gate 1, metrics and artifacts logged as each node completes; new Section 13 added; old Section 12 renumbered to 14. MLflow is a passive side effect: no new node, no GraphState fields, no new human checkpoint, failures are warnings only. |
| **v2** | 2026-09-07 | **Architecture simplification after two-round review.** (1) Three-gate mechanism (A/B/C) replaced by two gates: **Gate 1** (Pre-Flight Checklist, merges old Gate A + Supervisor-side Gate B) and **Gate 2** (Pre-Publication Review, replaces old Gate C). (2) `GateBEscalation` model and `gate_b_escalations` state field removed — Evaluator now always completes a full single-shot pass and attaches a `confidence: float` to each check instead of interrupting mid-run. (3) Artifact detection (ocular, muscle, electrode pop) moved from VLM visual judgment to deterministic statistical heuristics with named, manifest-visible threshold constants. (4) VLM scope narrowed strictly to Top-5 unresolved zero-lag pair topology reads. (5) `validate_metric_support` removed as a tool; metric canonicalization is now inline Supervisor logic against a hardcoded 5-item lookup table. (6) NetworkX dependency removed — degree/density/modularity metrics were declared but never surfaced in GraphState or the report. (7) Cosmetic constants (colormap, DPI, file-naming) removed from both gates; documented once in Appendix A as hardcoded named constants. |
| v1 | — | Initial design (3-gate, Gate B mid-run interrupt, VLM artifact detection, NetworkX declared). |

---

## 1. Problem Statement

Functional connectivity (FC) analysis measures statistical dependencies and phase synchronization between electrophysiological signals recorded at different EEG electrodes within defined frequency bands and experimental conditions.

In conventional laboratory practice, this analysis is performed via custom, one-off Python or MATLAB scripts. This manual approach is brittle and hard to audit:
- **Hidden, Ad-Hoc Parameter Decisions:** Choices regarding frequency boundaries, channel subsets, montages, and references are buried inside imperative code and rarely documented consistently.
- **Silent Assumptions:** Scripts frequently make default assumptions (e.g., assuming 8–12 Hz for alpha or picking an arbitrary reference) without verifying whether the underlying recording or hypothesis supports them.
- **Single-Metric Tunnel Vision:** Pipelines typically compute a single connectivity metric in isolation, blinding the researcher to metric-specific biases. In particular, single-metric analyses cannot distinguish true phase synchronization from volume-conduction artifacts.
- **Lack of Reproducibility:** Subsequent researchers cannot easily trace why specific parameters were selected or verify whether preprocessing artifacts affected the output metrics.

This agent-based system solves these problems by providing an automated, traceable pipeline that accepts natural-language analysis requests, interactively resolves ambiguity against dataset metadata without guessing, executes deterministic signal processing, computes a comprehensive multi-metric diagnostic battery, subjects results to multimodal quality inspection, and generates a self-contained, auditable scientific report.

Rather than computing a single metric in a vacuum or leaping to unsupported causal conclusions, the pipeline's core goal is to **compute all five foundational connectivity metrics in a single pass and synthesize them as complementary lines of evidence**, classifying each electrode pair by what the metrics jointly indicate without asserting causal mechanisms that sensor-level data cannot independently resolve.

---

## 2. Proposed Solution

The prototype is organized as a 5-node Directed Acyclic Graph (DAG) built with LangGraph. It is intentionally scoped as an early prototype to validate control flow, tool execution, and failure handling before adding system complexity.

```
[User Request + Raw Data]
           │
           ▼
     1. SUPERVISOR (ReAct Agent: LLM + Tools)
           │
           ▼
     2. DATA PREPARATION (Deterministic Function)
           │
           ▼
     3. CONNECTIVITY ANALYSIS (Deterministic Function)
           │
           ▼
     4. EVALUATOR (Multimodal Single-Shot VLM)
           │
           ▼
     5. FINAL SYNTHESIS (Single-Shot Text LLM)
           │
           ▼
  [report_<run_id>.md]
```

![Multi-Agent EEG Connectivity Analysis Workflow](./pipeline_workflow_diagram.png)
*Commit-Before-View · Deterministic Computation · Independent Evaluation*

### Pipeline Nodes in Order:
1. **Supervisor** (*ReAct Agent: LLM + Tools*): Resolves the user's natural-language request against dataset metadata, performs metric canonicalization via inline lookup, and compiles the complete **Gate 1 Pre-Flight Checklist** — scientific axes, metric selection, reference scheme, quality-gate thresholds, coupling thresholds, and artifact-detection thresholds — presenting one screen to the human before any computation starts. Values resolved with high confidence are pre-filled; values resolved with low confidence are flagged inline as **NEEDS YOUR INPUT**.
2. **Data Preparation** (*Deterministic Function*): Applies the Gate-1-confirmed reference scheme and filter parameters, detects and drops bad channels (with before/after plots), enforces the epoch-duration cycle check, bandpass-filters data, and extracts condition epochs without modifying phase dynamics.
3. **Connectivity Analysis** (*Deterministic Function*): Computes all five connectivity metrics in a single spectral pass via MNE-Connectivity, transforms imaginary coherence to its absolute magnitude ($|\text{imcoh}|$), executes cross-metric complementary evidence synthesis (classifying each channel pair into four deterministic evidence categories), writes intermediate CSV artifacts, renders multi-metric figures, and calculates summary metrics.
4. **Evaluator** (*Deterministic + Single-Shot VLM*): Runs deterministic mathematical sanity checks across all computed matrices; runs deterministic spatial/statistical artifact detection (ocular, muscle, electrode pop) against named threshold constants; then runs a single VLM call restricted to Top-5 unresolved zero-lag pair topology reads. Always completes a full pass — no mid-run interrupts. Each check carries a `confidence: float` score reviewed at Gate 2.
5. **Final Synthesis** (*Single-Shot Text LLM*): After **Gate 2 Pre-Publication Review** (human reviews complete `EvaluationVerdict` in one pass; fast rubber-stamp if clean PASS, otherwise resolves flagged items here), compiles the final self-contained Markdown report with embedded Cross-Metric Evidence Summary table, methodological caveats, and embedded base64 figures, performing zero re-analysis.

---

## 3. Fixed Constraints

The system operates under strict, immutable engineering constraints. None of the multi-metric expansions alter or relax these foundational rules:

1. **No ICA Anywhere in the Pipeline:**
   * *Reason:* Independent Component Analysis decomposes multi-channel data linearly and alters phase relationships across sensors, directly undermining the validity of phase-synchrony metrics (PLI, wPLI, ImCoh).
2. **No Amplitude-Based Epoch Rejection or Signal Cleaning Upstream:**
   * *Reason:* Dropping trials based on amplitude thresholds or applying signal modifications alters the underlying phase distribution. Residual artifacts (ocular blinks, muscle activity, channel pop) are left untouched in the signal and are instead visually detected and flagged downstream by the Evaluator.
3. **CSV is Produced at Exactly Two Pipeline Stages — Connectivity Analysis and Evaluator:**
   * *Reason:* Connectivity Analysis generates pairwise numerical matrices and evidence summary tables required for archival auditability. The Evaluator generates a per-check findings table (`evaluation_summary_<run_id>.csv`) required for cross-run tracking of quality metrics outside the MD report. CSV is never produced at Supervisor, Data Preparation, or Final Synthesis, and is never the final human-facing deliverable.
4. **Final Deliverable is a Single, Self-Contained Markdown (`.md`) File:**
   * *Reason:* The report must be portable and renderable client-side (e.g., in a browser or markdown viewer); all visual figures must be embedded directly as Base64 data URIs rather than local file path links.
5. **Reference Scheme is Discovered by Data Preparation and Proposed as a Default — Not Silently Applied:**
   * *Reason:* Reference is an intrinsic dataset property; the pipeline discovers it from the dataset header and proposes Common Average Reference (CAR) as the default if the recording is unreferenced. This proposed value is surfaced to the human as part of the Gate A manifest (Section 3.2) and requires explicit approval before Data Preparation applies it. The user is not asked to *derive* or *invent* the reference value — but they always see and approve what the pipeline will use.
6. **Explicit Plan Locking Governance:**
   * *Reason:* User-negotiated scientific variables are strictly locked prior to compute. The structural representation of metric execution within the locked plan is addressed in Section 10 as an explicit open decision.
7. **Universal Human-in-the-Loop Confirmation (No Silent Scientific Values):**
   * *Reason:* Every parameter that affects scientific conclusions — the three scientific axes, metric selection, reference scheme, quality-gate thresholds, coupling-classification thresholds, and artifact-detection thresholds — is surfaced to the human as an explicit, editable value in **Gate 1** and requires affirmative sign-off before computation starts. Purely cosmetic constants (colormap, figure DPI, file-naming convention) are hardcoded named constants documented in Appendix A and are never shown to the human for approval. See Section 3.2 for the full 2-gate mechanism.

### 3.1 Scope of the "Never Guesses" Principle

The system-wide design tenet that the pipeline "never guesses" applies to **every** parameter governing analysis execution — not only the scientific axes. Two categories exist, but both now require explicit human confirmation rather than silent application:

1. **Axes that Always Require Explicit Resolution Every Run (`freq_band`, `channels`, `condition`):**
   These three parameters represent genuine, unguessable scientific hypotheses where the researcher has specific experimental intent. The Supervisor will **never hallucinate, assume, or default them** without explicit user input or deterministic validation against dataset metadata. If any of these three axes is omitted or ambiguous in the prompt, execution immediately pauses for user clarification.
2. **Axis with a Proposed Default Battery (`metrics`):**
   `metrics` carries a principled, full-battery default: computing all five foundational metrics to enable complementary cross-metric evidence synthesis. That default is *proposed*, not silently applied — the human sees it as part of the confirmation manifest (Section 3.2) and can accept it or explicitly restrict it before the plan locks.

**Scientific engineering defaults are not silent.** Reference scheme, epoch boundaries and baseline handling, quality-gate thresholds, coupling-classification thresholds, and artifact-detection thresholds are pre-populated with sensible defaults but are always shown and editable in Gate 1. Purely cosmetic constants (colormap, DPI, file naming) are hardcoded — not shown for approval. See Section 3.2.

### 3.2 Human Confirmation Mechanism — Two Gates

The pipeline uses exactly **two** human checkpoints. Both are synchronous CLI prompts; neither requires a graphical UI.

#### Gate 1 — Pre-Flight Checklist (before any computation)

The Supervisor compiles a single screen listing every value the run will use that affects scientific conclusions. Rows are grouped by `risk_tier` and visually marked accordingly:

| Row Category | Example Parameters | Risk Tier | Default Behavior |
| :--- | :--- | :--- | :--- |
| Scientific Axes (must always be confirmed) | `freq_band`, `channels`, `condition` | `low` | No default; user must supply or confirm |
| Metric Selection | `metrics` (default: all five) | `low` | Pre-filled; editable |
| Reference Scheme | `reference` (default: CAR) | **`elevated`** | Pre-filled; editable |
| Quality-Gate Thresholds | `bad_channel_variance_threshold`, `min_cycles` | **`elevated`** | Pre-filled; editable |
| Coupling Thresholds | `τ_phase` (0.20), `τ_zerolag` (0.35) | **`elevated`** | Pre-filled; editable |
| Artifact-Detection Thresholds | `ocular_variance_ratio`, `muscle_power_threshold`, `electrode_pop_sigma` | **`elevated`** | Pre-filled; editable |

**Two orthogonal flags per row — both always visible:**
- `⚠ NEEDS YOUR INPUT` — epistemic uncertainty flag: the Supervisor's resolution confidence fell below threshold (fuzzy channel match, ambiguous annotation). Signals *"I wasn't sure."* Not set for deterministically assigned defaults.
- **`[ELEVATED RISK]`** — domain-risk flag: this parameter's value changes scientific classification outcomes regardless of how confidently it was resolved. A row can be `needs_human_input = False` AND `risk_tier = "elevated"` simultaneously — e.g., the Supervisor correctly read CAR from the header (high confidence), but reference scheme is still `elevated` because a wrong reference silently alters every downstream metric. The two flags are independent signals; neither implies the other.

No computation downstream of the Supervisor starts until the human submits Gate 1 (accepting values as-is or with overrides). The submitted values are written to `GraphState` as `preflight_confirmed = True` and `parameter_manifest` (the finalized manifest).

**Cosmetic constants are excluded from Gate 1** — colormap, figure DPI, and file-naming convention are hardcoded named constants (see Appendix A) and are never shown for human approval.

#### Gate 2 — Pre-Publication Review (before Final Synthesis)

After the Evaluator completes its full single-shot pass, the human reviews the complete `EvaluationVerdict` in one place, presented in two distinct subsections:

**Subsection A — Deterministic checks** (sanity assertions and artifact heuristics):
- Overall verdict badge (`PASS` / `WARNING` / `FAIL`).
- Per-metric sanity check results with `confidence` scores.
- Deterministic artifact-detection results (ocular, muscle, electrode pop) with `confidence` scores.

**Subsection B — Subjective visual reads** (rendered separately, labeled explicitly):
- Top-5 unresolved zero-lag pair topology reads from the VLM with `confidence` scores.
- This subsection is always rendered separately and labeled **"Subjective VLM Topology Reads — Visual Judgment Only"** to distinguish it from the deterministic rows above. These reads are the pipeline's only non-deterministic evaluation output; they must not be scanned past alongside routine PASS/FAIL sanity rows.

If everything is `PASS` and all confidence scores are above threshold, Gate 2 is a fast rubber-stamp. If any result is `WARNING` / `FAIL` or below the confidence threshold, the human resolves those items here before Final Synthesis proceeds.

**The Evaluator never interrupts mid-run.** It always completes its full single-shot pass; all uncertainty is deferred to Gate 2 via the `confidence` field on each check, not via a mid-run interrupt mechanism.

---

## 4. Tech Stack & The Five Connectivity Metrics

### 4.1 Orchestration & Modeling
- **Orchestration Framework:** LangChain & LangGraph.
- **Agent Architecture:** 
  - **ReAct Pattern:** Used **exclusively** in the Supervisor node. The Supervisor is the only node that interacts with external tools (querying file headers, event lists, ontology tables) and iterates over user input.
  - **Why Other Nodes are NOT ReAct Agents:**
    - *Data Preparation & Connectivity Analysis* are strictly deterministic signal-processing pipelines; introducing an LLM loop adds non-determinism, failure points, and latency without benefit.
    - *Evaluator* is a single-shot multimodal call (images + numbers in, structured verdict out); it requires no external tool execution mid-reasoning.
    - *Final Synthesis* is a single-shot formatting call; all required data is already present in state, eliminating the need for tool access.
- **Signal Processing & Connectivity:** MNE-Python (I/O, filtering, epoching) and MNE-Connectivity (spectral and phase connectivity metrics).
- **Experiment Tracking:** MLflow. Every pipeline run opens exactly one MLflow run, tagged with the same `run_id` used across all filesystem artifacts. MLflow logging is a **passive side effect** attached to each existing node via a thin centralized wrapper that reads `GraphState` after each node returns — the deterministic signal-processing code in Data Preparation and Connectivity Analysis remains free of any MLflow import. Logging failures are caught and emitted as warnings; they never fail or block a scientific run.
- **Vision Model:** Multimodal-capable Vision-Language Model (VLM) for the Evaluator node. VLM scope is **strictly and exclusively** the Top-5 unresolved zero-lag pair topology reads; no other check in the pipeline uses the VLM. Artifact detection (ocular, muscle, electrode pop) is performed deterministically by NumPy/SciPy heuristics, not the VLM.

### 4.2 The Five Connectivity Metrics: Absolute ImCoh & Complementary Synthesis

The pipeline incorporates five standard electrode-level connectivity metrics, divided into two theoretically distinct classes based on their treatment of zero-phase-lag interactions:

| Metric | Canonical ID | Mathematical Property | Zero-Lag Sensitivity | Primary Vulnerability / Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **Phase Lag Index** | `pli` | Asymmetry of phase difference distribution | Discards zero-lag ($\Delta\theta = 0, \pi$) | Invariant to volume conduction; sensitive to noise around zero lag |
| **Weighted Phase Lag Index** | `wpli` | Magnitude-weighted phase difference sign | Discards zero-lag (continuous weighting) | Phase-robust; suppresses small-angle noise around the real axis |
| **Absolute Imaginary Coherence** | `imcoh` | Absolute imaginary cross-spectrum ($|\text{Im}(S_{xy})|$) | Discards zero-lag ($\text{Im}(S_{xy})$ vanishes) | Phase-robust; isolates non-instantaneous spectral coupling; non-negative & symmetric |
| **Phase Locking Value** | `plv` | Absolute mean phase difference consistency | Includes zero-lag fully | Detects instantaneous and lagged coupling; highly sensitive to volume conduction |
| **Spectral Coherence** | `coh` | Normalized cross-spectral density magnitude | Includes zero-lag fully | Captures linear synchronization; conflates volume conduction with true coupling |

#### Decision on Imaginary Coherence: Absolute Value Transformation ($|\text{imcoh}|$) Everywhere
MNE-Connectivity computes signed imaginary coherence in $[-1, 1]$ by default, where the sign denotes lead/lag phase relationships between channel pairs. Because this pipeline performs **undirected, electrode-level functional connectivity**, the pipeline explicitly adopts **Option A**:
- Imaginary coherence is stored as its **absolute magnitude ($|\text{imcoh}|$)** across all pipeline stages (CSV exports, mathematical evaluations, and visual heatmaps).
- This ensures $|\text{imcoh}|$ is non-negative, bounded in $[0, 1]$, and mathematically symmetric ($M = M^T$), identical to PLI, wPLI, PLV, and Coherence.
- Crucially, this prevents signed values from destructively canceling during phase-robust group averaging ($S_{\text{phase}} = \text{mean}(\text{PLI}, \text{wPLI}, |\text{imcoh}|)$).
- Downstream sanity bounds check for $[0, 1]$, matrix symmetry checks verify $M = M^T$, and heatmap visualizations render on a uniform $[0, 1]$ sequential colormap.

#### Symmetric Blind Spots and the Need for Synthesis:
1. **The Phase-Robust Group (`pli`, `wpli`, `imcoh`):** Deliberately discards or suppresses interactions with zero or $\pi$ phase lag. By construction, volume conduction through the head occurs quasi-instantaneously (speed of light at biological scales), producing zero-lag correlations across sensors. Phase-robust metrics eliminate these spurious correlations. **Their blind spot:** They are completely blind to genuine near-zero-lag neural synchronization (e.g., synchronous inter-hemispheric gamma rhythms mediated by direct callosal projection).
2. **The Zero-Lag-Inclusive Group (`plv`, `coh`):** Measures phase consistency and cross-spectral coherence without penalizing zero-lag alignment. This enables them to capture instantaneous and near-zero-lag interactions. **Their blind spot:** They cannot distinguish whether zero-lag coherence reflects genuine neural synchronization or physical volume conduction from a single dipole source.
3. **Symmetric Inability to Resolve Divergence:**
   Neither group can resolve which explanation is correct on its own. When the two groups diverge on a channel pair (zero-lag is strong, phase-robust is weak), the data indicates elevated zero-lag coherence, but **cannot resolve** whether this is genuine neural coupling or volume conduction. Labeling such pairs as definitively "suspect" or "artifacts" overclaims what sensor-level metrics can prove.

Instead, the system employs a **complementary evidence synthesis**, classifying channel pairs into four evidence profiles based on what the metrics jointly provide:
- **`FULLY CONVERGENT`**: Both phase-robust and zero-lag-inclusive groups agree on strong coupling above heuristic thresholds. This provides the highest comparative confidence, supported independently by both estimators.
- **`CONFIRMED COUPLING`**: Phase-robust group shows strong coupling above the heuristic threshold, while zero-lag-inclusive metrics do not. Because zero-lag interaction is excluded by design, this coupling cannot be explained by volume conduction. It represents **phase-lagged coupling not explainable by volume conduction, though heuristic and not yet tested against a statistical null distribution**.
- **`ZERO-LAG COUPLING, CAUSE UNRESOLVED`**: Zero-lag-inclusive group shows strong coupling, but the phase-robust group does not. The pair exhibits robust zero-lag synchronization, but sensor-level metrics cannot determine whether the cause is genuine near-zero-lag neural communication or volume conduction. The pipeline explicitly reports this as unresolved rather than guessing.
- **`NO COUPLING`**: All metrics indicate weak or baseline interaction.

### 4.3 Single-Pass Computation Confirmation in MNE-Connectivity
All five metrics are natively supported in MNE-Connectivity and are computed within a **single execution pass** of `mne_connectivity.spectral_connectivity_epochs`. By supplying the full method list:
```python
methods = ['pli', 'wpli', 'imcoh', 'plv', 'coh']
con = spectral_connectivity_epochs(
    epochs,
    method=methods,
    indices=None,          # All pairwise channel combinations
    mode='multitaper',     # or 'fourier'
    sfreq=epochs.info['sfreq'],
    fmin=plan.freq_band.fmin,
    fmax=plan.freq_band.fmax,
    faverage=True
)
```
Following computation, imaginary coherence is converted to absolute values ($|\text{imcoh}|$) across all pairs to establish mathematical parity ($[0, 1]$ range, matrix symmetry) with the other four metrics. This architecture evaluates all five connectivity estimators over the identical spectral representation without redundant transforms.

---

## 5. Schemas

All schemas enforce strict JSON-serializability. In-memory data arrays (e.g., MNE `Raw`, `Epochs`, or `SpectralConnectivity` objects) are never stored in state; only file paths and primitive data structures are passed between nodes.

### 5.1 Typed Models for Evidence Synthesis & Verdict

```python
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class MetricEnum(str, Enum):
    PLI = "pli"
    WPLI = "wpli"
    IMAGINARY_COHERENCE = "imcoh"
    PLV = "plv"
    COHERENCE = "coh"

class EvidenceCategoryEnum(str, Enum):
    FULLY_CONVERGENT = "FULLY CONVERGENT"
    CONFIRMED_COUPLING = "CONFIRMED COUPLING"
    ZERO_LAG_UNRESOLVED = "ZERO-LAG COUPLING, CAUSE UNRESOLVED"
    NO_COUPLING = "NO COUPLING"

class VisualTopologyReadEnum(str, Enum):
    PLAUSIBLE_NEURAL_TOPOLOGY = "plausible_neural_topology"
    CONSISTENT_WITH_ARTIFACT = "consistent_with_artifact"
    INDETERMINATE = "indeterminate"

class PairEvidenceSynthesis(BaseModel):
    channel_pair: List[str]            # e.g., ["F3", "F4"]
    phase_robust_mean: float           # Mean across computed phase-robust metrics (PLI, wPLI, |ImCoh|)
    zero_lag_mean: float               # Mean across computed zero-lag-inclusive metrics (PLV, Coherence)
    classification: EvidenceCategoryEnum # Deterministic 4-category classification from Node 3
    evaluator_visual_read: Optional[VisualTopologyReadEnum] = None # Supporting visual observation for top-N unresolved pairs
    evaluator_visual_note: Optional[str] = None # Supporting visual context (supporting note, NOT asserting a cause)

class ArtifactDetectionResult(BaseModel):
    """Result of a single deterministic artifact heuristic check."""
    detected: bool                     # True if the artifact pattern was found
    affected_channels: List[str]       # Channels implicated (empty if not detected)
    confidence: float                  # 0.0–1.0; how clearly the heuristic signature was met
    detail: str                        # Human-readable description of measured values vs. threshold

class ArtifactSignatures(BaseModel):
    """Deterministic spatial/statistical artifact detection results."""
    ocular_contamination: ArtifactDetectionResult   # Elevated variance/PLV at Fp1/Fp2/AF vs. posterior
    muscle_contamination: ArtifactDetectionResult   # Broadband power elevation >40 Hz at temporal/occipital
    electrode_pop: ArtifactDetectionResult          # Single-channel variance outlier >3σ, continuous
    top_unresolved_visual_reads: List[PairEvidenceSynthesis]  # VLM topology reads, Top-5 only

class SanityCheckFlags(BaseModel):
    values_in_bounds: bool             # All values within [0, 1] (holds for |imcoh|)
    matrix_is_symmetric: bool          # M = M^T verified for all non-directional metrics
    is_sparse_or_flat: bool            # Flags SNR collapse (all near 0 or all near 1)
    confidence: float                  # 0.0–1.0; confidence in this metric's sanity assessment

class ParameterManifestEntry(BaseModel):
    name: str                          # e.g., "reference_scheme", "tau_phase", "ocular_variance_ratio"
    category: str                      # "scientific_axis" | "metric_selection" | "engineering_threshold"
    proposed_value: str                # Pre-filled default, shown to the human
    confidence: Optional[float] = None # Supervisor resolution confidence (None = not LLM-resolved)
    needs_human_input: bool = False    # True if confidence below threshold; shown as ⚠ NEEDS YOUR INPUT
    risk_tier: str = "low"             # "low" | "elevated" — static per parameter name, set by Supervisor at manifest compile time
                                       # "elevated": reference, τ_phase, τ_zerolag, ocular_variance_ratio,
                                       #             muscle_power_threshold, electrode_pop_sigma,
                                       #             bad_channel_variance_threshold, min_cycles
                                       # "low":      freq_band, channels, condition, metrics
                                       # NOTE: orthogonal to needs_human_input — a row can be high-confidence AND elevated-risk
    human_approved_value: Optional[str] = None  # Set once Gate 1 sign-off is given

class OverallVerdictEnum(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"

class EvaluationVerdict(BaseModel):
    overall_verdict: OverallVerdictEnum
    per_metric_sanity: Dict[str, SanityCheckFlags] # Sanity flags per computed metric, with confidence
    artifacts: ArtifactSignatures                  # Deterministic artifact results + VLM topology reads
    visual_observations: str                       # Narrative summary across all checks
    recommendations_for_user: List[str]            # Actionable scientific caveats
```

### 5.2 Shared `GraphState`
```python
from typing import TypedDict, Optional, Dict, List

class GraphState(TypedDict):
    # Input & Dataset Reference
    raw_data_path: str
    user_request: str
    
    # 1. Supervisor Outputs
    clarification_question: Optional[str]
    plan: Optional[AnalysisPlan]
    parameter_manifest: Optional[List[ParameterManifestEntry]]  # Gate 1: full checklist incl. all scientific thresholds
    preflight_confirmed: bool                                   # Gate 1 sign-off flag; blocks Data Prep until True
    
    # 2. Data Preparation Outputs
    bad_channels_dropped: Optional[List[str]]        # Labels of channels dropped for near-zero variance
    channel_plot_paths: Optional[Dict[str, str]]     # "before" -> PNG path, "after" -> PNG path
    preprocessed_data_path: Optional[str]
    data_prep_error: Optional[str]
    
    # 3. Connectivity Analysis Outputs
    metric_csv_paths: Optional[Dict[str, str]]       # Map of metric -> intermediate CSV path, plus "evidence_summary" -> CSV path
    heatmap_image_paths: Optional[Dict[str, str]]    # Map of metric -> rendered heatmap PNG path
    network_image_paths: Optional[Dict[str, str]]    # Map of metric -> rendered network graph PNG path
    numerical_summaries: Optional[Dict[str, Dict[str, float]]] # Map of metric -> summary statistics dict
    evidence_summary: Optional[List[PairEvidenceSynthesis]] # Strongly typed full list of pairwise classifications
    
    # 4. Evaluator Outputs
    evaluation_verdict: Optional[EvaluationVerdict]
    evaluation_summary_csv_path: Optional[str]       # Path to evaluation_summary_<run_id>.csv
    
    # 5. Final Synthesis Outputs
    final_report_path: Optional[str]
```

### 5.3 `AnalysisPlan` Schema & Metric Configuration

```python
class FrequencyBand(BaseModel):
    name: str              # e.g., "alpha", "theta", "custom"
    fmin: float = Field(..., gt=0.0)
    fmax: float = Field(..., gt=0.0)

class AnalysisPlan(BaseModel):
    metrics: List[MetricEnum] = Field(
        default_factory=lambda: [
            MetricEnum.PLI,
            MetricEnum.WPLI,
            MetricEnum.IMAGINARY_COHERENCE,
            MetricEnum.PLV,
            MetricEnum.COHERENCE
        ],
        description="List of metrics to compute. Defaults to all five foundational metrics to enable complementary evidence synthesis."
    )
    freq_band: FrequencyBand
    channels: List[str]    # Must contain >= 2 canonical channel names
    condition: str         # Must match an existing dataset annotation/event
```

*(Note: See Section 10 for the open architectural decision regarding whether `metrics` should be a locked axis or completely dropped from user configuration).*

---

## 6. Agent-Level Specifications

```
  +-------------------------------------------------------------------------+
  |                              1. SUPERVISOR                              |
  |                           (ReAct LLM + Tools)                           |
  | Reads: user_request, raw_data_path                                      |
  | Writes: plan (AnalysisPlan), clarification_question                     |
  +-------------------------------------------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                           2. DATA PREPARATION                           |
  |                        (Deterministic Function)                         |
  | Reads: raw_data_path, plan                                              |
  | Writes: bad_channels_dropped, channel_plot_paths,                       |
  |         preprocessed_data_path, data_prep_error                         |
  | Performs: CAR reference, Bad Channel Detection & Drop, before/after     |
  |           channel plots, Epoch Frequency Gate, bandpass filter,         |
  |           epoching (NO ICA / NO amplitude rejection)                    |
  +-------------------------------------------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                        3. CONNECTIVITY ANALYSIS                         |
  |                        (Deterministic Function)                         |
  | Reads: preprocessed_data_path, plan                                     |
  | Writes: metric_csv_paths, heatmap_image_paths, network_image_paths,     |
  |         numerical_summaries, evidence_summary                           |
  | Performs: 5-metric single-pass, |imcoh| transform, heuristic synthesis  |
  +-------------------------------------------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                              4. EVALUATOR                               |
  |                       (Multimodal Single-Shot VLM)                      |
  | Reads: heatmap_image_paths, network_image_paths, numerical_summaries,   |
  |        evidence_summary, plan                                           |
  | Writes: evaluation_verdict (EvaluationVerdict)                          |
  | Inspects: Sanity bounds, artifacts, Top-5 unresolved visual reads       |
  +-------------------------------------------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  |                           5. FINAL SYNTHESIS                            |
  |                         (Single-Shot Text LLM)                          |
  | Reads: plan, numerical_summaries, evidence_summary,                     |
  |        evaluation_verdict, image paths                                  |
  | Writes: final_report_path (Self-contained report_<run_id>.md)           |
  | Note: Pure compilation, zero re-analysis of data or images              |
  +-------------------------------------------------------------------------+
```

### 6.1 Supervisor Node
* **Mechanism:** ReAct Agent (LLM with iterative tool access).
* **State Reads:** `user_request`, `raw_data_path`.
* **State Writes:** `plan`, `parameter_manifest`, `preflight_confirmed` (or `clarification_question` if a scientific axis is missing).
* **Internal Logic & Tool Set:**
  1. **Tool `get_dataset_info(data_path: str)`:**
     - Reads header to retrieve sampling frequency (`sfreq`), Nyquist limit (`sfreq / 2`), duration, and available channel labels. Always called first.
  2. **Tool `get_dataset_conditions(data_path: str)`:**
     - Inspects trigger channels/annotations to retrieve valid condition labels and trial counts.
  3. **Tool `resolve_frequency_band(band_name_or_range: str, sfreq: float)`** *(fuzzy resolution, confidence-scored)*:
     - Maps band names or custom ranges to explicit `fmin` and `fmax` against lookup table (Delta: 1.0–4.0 Hz, Theta: 4.0–8.0 Hz, Alpha: 8.0–12.0 Hz, Beta: 13.0–30.0 Hz, Gamma: 30.0–45.0 Hz). Rejects any band where $f_{\max} \ge \text{Nyquist}$.
     - Returns a `confidence` score. Scores below the tool's threshold produce a `needs_human_input = True` manifest entry.
  4. **Tool `resolve_channel_selection(requested_channels_or_region: str, available_channels: list[str])`** *(fuzzy resolution, confidence-scored)*:
     - Executes 4-stage channel normalization (clean → exact match → 10-20 alias match → region mapping).
     - Returns a `confidence` score per resolved label. Any label below threshold produces a `needs_human_input = True` manifest entry.
  5. **Metric Canonicalization — Inline Logic (not a tool call):**
     - Metric names supplied by the user are normalized by the Supervisor directly against the following hardcoded 5-item lookup table. No tool call or confidence score is involved — this is a deterministic string lookup:
       | Input String Variation (Case-Insensitive) | Canonical ID | Theoretical Group |
       | :--- | :--- | :--- |
       | `"pli"`, `"phase lag index"`, `"phase-lag index"` | `"pli"` | Phase-Robust |
       | `"wpli"`, `"weighted phase lag index"`, `"weighted pli"` | `"wpli"` | Phase-Robust |
       | `"imcoh"`, `"imaginary coherence"`, `"icoh"`, `"imaginary part of coherence"` | `"imcoh"` | Phase-Robust |
       | `"plv"`, `"phase locking value"`, `"phase lock"` | `"plv"` | Zero-Lag-Inclusive |
       | `"coh"`, `"coherence"`, `"spectral coherence"` | `"coh"` | Zero-Lag-Inclusive |
     - Any string not matching any row returns `UNSUPPORTED_METRIC` and the Supervisor asks the user to clarify before building the manifest.
     - If the user specifies no metrics, the manifest entry is pre-filled with all five as the default battery.
  6. **Gate 1 — Pre-Flight Checklist (`compile_and_confirm_manifest`):**
     - Validates structural invariants: $\text{channel count} \ge 2$, all 3 hypothesis axes populated, $f_{\min} < f_{\max}$, metric list non-empty.
     - Compiles the **full Gate 1 manifest**: the three scientific axes, metric selection, reference scheme, the two Data-Prep quality-gate thresholds, the two coupling-classification thresholds ($\tau_{\text{phase}}, \tau_{\text{zerolag}}$), and the three artifact-detection thresholds (`ocular_variance_ratio`, `muscle_power_threshold`, `electrode_pop_sigma`) — all pre-filled with defaults. Low-confidence entries are flagged `⚠ NEEDS YOUR INPUT` inline.
     - **Risk-tier assignment (static, no LLM judgment):** The Supervisor sets `risk_tier = "elevated"` on the following parameters at manifest compile time: `reference`, `tau_phase`, `tau_zerolag`, `ocular_variance_ratio`, `muscle_power_threshold`, `electrode_pop_sigma`, `bad_channel_variance_threshold`, `min_cycles`. All other parameters (`freq_band`, `channels`, `condition`, `metrics`) receive `risk_tier = "low"`. This assignment is a hardcoded lookup — the same parameter name always gets the same tier, regardless of how it was resolved or what `confidence` value was produced.
     - Presents one screen to the human, rows grouped by `risk_tier`. Blocks all downstream nodes until the human submits. Sets `preflight_confirmed = True` on sign-off.

---

### 6.2 Data Preparation Node
* **Mechanism:** Pure Deterministic Function.
* **State Reads:** `raw_data_path`, `plan`.
* **State Writes:** `bad_channels_dropped`, `channel_plot_paths`, `preprocessed_data_path`, `data_prep_error`.
* **Internal Logic & Signal Quality Gates:**
  1. Loads raw recording via MNE.
  2. Discovers hardware reference. Applies Common Average Reference (CAR) by default.
  3. **Quality Gate 1 — Bad Channel Detection, Visualization & Drop:**
     - Computes temporal variance $\sigma_c^2 = \text{Var}(x_c(t))$ for every channel $c \in \text{plan.channels}$.
     - Identifies any channel with near-zero variance ($\sigma_c^2 < 10^{-15} \text{ V}^2$), indicating a disconnected, flatline, or dead sensor.
     - **Before Plot:** Renders a raw channel time-series plot (`outputs/channels_before_<run_id>.png`) showing *all* channels, with detected bad channels visually highlighted (e.g., distinct color). This plot is generated *before* any channel is removed.
     - **Channel Drop:** Removes all identified bad channels from the active channel set using `raw.drop_channels(bad_labels)`. The dropped labels are written to `bad_channels_dropped` in `GraphState`.
     - **After Plot:** Renders a second channel time-series plot (`outputs/channels_after_<run_id>.png`) showing only the retained channels after the drop.
     - Both PNG paths are written to `channel_plot_paths = {"before": ..., "after": ...}` in `GraphState`.
     - If the drop leaves fewer than 2 channels in `plan.channels`, Data Prep halts with:
       `data_prep_error = "INSUFFICIENT_CHANNELS: After dropping bad channels <labels>, fewer than 2 channels remain. Connectivity analysis requires >= 2 channels."`
     - **No ICA, No Interpolation:** Dropped channels are excluded from further processing without interpolation. Phase dynamics of retained channels are not altered.
  4. **Quality Gate 2 — Epoch Duration vs. Lowest Frequency Check ($f_{\min}$ Cycle Check):**
     - Inspects the duration $T_{\text{epoch}}$ of epochs defined by the locked `condition` events.
     - Enforces that epoch length accommodates at least $k_{\min} = 3$ full oscillation cycles of the band's lowest frequency $f_{\min}$:
       $$T_{\text{epoch}} \ge \frac{3.0}{f_{\min}}$$
     - (e.g., for Delta at $f_{\min} = 1.0\text{ Hz}$, requires $T_{\text{epoch}} \ge 3.0\text{ s}$; for Alpha at $f_{\min} = 8.0\text{ Hz}$, requires $T_{\text{epoch}} \ge 0.375\text{ s}$).
     - If $T_{\text{epoch}} < \frac{3.0}{f_{\min}}$, Data Prep halts immediately with:
       `data_prep_error = "INSUFFICIENT_EPOCH_LENGTH: Epoch duration <T>s provides fewer than 3 full cycles of lowest frequency fmin=<fmin>Hz. Reliable phase estimation impossible."`
  5. Applies zero-phase bandpass filter targeting the plan's exact `[fmin, fmax]` range.
  6. Slices continuous recording into epochs matching the locked `condition` events.
  7. **Explicitly Excluded:** No ICA decomposition, no component pruning, no peak-to-peak amplitude thresholding, no trial dropping, and no channel interpolation.
  8. Saves preprocessed epochs to a temporary scratch `.fif` file and writes path to state.

---

### 6.3 Connectivity Analysis Node
* **Mechanism:** Pure Deterministic Function.
* **State Reads:** `preprocessed_data_path`, `plan`.
* **State Writes:** `metric_csv_paths`, `heatmap_image_paths`, `network_image_paths`, `numerical_summaries`, `evidence_summary`.
* **Internal Logic:**
  1. Loads preprocessed epochs.
  2. Executes `mne_connectivity.spectral_connectivity_epochs` in a **single pass** for all metrics specified in `plan.metrics` (defaulting to all five: `['pli', 'wpli', 'imcoh', 'plv', 'coh']`).
  3. **Absolute Value Transformation of Imaginary Coherence:**
     - Following computation, imaginary coherence is transformed to its absolute magnitude ($|\text{imcoh}|$) across all electrode pairs, yielding a symmetric, $[0, 1]$-bounded matrix that aligns mathematically with PLI and wPLI.
  4. Extracts the symmetric $N \times N$ connectivity matrices for each metric.
  5. **Multi-Metric Artifact Generation:**
     - *Intermediate CSVs:* Writes pairwise values and coordinates for each metric to `outputs/<metric>_<run_id>.csv`. Also writes a consolidated evidence summary CSV `outputs/evidence_summary_<run_id>.csv` containing per-pair values across all metrics and their deterministic classifications.
     - *Heatmap Plots:* Renders individual seaborn/matplotlib heatmaps bounded $[0, 1]$ with sequential colormaps and explicit channel labels for each metric, saved to `outputs/heatmap_<metric>_<run_id>.png`.
     - *Network Graphs:* Renders 2D circular/spring graphs showing thresholded edges for each metric, saved to `outputs/network_<metric>_<run_id>.png`.
  6. **Per-Metric Summary Calculations:** Computes mean, std, median, min, max connectivity, and matrix sparsity for each metric.
  7. **Deterministic Evidence Synthesis Classification (Fixed Heuristic Thresholds):**
     - **Where Computed:** Computed strictly and deterministically in Node 3 (Connectivity Analysis), ensuring identical mathematical evaluation across all runs without reliance on VLM judgment.
     - **Fixed Coupling Thresholds:**
       Because phase-robust metrics discard the real-axis component of the cross-spectral density, their theoretical and empirical dynamic range is lower than that of zero-lag metrics (PLV and Coherence). To account for this difference, the system uses two fixed, deterministic thresholds:
       - **Phase-Robust Group Threshold:** $\tau_{\text{phase}} = 0.20$
       - **Zero-Lag-Inclusive Group Threshold:** $\tau_{\text{zerolag}} = 0.35$
     - **Methodological Status of Thresholds:**
       The fixed thresholds ($\tau_{\text{phase}} = 0.20$ and $\tau_{\text{zerolag}} = 0.35$) are operational heuristics rather than derived statistical significance cutoffs. Phase-synchrony metrics exhibit positive sample-size bias at low epoch counts. The current prototype does not compute permutation/surrogate null distributions, nor does it correct for multiple comparisons across the $\sim N(N-1)/2$ simultaneous pairwise tests (~2,016 pairs for a 64-channel montage). These thresholds serve strictly to stratify observed values into comparative qualitative profiles.
     - **Group Mean Calculations:**
       For each channel pair $(i, j)$:
       $$S_{\text{phase}}(i, j) = \frac{1}{|M_{\text{phase}}|}\sum_{m \in M_{\text{phase}}} C_m(i, j) \quad \text{where } M_{\text{phase}} \subseteq \{\text{pli}, \text{wpli}, |\text{imcoh}|\}$$
       $$S_{\text{zerolag}}(i, j) = \frac{1}{|M_{\text{zerolag}}|}\sum_{m \in M_{\text{zerolag}}} C_m(i, j) \quad \text{where } M_{\text{zerolag}} \subseteq \{\text{plv}, \text{coh}\}$$
     - **Deterministic 4-Category Decision Rule:**
       1. **`FULLY CONVERGENT`**:
          $$S_{\text{phase}}(i, j) \ge \tau_{\text{phase}} \quad \text{AND} \quad S_{\text{zerolag}}(i, j) \ge \tau_{\text{zerolag}}$$
          *Interpretation:* All metrics agree on strong coupling above heuristic thresholds. Supported independently by both phase-robust and zero-lag-inclusive estimators.
       2. **`CONFIRMED COUPLING`**:
          $$S_{\text{phase}}(i, j) \ge \tau_{\text{phase}} \quad \text{AND} \quad S_{\text{zerolag}}(i, j) < \tau_{\text{zerolag}}$$
          *Interpretation:* Phase-robust group shows strong coupling. Since zero-lag interactions are eliminated by design, this coupling cannot be explained by volume conduction. It represents phase-lagged coupling not explainable by volume conduction, though heuristic and not yet tested against a statistical null distribution.
       3. **`ZERO-LAG COUPLING, CAUSE UNRESOLVED`**:
          $$S_{\text{phase}}(i, j) < \tau_{\text{phase}} \quad \text{AND} \quad S_{\text{zerolag}}(i, j) \ge \tau_{\text{zerolag}}$$
          *Interpretation:* Zero-lag-inclusive metrics show strong coupling, but phase-robust metrics do not. The pair exhibits robust zero-lag synchronization, but sensor-level metrics cannot determine whether the cause is genuine near-zero-lag neural communication or volume conduction. The pipeline explicitly reports this as unresolved rather than guessing.
       4. **`NO COUPLING`**:
          $$S_{\text{phase}}(i, j) < \tau_{\text{phase}} \quad \text{AND} \quad S_{\text{zerolag}}(i, j) < \tau_{\text{zerolag}}$$
          *Interpretation:* Weak or baseline interaction across all metrics.
     - Compiles the classifications and per-pair values into `evidence_summary` in `GraphState`.

---

### 6.4 Evaluator Node
* **Mechanism:** Deterministic Python (steps 1–2) + Single-Shot VLM (step 3) + CSV Serialization (step 4). No mid-run interrupt of any kind.
* **State Reads:** `plan`, `parameter_manifest` (for artifact threshold values), `heatmap_image_paths`, `network_image_paths`, `numerical_summaries`, `evidence_summary`.
* **State Writes:** `evaluation_verdict` (`EvaluationVerdict`), `evaluation_summary_csv_path`.
* **Internal Logic — Four Sequential Steps, Always Completed in Full:**
  1. **Deterministic Sanity Checks (NumPy):**
     - Evaluates mathematical bounds $[0, 1]$, zeroed diagonal, and symmetry ($M = M^T$) across *each* computed metric matrix.
     - Each check populates a `SanityCheckFlags` entry with a `confidence: float`. For deterministic math checks, confidence is either `1.0` (assertion passes) or `0.0` (assertion fails); intermediate values are never produced here.
  2. **Deterministic Artifact Detection (NumPy/SciPy — three independent heuristics):**
     - All threshold values are read from `parameter_manifest` (set at Gate 1; see named constants in Appendix A for defaults).
     - **Ocular Contamination:** Computes the ratio of mean channel variance at fronto-polar channels (Fp1, Fp2, AF3, AF4) to mean variance at posterior channels (O1, O2, P7, P8). If ratio exceeds `ocular_variance_ratio` threshold, detection fires. Also checks for elevated PLV between fronto-polar pairs above `τ_zerolag`. Populates `artifacts.ocular_contamination` with `detected`, `affected_channels`, `confidence`, and `detail`.
     - **Muscle Artifact:** Computes broadband power above 40 Hz at temporal and occipital electrodes relative to the full-band mean. If elevation exceeds `muscle_power_threshold`, detection fires. Populates `artifacts.muscle_contamination`.
     - **Electrode Pop:** For each channel in `plan.channels`, computes its variance and flags any channel where variance exceeds 3 × `electrode_pop_sigma` standard deviations above the cross-channel mean. Populates `artifacts.electrode_pop`.
     - Each `ArtifactDetectionResult` carries a `confidence` score reflecting how clearly the measured value exceeded (or fell short of) the threshold; a value just at the threshold boundary produces a lower confidence than one far above it.
  3. **VLM Topology Reads — Top-5 Unresolved Pairs Only:**
     - The VLM is invoked **once**, in a single call, strictly for the Top-5 unresolved zero-lag pairs ranked by `zero_lag_mean` strength.
     - The VLM examines the spatial layout of each pair on the rendered heatmaps and network graphs, noting whether the pattern is more consistent with a known artifact signature or a plausible neural network topology.
     - Output is recorded as `evaluator_visual_read` and `evaluator_visual_note` on each `PairEvidenceSynthesis` entry. This is a **supporting visual observation only** — it does NOT re-classify the pair; the deterministic `ZERO-LAG COUPLING, CAUSE UNRESOLVED` label remains locked.
     - Each topology read carries a `confidence: float`. All remaining unresolved pairs beyond the Top-5 retain their classification with no VLM commentary.
     - **Gate 2 rendering:** These reads are surfaced in a dedicated **Subsection B — Subjective VLM Topology Reads**, rendered separately from the deterministic Subsection A (sanity checks and artifact heuristics), and labeled explicitly as the pipeline's only non-deterministic evaluation output. This prevents them from being scanned past alongside routine PASS/FAIL rows.
  4. **CSV Serialization:** Flattens the complete `EvaluationVerdict` into `outputs/evaluation_summary_<run_id>.csv`, one row per check, with columns:

     | Column | Description |
     | :--- | :--- |
     | `run_id` | Shared run identifier across all artifacts |
     | `check_category` | `sanity` \| `artifact` \| `topology_read` |
     | `check_name` | e.g., `pli_values_in_bounds`, `ocular_contamination`, `Fp1-Fp2_topology` |
     | `result` | `pass` \| `fail` \| `detected` \| `clear` \| `plausible_neural` \| `consistent_with_artifact` \| `indeterminate` |
     | `confidence` | `float` in $[0, 1]$ |
     | `affected_channels` | Comma-separated channel labels, or empty |
     | `detail` | Human-readable description of the measured value vs. threshold |

     Writes the path to `evaluation_summary_csv_path` in `GraphState`. This is pure serialization of already-computed state; no new computation occurs.

---

### 6.5 Final Synthesis Node
* **Mechanism:** Single-Shot Text LLM.
* **State Reads:** `plan`, `numerical_summaries`, `evidence_summary`, `evaluation_verdict`, `heatmap_image_paths`, `network_image_paths`.
* **State Writes:** `final_report_path`.
* **Internal Logic:**
  - **Gate 2 — Pre-Publication Review:** Halts every run without exception — there is no silent auto-advance path. The human reviews the complete `EvaluationVerdict` in one pass: overall verdict badge, per-metric sanity flags with confidence scores, deterministic artifact-detection results with confidence scores, and Top-5 VLM topology reads with confidence scores. When the verdict is a clean `PASS` and all confidence scores are above threshold, no items require resolution and this is a single-click confirm; when any result is `WARNING` / `FAIL` or any confidence score is below threshold, the human resolves those items inline before Final Synthesis runs.
  - Converts all generated local PNG figures (multi-metric heatmaps and network graphs) into Base64 data URIs: `data:image/png;base64,...`.
  - Transcribes parameter configurations, numerical summaries, Evaluator audit verdicts, and the Cross-Metric Evidence Summary table into the structured Markdown template.
  - Formulates mandatory methodological caveats regarding heuristic thresholding and multiple comparisons.
  - **Zero Re-Analysis:** Pure compilation and document formatting. Performs zero re-computation or secondary data interpretation.

---

## 7. Final Output Specification: `report_<run_id>.md`

The deliverable is a single, client-side renderable Markdown document structured in the following order:

1. **Header & Execution Metadata:** Run ID, timestamp, target dataset filename, and overall status badge (`PASS`, `WARNING`, or `FAIL`).
2. **Analysis Plan & Methodology Summary:**
   - Markdown table of locked parameters: Metrics Computed (list of 5 canonical IDs), Frequency Band ($f_{\min} - f_{\max}$ Hz), Included Channels ($N$), Condition Name.
   - **Full Confirmed Manifest (Gate 1):** Every value from `parameter_manifest`, presented as a table with four columns: `proposed_value` (the pre-filled default), whether the row was flagged `⚠ NEEDS YOUR INPUT`, `risk_tier` (`low` or **`elevated`**), and `human_approved_value` (the final confirmed value, which may differ from the proposed default if the human overrode it). Downstream readers auditing the report without having seen Gate 1 can use the `risk_tier` column to identify which confirmed values carry domain-level consequence if wrong.
   - Preprocessing note: Confirms filtering parameters, Common Average Reference (CAR), bad channel detection & drop (labels listed, with retained channel count), epoch-length quality gate, and explicit notice of un-manipulated phase dynamics (No ICA / No amplitude rejection / No channel interpolation).
   - Mathematical Transformation Note: Explains that imaginary coherence is stored and evaluated as its absolute magnitude ($|\text{imcoh}|$) to establish a non-negative, $[0, 1]$-bounded, symmetric metric matching PLI and wPLI.
3. **Cross-Metric Evidence Summary:**
   - Per-channel-pair complementary evidence synthesis table:
     | Channel Pair | Phase-Robust Mean (PLI / wPLI / \|ImCoh\|) | Zero-Lag Mean (PLV / Coherence) | Joint Evidence Classification | Evaluator Supporting Visual Read |
     | :--- | :--- | :--- | :--- | :--- |
     | `C3 - P3` | `0.42 (Strong)` | `0.58 (Strong)` | **`FULLY CONVERGENT`** | *(Visual anchor: Plausible intra-hemispheric sensorimotor network)* |
     | `F3 - P3` | `0.26 (Strong)` | `0.30 (Weak)` | **`CONFIRMED COUPLING`** | *(Phase-lagged coupling; volume conduction excluded; uncorrected for null distribution)* |
     | `Fp1 - Fp2` | `0.04 (Weak)` | `0.82 (Strong)` | **`ZERO-LAG COUPLING, CAUSE UNRESOLVED`** | *(Top-5 anchor: Spatial adjacency near frontal pole; ocular proximity noted)* |
     | `T7 - O1` | `0.02 (Weak)` | `0.06 (Weak)` | **`NO COUPLING`** | *(Baseline / noise floor)* |
   - *(Note: Individual Evaluator visual topology reads are reported strictly for the Top-5 strongest unresolved zero-lag pairs).*
   - **Mandatory Methodological Caveats & Interpretive Guidance:**
     > [!WARNING]
     > **Mandatory Methodological Caveats on Thresholding & Multiple Testing:**
     > 1. **Heuristic Thresholds & Sample-Size Bias:** The coupling thresholds ($\tau_{\text{phase}} = 0.20, \tau_{\text{zerolag}} = 0.35$) are deterministic operational heuristics uncorrected for epoch count. Because phase-synchrony metrics (PLV, Coherence, and sample PLI/wPLI) exhibit positive bias in small sample sizes, *CONFIRMED COUPLING* denotes phase-lagged coupling not explainable by volume conduction, but is not statistically validated against a surrogate null distribution.
     > 2. **Uncorrected Multiple Comparisons:** No family-wise error rate or False Discovery Rate (FDR) correction is applied across the $N(N-1)/2$ pairwise comparisons (~2,016 simultaneous tests for 64 channels).
     > 3. **Guidance on ZERO-LAG COUPLING, CAUSE UNRESOLVED:** Pairs classified as *ZERO-LAG COUPLING, CAUSE UNRESOLVED* indicate significant zero-lag synchronization that lacks phase-lagged confirmation. This classification means the underlying cause is **genuinely unresolved** — it is not a euphemism for "probably an artifact." Such patterns can arise equally from genuine near-zero-lag biological coordination or instantaneous volume conduction from shared cortical sources. Sensor-level metrics cannot distinguish between these two hypotheses on their own.
     > 4. **Elevated-Risk Confirmed Values:** The Full Confirmed Manifest table marks certain parameters as `elevated` risk tier. These values — reference scheme, coupling thresholds, artifact-detection thresholds, and quality-gate thresholds — directly control which channel pairs are classified and whether artifact flags fire. Readers auditing this report should review those rows in the manifest regardless of whether the human overrode them; a confidently resolved default is not the same as a scientifically validated choice for this dataset.
4. **Per-Metric Connectivity Numerical Summaries:**
   - Individual summary tables for each computed metric: Mean, Median, Std Dev, Min/Max, Sparsity, and Top-3 connected pairs.
5. **Data Cleaning Visualization (Base64 Embedded):**
   - `![Channels Before Cleaning](data:image/png;base64,...)` — Raw channel time-series with bad channels highlighted in a distinct color, generated *before* any channel is removed.
   - `![Channels After Cleaning](data:image/png;base64,...)` — Retained channel time-series after bad channel drop.
   - *(If no bad channels were detected, this section states: "No bad channels detected; no channels were dropped." and the plot shows the full channel set.)*
6. **Visual Figure Gallery (Base64 Embedded):**
   - Embedded Heatmaps for all computed metrics: `![PLI Heatmap](data:image/png;base64,...)`, `![wPLI Heatmap](data:image/png;base64,...)`, `![|ImCoh| Heatmap](data:image/png;base64,...)`, `![PLV Heatmap](data:image/png;base64,...)`, `![Coherence Heatmap](data:image/png;base64,...)`.
   - Embedded Network Graphs for all computed metrics.
7. **Evaluator Quality Audit & Forensic Findings:**
   - Per-check confidence scores: for each metric, whether sanity checks passed (`values_in_bounds`, `matrix_is_symmetric`, `is_sparse_or_flat`) alongside the `confidence` value for that metric's assessment; for each artifact type (ocular, muscle, electrode pop), whether `detected` and the `confidence` score reflecting how clearly the heuristic threshold was met or exceeded; for each Top-5 VLM topology read, the `confidence` score attached to that visual judgment.
   - Sanity check checklist across all metrics (Bounds $[0, 1]$, Symmetry $M = M^T$, Flatness).
   - Artifact Risk Callout Box:
     - Ocular Contamination: [Detected / Clear]
     - Muscle Intrusion: [Detected / Clear]
     - Electrode Pop: [Detected / Clear]
     - Flagged Channels: Listed if any are contaminated.
   - Evaluator visual observations transcript.
   - *(Full per-check findings are also available as a machine-readable table in `outputs/evaluation_summary_<run_id>.csv` for cross-run quality tracking outside the report.)*
8. **Scientific Recommendations & Caveats:**
   - Actionable caveats generated by the Evaluator guiding the researcher on interpreting confirmed versus unresolved couplings.

---

## 8. Reasoning Trace Visibility (Supervisor Audit Log)

A significant operational gap in LLM agent pipelines is the loss of reasoning context: under the baseline state design, the Supervisor's ReAct scratchpad is discarded once the node finishes execution, and only the final resolved `AnalysisPlan` enters `GraphState`.

To preserve auditability without violating the core architectural rules (no state pollution and keeping `report_<run_id>.md` clean and user-centric), the system introduces a **lightweight, per-run audit trace logger**:
1. **Dedicated Log File:** For every run, the Supervisor writes its full interaction trace to `logs/trace_<run_id>.json` (or `.log`).
2. **Captured Information:** Initial user prompt, model thoughts, tool calls, tool observations, clarifications, and plan validation checks.
3. **Architectural Isolation Rules:**
   - *No GraphState Pollution:* Trace is written directly to disk via standard logging handlers.
   - *No Report Bloat:* Trace log is omitted from `report_<run_id>.md`.
   - *Zero Pipeline Complexity:* Implemented purely as a logger attached to the Supervisor node.

---

## 9. Code and Script Reviewability: Deterministic vs. Probabilistic Nodes

| Pipeline Component | Underlying Mechanism | Reviewability & Audit Paradigm |
| :--- | :--- | :--- |
| **Data Preparation** (Node 2) | Deterministic Python (MNE-Python) | **Source Code Inspection:** Plain, imperative Python code. Code review verifies mathematical correctness of referencing, dead-channel variance checks, epoch length $f_{\min}$ checks, and filter bands. |
| **Connectivity Analysis** (Node 3) | Deterministic Python (MNE-Connectivity, NetworkX) | **Source Code & Intermediate Output Inspection:** Fully inspectable, reproducible Python code. Numerical algorithms, $|\text{imcoh}|$ transformation, intermediate outputs (`outputs/<metric>_<run_id>.csv`, `outputs/evidence_summary_<run_id>.csv`), and heuristic threshold comparisons can be audited directly against mathematical formulas. |
| **Evaluator Sanity Checks** (Node 4 - Part 1) | Deterministic Python / NumPy | **Source Code Inspection:** Deterministic mathematical assertions checking $[0, 1]$ bounds, matrix symmetry ($M = M^T$), and flatline SNR thresholds across all matrices. |
| **Supervisor** (Node 1) | Probabilistic LLM (ReAct Agent + Tools) | **Trace Log Audit:** Auditable exclusively through the per-run reasoning trace log (`logs/trace_<run_id>.json`), recording the exact sequence of thoughts and tool calls. |
| **Evaluator Visual Inspection** (Node 4 - Part 2) | Probabilistic Multimodal VLM | **Structured Verdict & Observations Audit:** Visual artifact detections and Top-5 visual topology reads are audited via the structured fields in `EvaluationVerdict`. |
| **Final Synthesis** (Node 5) | Single-Shot Text LLM | **Report vs. State Cross-Verification:** Pure transcription and formatting. Audited by verifying that reported metrics, flags, and figures strictly match upstream values in `GraphState`. |

---

## 10. Open Decision: AnalysisPlan Representation of Multi-Metric Execution

In the original single-metric design, `AnalysisPlan.metric` was a single locked axis: the user chose exactly one metric before anything ran. With the paradigm shifting to multi-metric complementary evidence synthesis by default, how `AnalysisPlan` represents metric execution remains an **OPEN ARCHITECTURAL DECISION** requiring user confirmation.

### Option (a): Drop `metric` as a Locked Axis Entirely (Fixed 3-Axis Plan)
- **Design:** The locked plan contains exactly three axes: `freq_band`, `channels`, and `condition`. The system hardcodes the execution of all five metrics (`pli`, `wpli`, `imcoh`, `plv`, `coh`) on every single run.
- **Trade-Offs:**
  - *Advantages:* Maximizes simplicity and standardizes pipeline execution. Every single report contains the complete 5-metric battery, ensuring complementary evidence synthesis and the 4-category classification are guaranteed to be present and mathematically comparable across all runs. Completely eliminates user confusion or ambiguity regarding metric selection.
  - *Disadvantages:* Imposes higher computational cost on every run and produces heavier reports, even when a user only requires a rapid check or specific metric. Removes researcher autonomy to restrict execution.

### Option (b): Keep `metrics` in the Plan as a List, Defaulting to All Five (Flexible 4-Axis Plan)
- **Design:** `AnalysisPlan` retains metric selection as an axis, typed as `metrics: List[MetricEnum]`. It defaults automatically to all five metrics unless the user explicitly requests a restricted subset (e.g., `"compute only PLI and Coherence"`).
- **Trade-Offs:**
  - *Advantages:* Provides flexibility for targeted scientific inquiries or constrained compute environments. Retains full multi-metric default behavior without forcing unnecessary compute when a user explicitly wants fewer metrics.
  - *Disadvantages:* Reintroduces edge cases regarding diagnostic interpretation: if a user requests only phase-robust metrics (e.g., `["pli", "wpli"]`) or only zero-lag metrics (`["plv"]`), the cross-group complementary synthesis cannot run, requiring fallback logic in the Evaluator and Report Synthesis.

### Architectural Recommendation
> [!IMPORTANT]
> **Recommendation: Option (b) with Synthesis Guardrails.**
> We recommend **Option (b)**: define `metrics: List[MetricEnum]` with a default factory supplying all five foundational metrics. If a user does not specify a metric, the pipeline computes all five and runs the full complementary evidence synthesis. If the user explicitly requests a subset that lacks representation from either the phase-robust or zero-lag group, the Supervisor locks the requested subset but logs an advisory note that the 4-category cross-metric evidence synthesis will be disabled.
>
> *Status: Marked as an OPEN DECISION requiring explicit confirmation before implementation.*

---

## 11. RESOLVED: Human-Confirmation Granularity

**Status: RESOLVED as of v2 (2-gate architecture, Section 3.2).** This is no longer an open decision.

The adopted design uses exactly two gates: **Gate 1** (Pre-Flight Checklist) covers all parameters that affect scientific conclusions — including the three scientific axes, metric selection, reference scheme, quality-gate thresholds, coupling thresholds, and artifact-detection thresholds — shown as a single screen with inline confidence-based flagging. **Gate 2** (Pre-Publication Review) collects the Evaluator’s full `EvaluationVerdict` (with per-check confidence scores) for human review before the report is compiled. Cosmetic constants are hardcoded, not shown at either gate. The Evaluator never interrupts mid-run — all uncertainty is expressed via confidence scores deferred to Gate 2.

---

## 13. Observability: Trace Logs & Experiment Tracking

The pipeline writes two complementary observability artefacts per run without touching `GraphState` or introducing any new human checkpoint or control-flow branch.

### 13.1 Supervisor Reasoning Trace (existing — see Section 8)
The Supervisor writes its full ReAct scratchpad to `logs/trace_<run_id>.json` for tool-call-level auditability. See Section 8 for details.

### 13.2 MLflow Experiment Tracking (passive side-effect wrapper)

A thin centralized wrapper reads `GraphState` after each node returns and logs to MLflow. The deterministic signal-processing code in Data Preparation and Connectivity Analysis contains **zero MLflow imports**. All MLflow calls are wrapped in `try/except`; any failure emits a warning and execution continues unaffected.

Every pipeline run opens one MLflow run tagged with the shared `run_id`. The following data is logged:

**Params** *(logged once at Gate 1 sign-off, from `parameter_manifest.human_approved_value`):*
- All three scientific axes: `freq_band_name`, `fmin`, `fmax`, `channels` (comma-joined), `condition`.
- Metric selection: `metrics` (comma-joined canonical IDs).
- Reference scheme: `reference`.
- All quality-gate thresholds: `bad_channel_variance_threshold`, `min_cycles`.
- All coupling thresholds: `tau_phase`, `tau_zerolag`.
- All artifact-detection thresholds: `ocular_variance_ratio`, `muscle_power_threshold`, `electrode_pop_sigma`.

**Metrics** *(logged as each node completes):*
- *Data Preparation:* `channels_dropped_count`, `channels_retained_count`, `epoch_duration_s`.
- *Connectivity Analysis:* for each of the 5 metrics — `{metric}_mean`, `{metric}_std`, `{metric}_sparsity`; plus `count_fully_convergent`, `count_confirmed_coupling`, `count_zero_lag_unresolved`, `count_no_coupling`.
- *Evaluator:* for each metric — `sanity_{metric}_confidence`; for each artifact type — `artifact_{type}_detected` (0/1), `artifact_{type}_confidence`; for each Top-5 topology read — `topology_{pair}_confidence`.

**Artifacts** *(logged as each node completes):*
- All files written to `outputs/` and `logs/`: per-metric connectivity CSVs, evidence summary CSV, `evaluation_summary_<run_id>.csv`, all heatmap and network graph PNGs, channel before/after plots, `report_<run_id>.md`, `trace_<run_id>.json`.

**Tags** *(set at run close):*
- `run_id`, `overall_verdict` (`PASS`/`WARNING`/`FAIL`), `gate1_had_overrides` (`true`/`false`).

MLflow is an additional cross-run index over data that already exists in the filesystem — it does not replace the MD report as the human-facing deliverable or the evaluation CSV as the machine-readable findings table.

---

## 14. Deliberately Deferred Capabilities

The following capabilities are explicitly out of scope for this prototype:
- **Statistical Surrogate & Null-Distribution Testing:** Generation of phase-shuffled or trial-shuffled surrogate data distributions to establish empirical statistical significance per pair is deferred. Computing hundreds of surrogate iterations per pair is computationally prohibitive for an early prototype. Consequently, the fixed thresholds ($\tau_{\text{phase}} = 0.20, \tau_{\text{zerolag}} = 0.35$) are deterministic operational heuristics uncorrected for epoch count.
- **Multiple Comparisons Correction (e.g., FDR / Bonferroni):** Statistical adjustment for the simultaneous evaluation of up to thousands of pairwise connections is deferred; uncorrected exploratory findings are reported with explicit methodological caveats.
- **Persistent Cross-Run Memory & Database Storage:** No external vector store or multi-session database.
- **Advanced Connectivity Metrics Beyond the Five Core Metrics:** Directed metrics (dPLI, Generalized Partial Directed Coherence), amplitude-based metrics (AEC - Amplitude Envelope Correlation), and nonlinear information metrics (Mutual Information) are deferred.
- **Web User Interface:** Execution is driven via CLI and outputs standalone Markdown files. Gates 1–2 (Section 3.2) are surfaced as CLI prompts, not a graphical review UI.
- **Iterative Evaluator Loops:** The Evaluator issues a single complete verdict per run; low-confidence items are resolved by the human at Gate 2 rather than triggering a re-evaluation pass.
- **Dynamic Source Localization:** Analysis is strictly at the electrode/sensor level; no forward/inverse modeling or dipole source estimation.

---

## 15. Project Folder Structure

### 15.1 Scalability & Architectural Role Separation

In this project, **scalability** means that the directory and package layout organizes code by its fundamental **architectural role** (deterministic computation vs. probabilistic/agentic reasoning vs. data schemas vs. configuration/defaults vs. human interaction gates vs. observability) rather than grouping files by sequential pipeline stage numbers.

This role-based boundary ensures long-term architectural stability across four key growth axes:
1. **Metric Expansion Beyond the Core 5:** Adding new connectivity algorithms (or advanced metrics like dPLI, AEC, or mutual information) requires adding independent computation functions under the deterministic connectivity module without modifying agent logic, schema definitions, or other metrics.
2. **Pipeline Node Expansion (e.g., a 6th Node):** Adding future stages (such as dynamic source localization or statistical surrogate testing) introduces a new module under its appropriate role without forcing renumbering, restructuring, or breaking existing node boundaries.
3. **UI Portability (CLI to Web UI):** The human-in-the-loop gates (Gate 1 Pre-Flight and Gate 2 Pre-Publication) are cleanly isolated behind an abstract gate interface. When transitioning from CLI prompts to a graphical web dashboard (e.g., FastAPI/React), the UI layer can be replaced or extended with zero modifications to the core node functions or LangGraph state logic.
4. **Data Ingestion & Format Agnosticism:** Dataset querying and loading are decoupled through dedicated tool/loader interfaces, allowing future extensions to BIDS, EDF, or custom binary EEG recordings beyond current MNE-compatible formats.

**Physical Boundary for Deterministic Code:**
The `src/fc_pipeline/deterministic/` directory tree contains pure, inspectable mathematical and signal-processing logic. It is subject to a strict architectural invariant: **zero imports** from LangChain, LangGraph, multimodal VLM/LLM providers, or MLflow. This enables the complete deterministic test suite to execute rapidly and locally in 100% offline isolation without network calls, API keys, or mocked LLM clients.

### 15.2 Directory Tree (Down to File Level)

```
fc_pipeline/
├── pyproject.toml                                      # Project metadata, build system, entry points, and dependency declarations
├── requirements.txt                                    # Direct pinned dependencies (MNE, LangGraph, MLflow, Pydantic, etc.)
├── README.md                                           # Setup, installation, environment variables, and CLI quickstart guide
├── outputs/                                            # Target directory for generated PNG figures and CSV tables (Appendix A)
│   └── .gitkeep                                        # Placeholder keeping empty directory in version control
├── logs/                                               # Target directory for Supervisor JSON reasoning traces (Appendix A)
│   └── .gitkeep                                        # Placeholder keeping empty directory in version control
├── src/                                                # Top-level Python source root
│   └── fc_pipeline/                                    # Main package namespace
│       ├── __init__.py                                 # Top-level package initialization and version string
│       ├── main.py                                     # CLI entry point parsing user arguments and launching graph execution
│       ├── config/                                     # Single source of truth for constants, thresholds, and lookups
│       │   ├── __init__.py                             # Re-exports configuration constants and lookup dictionaries
│       │   ├── constants.py                            # Appendix A cosmetic constants, colormaps, DPI, and file templates
│       │   ├── thresholds.py                           # Quality gate, coupling classification (tau), and artifact thresholds
│       │   └── metric_canonicalization.py              # Section 6.1 hardcoded lookup table mapping metric aliases to MetricEnum
│       ├── schemas/                                    # Single source of truth for shared Pydantic and TypedDict models (Section 5)
│       │   ├── __init__.py                             # Re-exports all schema classes for clean package-level imports
│       │   ├── enums.py                                # MetricEnum, EvidenceCategoryEnum, VisualTopologyReadEnum, OverallVerdictEnum
│       │   ├── plan.py                                 # FrequencyBand and AnalysisPlan data models
│       │   ├── manifest.py                             # ParameterManifestEntry data model with risk_tier field
│       │   ├── evidence.py                             # PairEvidenceSynthesis data model for 4-category classification
│       │   ├── evaluation.py                           # SanityCheckFlags, ArtifactDetectionResult, ArtifactSignatures, EvaluationVerdict
│       │   └── state.py                                # GraphState TypedDict defining shared LangGraph pipeline state
│       ├── deterministic/                              # Pure mathematical signal processing (ZERO LLM/VLM/MLflow imports)
│       │   ├── __init__.py                             # Package marker for deterministic processing namespace
│       │   ├── data_prep/                              # Node 2 deterministic preprocessing routines (Section 6.2)
│       │   │   ├── __init__.py                         # Re-exports data preparation functions
│       │   │   ├── cleaning.py                         # Variance-based flatline/bad-channel identification and channel removal
│       │   │   ├── referencing.py                      # Re-referencing implementation (average, mastoid, bipolar)
│       │   │   ├── epoching.py                         # Filtering, epoch segmentation, and epoch duration vs fmin checks
│       │   │   └── plotting.py                         # Pre/post cleaning channel time-series diagnostic figure rendering
│       │   ├── connectivity/                           # Node 3 deterministic functional connectivity engine (Section 6.3)
│       │   │   ├── __init__.py                         # Re-exports connectivity and classification functions
│       │   │   ├── metrics.py                          # Single-pass 5-metric spectral estimation (PLI, wPLI, ImCoh, PLV, Coherence)
│       │   │   ├── classification.py                   # Pairwise 4-category evidence classification and CSV serialization
│       │   │   └── plotting.py                         # Viridis heatmap and circular network connectivity graph rendering
│       │   ├── artifact_detection/                     # Node 4 deterministic statistical heuristic detectors (Section 6.4)
│       │   │   ├── __init__.py                         # Re-exports statistical artifact detection routines
│       │   │   └── heuristics.py                       # Ocular variance ratio, muscle band power, and electrode pop sigma checks
│       │   └── sanity_checks/                          # Node 4 mathematical matrix sanity verifications (Section 6.4)
│       │       ├── __init__.py                         # Re-exports sanity check functions
│       │       └── matrix_checks.py                    # Unit interval [0,1] bounds, symmetry (M = M^T), and flatline SNR assertions
│       ├── agentic/                                    # Probabilistic LLM/VLM orchestration, prompts, and tools
│       │   ├── __init__.py                             # Package marker for agentic components namespace
│       │   ├── supervisor/                             # Node 1 Supervisor ReAct agent and tool bindings (Section 6.1)
│       │   │   ├── __init__.py                         # Re-exports Supervisor agent runner function
│       │   │   ├── agent.py                            # Supervisor ReAct agent loop, parameter resolution, and manifest assembly
│       │   │   ├── prompts.py                          # System prompts and ReAct reasoning instructions for parameter extraction
│       │   │   ├── tracer.py                           # File-based logger writing ReAct scratchpad to logs/trace_<run_id>.json
│       │   │   └── tools/                              # Isolated LangChain tool functions for dataset inspection
│       │   │       ├── __init__.py                     # Re-exports all four Supervisor inspection tools
│       │   │       ├── dataset_info.py                 # Tool: get_dataset_info (header, channels, sample rate, duration)
│       │   │       ├── dataset_conditions.py           # Tool: get_dataset_conditions (epoch event triggers and condition labels)
│       │   │       ├── frequency_band.py               # Tool: resolve_frequency_band (standard band boundaries from query)
│       │   │       └── channel_selection.py            # Tool: resolve_channel_selection (channel name fuzzy matching & validation)
│       │   ├── evaluator/                              # Node 4 Multimodal VLM visual inspection (Section 6.4)
│       │   │   ├── __init__.py                         # Re-exports Evaluator VLM inspection runner
│       │   │   ├── agent.py                            # Evaluator coordinator orchestrating sanity checks, heuristics, and VLM
│       │   │   ├── prompts.py                          # Multimodal prompts for visual artifact and Top-5 topology reads
│       │   │   ├── vlm_client.py                       # VLM API payload builder, base64 image encoder, and response parser
│       │   │   └── csv_writer.py                       # Flattens EvaluationVerdict into evaluation_summary_<run_id>.csv per Section 6.4 Step 4; pure serialization, no new computation.
│       │   └── synthesis/                              # Node 5 Final Synthesis LLM and report generator (Section 6.5 & 7)
│       │       ├── __init__.py                         # Re-exports Final Synthesis node runner
│       │       ├── agent.py                            # Synthesis LLM prompt invocation for structured report narrative
│       │       ├── prompts.py                          # Synthesis system prompt enforcing strict grounded transcription
│       │       └── report_builder.py                   # Self-contained Markdown assembler with embedded base64 image assets
│       ├── gates/                                      # Human-in-the-loop checkpoint interfaces (Section 3.2 & 14)
│       │   ├── __init__.py                             # Re-exports active gate interface renderers
│       │   ├── base.py                                 # Abstract gate interface defining render_gate1 and render_gate2 signatures
│       │   └── cli/                                    # Terminal CLI implementation of interactive review gates
│       │       ├── __init__.py                         # Re-exports CLI gate renderers
│       │       ├── gate1_preflight.py                  # Gate 1 interactive CLI checklist (values, needs_input flag, risk_tier)
│       │       ├── gate2_prepublication.py             # Gate 2 review CLI (PASS/WARNING/FAIL, confidence, Top-5 topology read)
│       │       └── renderers.py                        # Terminal table formatting and prompt styling utilities
│       ├── observability/                              # Passive experiment tracking and metrics monitoring (Section 13.2)
│       │   ├── __init__.py                             # Re-exports MLflow tracker wrapper
│       │   └── mlflow_tracker.py                       # Non-blocking wrapper logging params, metrics, and artifacts to MLflow
│       └── pipeline/                                   # LangGraph pipeline construction and node wiring (Section 2 & 3.1)
│           ├── __init__.py                             # Re-exports compiled pipeline application graph
│           ├── nodes.py                                # LangGraph node adapter functions matching Callable[[GraphState], GraphState]
│           └── graph.py                                # StateGraph assembly, node addition, edge routing, and checkpoint gates
└── tests/                                              # Comprehensive test suite mirroring source structure
    ├── __init__.py                                     # Test package initialization
    ├── conftest.py                                     # Pytest fixtures: synthetic MNE Raw/Epochs datasets and sample GraphState
    ├── unit/                                           # Isolated unit test suites
    │   ├── __init__.py                                 # Unit test namespace marker
    │   ├── schemas/                                    # Validation tests for Pydantic models and serialization
    │   │   ├── __init__.py                             # Schema test marker
    │   │   └── test_schemas.py                         # Tests ensuring schema validation rules and constraints pass
    │   ├── config/                                     # Tests for configuration defaults and canonicalization lookups
    │   │   ├── __init__.py                             # Config test marker
    │   │   └── test_config.py                          # Tests verifying canonicalization aliases and threshold definitions
    │   ├── deterministic/                              # OFFLINE tests: zero LLM/VLM dependencies, zero network calls
    │   │   ├── __init__.py                             # Deterministic test marker
    │   │   ├── test_data_prep.py                       # Tests for referencing, channel filtering, and epoch duration checks
    │   │   ├── test_connectivity.py                    # Tests verifying 5-metric computation and matrix symmetry
    │   │   ├── test_classification.py                  # Tests for 4-category threshold comparison logic and CSV outputs
    │   │   ├── test_artifact_detection.py              # Tests for deterministic ocular, muscle, and pop threshold heuristics
    │   │   └── test_sanity_checks.py                   # Tests verifying matrix [0, 1] range, symmetry, and flatline SNR assertions
    │   ├── agentic/                                    # Unit tests for agent tools, prompt formatting, and clients (mocked LLM)
    │   │   ├── __init__.py                             # Agentic test marker
    │   │   ├── test_supervisor_tools.py                # Tests verifying Supervisor tool functions against mock EEG headers
    │   │   ├── test_supervisor_agent.py                # Tests verifying ReAct manifest extraction logic with mock LLM
    │   │   ├── test_evaluator_agent.py                 # Tests verifying VLM payload generation and response parsing with mock VLM
    │   │   ├── test_evaluator_csv_writer.py            # Tests verifying EvaluationVerdict serialization to CSV matching schema
    │   │   └── test_synthesis_agent.py                 # Tests verifying Markdown report transcription rules with mock LLM
    │   └── gates/                                      # Unit tests for CLI gate prompt rendering and user override handling
    │       ├── __init__.py                             # Gate test marker
    │       └── test_cli_gates.py                       # Tests for Gate 1 and Gate 2 terminal prompt parsing and input validation
    └── integration/                                    # Multi-component workflow integration tests
        ├── __init__.py                                 # Integration test marker
        ├── test_pipeline_graph.py                      # End-to-end LangGraph DAG execution test with deterministic nodes
        └── test_mlflow_integration.py                  # Passive MLflow logging verification ensuring no node execution failures
```

### 15.3 Directory to Specification Mapping Table

| Directory | Spec Section(s) Implemented |
| :--- | :--- |
| `config/` | Section 6.1 (metric canonicalization lookup), Section 6.3 (coupling thresholds $\tau$), Section 6.4 (artifact thresholds), Appendix A (cosmetic constants & path templates) |
| `schemas/` | Section 5.1 (Pydantic models: `MetricEnum`, `EvidenceCategoryEnum`, `VisualTopologyReadEnum`, `ParameterManifestEntry`, `EvaluationVerdict`, etc.), Section 5.2 (`GraphState` TypedDict) |
| `deterministic/` | Section 6.2 (Node 2: Data Preparation), Section 6.3 (Node 3: Connectivity Analysis & 4-category classification), Section 6.4 (Node 4: Steps 1–2 deterministic sanity & artifact checks) |
| `agentic/` | Section 6.1 (Node 1: Supervisor ReAct agent & tools), Section 6.4 (Node 4: Steps 3–4 VLM topology read & CSV serialization), Section 6.5 (Node 5: Final Synthesis), Section 7 (Report spec), Section 8 & 13.1 (Reasoning trace) |
| `gates/` | Section 3.2 (Gate 1 Pre-Flight & Gate 2 Pre-Publication review), Section 14 (CLI-rendered human checkpoints) |
| `observability/` | Section 13.2 (MLflow passive wrapper logging params, metrics, and artifacts) |
| `pipeline/` | Section 2 (5-Node DAG architecture), Section 3.1 (LangGraph state graph compilation & node wiring) |
| `tests/` | Section 9 (Deterministic offline test isolation & agentic mock testing) |

---

## Appendix A — Hardcoded Cosmetic Constants

The following constants are hardcoded in source and are **never shown to the human for approval** at either gate. They do not affect scientific conclusions. To change them, edit the source directly.

| Constant Name | Value | Purpose |
| :--- | :--- | :--- |
| `HEATMAP_COLORMAP` | `"viridis"` | Sequential colormap for all connectivity heatmaps (uniform $[0,1]$ perceptual scale) |
| `HEATMAP_VMIN` | `0.0` | Fixed lower bound for heatmap color axis |
| `HEATMAP_VMAX` | `1.0` | Fixed upper bound for heatmap color axis |
| `FIGURE_DPI` | `150` | Resolution for all PNG output figures |
| `FIGURE_SIZE_INCHES` | `(10, 8)` | Default figure dimensions (width × height) |
| `NETWORK_GRAPH_LAYOUT` | `"circular"` | Node layout algorithm for network graph renders |
| `OUTPUT_DIR` | `"outputs/"` | Root directory for all intermediate PNG and CSV artifacts |
| `LOG_DIR` | `"logs/"` | Root directory for Supervisor reasoning trace logs |
| `REPORT_FILENAME_TEMPLATE` | `"report_{run_id}.md"` | Final report filename format |
| `TRACE_FILENAME_TEMPLATE` | `"trace_{run_id}.json"` | Supervisor audit trace filename format |
| `CONNECTIVITY_CSV_FILENAME_TEMPLATE` | `"{metric}_{run_id}.csv"` | Per-metric connectivity matrix CSV filename format (written to `OUTPUT_DIR`) |
| `EVIDENCE_CSV_FILENAME_TEMPLATE` | `"evidence_summary_{run_id}.csv"` | Complementary evidence synthesis CSV filename format (written to `OUTPUT_DIR`) |
| `EVALUATION_CSV_FILENAME_TEMPLATE` | `"evaluation_summary_{run_id}.csv"` | Evaluator sanity & artifact check summary CSV filename format (written to `OUTPUT_DIR`) |