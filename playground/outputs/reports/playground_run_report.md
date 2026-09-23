# Supervisor Agent Integration Playground Run Report

## Execution Environment & Overview
- **Model**: `[CONFIGURED: RedHatAI/Ministral-3-14B-Instruct-2512]`
- **Endpoint**: `http://[CONFIGURED - INTERNAL IP/HOST MASKED]`
- **Data Source**: `REAL DATASET`
- **Dataset Path**: `[LOCAL_ROOT]/OneDrive\Desktop\CDAC\eeg_iq_pipeline\eeg_iq_pipeline\ds004796_split\sub-01\sub-01_eyesclosed_raw.fif`
- **Channels**: 127 (1000.0 Hz, Nyquist: 500.0 Hz)
- **Conditions**: `{'Stimulus/S  4': 1, 'Stimulus/S 11': 1}`

## Summary Results Table
| Case | Scenario Name | Routed Node | Status | Trace Events | MLflow Run ID |
|:---|:---|:---|:---:|:---:|:---|
| `playground_test_01` | Case 1: Fully-Specified Query | `__interrupt__` | 🛑 **FAIL** | 15 | `4ab00d3dad2343db842eeea1640754ec` |
| `playground_test_02` | Case 2: Ambiguous Query (Regional Channels) | `__interrupt__` | 🛑 **FAIL** | 15 | `b191d80d15be4e47b892711f6ddfc156` |
| `playground_test_03` | Case 3: Incomplete Query (Zero-Guessing Guardrail) | `informational_complete` | 🛑 **FAIL** | 9 | `d3702b53efd249eba49c8453df7b5330` |
| `playground_test_04` | Case 4: Metric-Subset Query (Synthesis Advisory Guardrail) | `__interrupt__` | 🛑 **FAIL** | 14 | `3b4bf26ed41b4de19e28121de7469651` |
| `playground_test_05` | Case 5: Balanced Metric Query (Synthesis Advisory Absent) | `__interrupt__` | 🛑 **FAIL** | 15 | `37d0f858606642bf804d9f3c71e41559` |
| `playground_test_06` | Case 6: Overview Plot Request + Analysis Plan | `clarification_pause` | 🛑 **FAIL** | 15 | `5afd249c30744e8aba48884a49cb45f4` |

---

## Case 1: Fully-Specified Query
- **Scenario ID**: `playground_test_01`
- **Query**: "compute PLI and coherence for alpha band on Fp1, Fz during Stimulus/S  4"
- **Goal**: Should resolve all parameters with high confidence, generate a complete plan, require no human intervention, and route to gate_1_review.
- **Routed Node**: `__interrupt__`
- **Status**: **FAIL** (Real Data: Neither plan nor clarification produced (routed to __interrupt__).)
- **Trace Event Count**: 15
- **MLflow Run ID**: `4ab00d3dad2343db842eeea1640754ec`

### Parameter Manifest
| Parameter | Category | Proposed Value | Confidence | Needs Human Input | Risk Tier |
|:---|:---|:---|:---:|:---:|:---:|
| `freq_band` | scientific_axis | alpha (8.0 - 12.0 Hz) | 1.00 | No | low |
| `channels` | scientific_axis | Fp1, Fz | 1.00 | No | low |
| `condition` | scientific_axis | Stimulus/S  4 | 1.00 | No | low |
| `metrics` | metric_selection | pli, coh | — | No | low |
| `reference` | engineering_threshold | unreferenced (proposed: average) | — | No | **ELEVATED** |
| `bad_channel_variance_threshold` | engineering_threshold | 1e-15 | — | No | **ELEVATED** |
| `min_cycles` | engineering_threshold | 3 | — | No | **ELEVATED** |
| `tau_phase` | engineering_threshold | 0.2 | — | No | **ELEVATED** |
| `tau_zerolag` | engineering_threshold | 0.35 | — | No | **ELEVATED** |
| `ocular_variance_ratio` | engineering_threshold | 1.5 | — | No | **ELEVATED** |
| `muscle_power_threshold` | engineering_threshold | 2.0 | — | No | **ELEVATED** |
| `electrode_pop_sigma` | engineering_threshold | 3.0 | — | No | **ELEVATED** |
| `trial_adequacy` | advisory | Only 1 trial(s) available for condition 'Stimulus/S  4'. Connectivity estimates from very few trials have limited statistical reliability. | — | ⚠️ **YES** | **ELEVATED** |
| `dataset_overview_plot` | informational | generate_dataset_overview_plot available (optional pre-preprocessing diagnostic) | — | No | low |
| `bad_channel_screening` | informational | Not yet performed. Will run in Data Preparation with before/after variance audit. | — | No | low |

### Dataset Overview Plot
*plot tool not invoked for this query*

---

## Case 2: Ambiguous Query (Regional Channels)
- **Scenario ID**: `playground_test_02`
- **Query**: "compute PLI and coherence for alpha band on frontal channels during Stimulus/S  4"
- **Goal**: Should resolve band, condition, and balanced metrics (PLI+Coh), map frontal channels with region confidence (< 0.8), setting needs_human_input=True ONLY on channels row, and route to gate_1_review.
- **Routed Node**: `__interrupt__`
- **Status**: **FAIL** (Real Data: Failed to process regional query (routed to __interrupt__).)
- **Trace Event Count**: 15
- **MLflow Run ID**: `b191d80d15be4e47b892711f6ddfc156`

### Parameter Manifest
| Parameter | Category | Proposed Value | Confidence | Needs Human Input | Risk Tier |
|:---|:---|:---|:---:|:---:|:---:|
| `freq_band` | scientific_axis | alpha (8.0 - 12.0 Hz) | 1.00 | No | low |
| `channels` | scientific_axis | Fp1, Fp2, F3, F4, F7, F8, Fz | 0.70 | ⚠️ **YES** | low |
| `condition` | scientific_axis | Stimulus/S  4 | 1.00 | No | low |
| `metrics` | metric_selection | pli, coh | — | No | low |
| `reference` | engineering_threshold | unreferenced (proposed: average) | — | No | **ELEVATED** |
| `bad_channel_variance_threshold` | engineering_threshold | 1e-15 | — | No | **ELEVATED** |
| `min_cycles` | engineering_threshold | 3 | — | No | **ELEVATED** |
| `tau_phase` | engineering_threshold | 0.2 | — | No | **ELEVATED** |
| `tau_zerolag` | engineering_threshold | 0.35 | — | No | **ELEVATED** |
| `ocular_variance_ratio` | engineering_threshold | 1.5 | — | No | **ELEVATED** |
| `muscle_power_threshold` | engineering_threshold | 2.0 | — | No | **ELEVATED** |
| `electrode_pop_sigma` | engineering_threshold | 3.0 | — | No | **ELEVATED** |
| `trial_adequacy` | advisory | Only 1 trial(s) available for condition 'Stimulus/S  4'. Connectivity estimates from very few trials have limited statistical reliability. | — | ⚠️ **YES** | **ELEVATED** |
| `dataset_overview_plot` | informational | generate_dataset_overview_plot available (optional pre-preprocessing diagnostic) | — | No | low |
| `bad_channel_screening` | informational | Not yet performed. Will run in Data Preparation with before/after variance audit. | — | No | low |

### Dataset Overview Plot
*plot tool not invoked for this query*

---

## Case 3: Incomplete Query (Zero-Guessing Guardrail)
- **Scenario ID**: `playground_test_03`
- **Query**: "analyze this EEG"
- **Goal**: Missing condition, channels, and band. Supervisor must halt with clarification_question, plan=None, and route to clarification_pause.
- **Routed Node**: `informational_complete`
- **Status**: **FAIL** (Real Data: Failed guardrail — plan=False, manifest=False, routed_node=informational_complete.)
- **Trace Event Count**: 9
- **MLflow Run ID**: `d3702b53efd249eba49c8453df7b5330`

### Parameter Manifest
*No parameter manifest generated (guardrail halted execution with clarification question).*

### Dataset Overview Plot
*plot tool not invoked for this query*

---

## Case 4: Metric-Subset Query (Synthesis Advisory Guardrail)
- **Scenario ID**: `playground_test_04`
- **Query**: "compute PLI and wPLI for alpha band on Fp1, Fz during Stimulus/S  4"
- **Goal**: Selected metrics contain only Phase-Robust metrics (PLI+wPLI, no Zero-Lag). Must trigger cross_metric_synthesis advisory row (risk_tier='elevated', needs_human_input=True) and route to gate_1_review.
- **Routed Node**: `__interrupt__`
- **Status**: **FAIL** (Real Data: Metric subset check failed (advisory=True, routed_node=__interrupt__).)
- **Trace Event Count**: 14
- **MLflow Run ID**: `3b4bf26ed41b4de19e28121de7469651`

### Parameter Manifest
| Parameter | Category | Proposed Value | Confidence | Needs Human Input | Risk Tier |
|:---|:---|:---|:---:|:---:|:---:|
| `freq_band` | scientific_axis | alpha (8.0 - 12.0 Hz) | 1.00 | No | low |
| `channels` | scientific_axis | Fp1, Fz | 1.00 | No | low |
| `condition` | scientific_axis | Stimulus/S  4 | 1.00 | No | low |
| `metrics` | metric_selection | pli, wpli | — | No | low |
| `reference` | engineering_threshold | unreferenced (proposed: average) | — | No | **ELEVATED** |
| `bad_channel_variance_threshold` | engineering_threshold | 1e-15 | — | No | **ELEVATED** |
| `min_cycles` | engineering_threshold | 3 | — | No | **ELEVATED** |
| `tau_phase` | engineering_threshold | 0.2 | — | No | **ELEVATED** |
| `tau_zerolag` | engineering_threshold | 0.35 | — | No | **ELEVATED** |
| `ocular_variance_ratio` | engineering_threshold | 1.5 | — | No | **ELEVATED** |
| `muscle_power_threshold` | engineering_threshold | 2.0 | — | No | **ELEVATED** |
| `electrode_pop_sigma` | engineering_threshold | 3.0 | — | No | **ELEVATED** |
| `trial_adequacy` | advisory | Only 1 trial(s) available for condition 'Stimulus/S  4'. Connectivity estimates from very few trials have limited statistical reliability. | — | ⚠️ **YES** | **ELEVATED** |
| `cross_metric_synthesis` | advisory | DISABLED: Requested metrics lack representation from both Phase-Robust (PLI/wPLI/ImCoh) and Zero-Lag (PLV/Coh) groups. 4-category artifact disambiguation will not run. | — | ⚠️ **YES** | **ELEVATED** |
| `dataset_overview_plot` | informational | generate_dataset_overview_plot available (optional pre-preprocessing diagnostic) | — | No | low |
| `bad_channel_screening` | informational | Not yet performed. Will run in Data Preparation with before/after variance audit. | — | No | low |

### Dataset Overview Plot
*plot tool not invoked for this query*

---

## Case 5: Balanced Metric Query (Synthesis Advisory Absent)
- **Scenario ID**: `playground_test_05`
- **Query**: "compute PLI and coherence for alpha band on Fp1, Fz during Stimulus/S  4"
- **Goal**: Selected metrics contain both Phase-Robust (PLI) and Zero-Lag (Coherence). Must produce a valid plan with cross_metric_synthesis advisory ABSENT, and route to gate_1_review.
- **Routed Node**: `__interrupt__`
- **Status**: **FAIL** (Real Data: Balanced metric check failed (advisory_absent=True, routed_node=__interrupt__).)
- **Trace Event Count**: 15
- **MLflow Run ID**: `37d0f858606642bf804d9f3c71e41559`

### Parameter Manifest
| Parameter | Category | Proposed Value | Confidence | Needs Human Input | Risk Tier |
|:---|:---|:---|:---:|:---:|:---:|
| `freq_band` | scientific_axis | alpha (8.0 - 12.0 Hz) | 1.00 | No | low |
| `channels` | scientific_axis | Fp1, Fz | 1.00 | No | low |
| `condition` | scientific_axis | Stimulus/S  4 | 1.00 | No | low |
| `metrics` | metric_selection | pli, coh | — | No | low |
| `reference` | engineering_threshold | unreferenced (proposed: average) | — | No | **ELEVATED** |
| `bad_channel_variance_threshold` | engineering_threshold | 1e-15 | — | No | **ELEVATED** |
| `min_cycles` | engineering_threshold | 3 | — | No | **ELEVATED** |
| `tau_phase` | engineering_threshold | 0.2 | — | No | **ELEVATED** |
| `tau_zerolag` | engineering_threshold | 0.35 | — | No | **ELEVATED** |
| `ocular_variance_ratio` | engineering_threshold | 1.5 | — | No | **ELEVATED** |
| `muscle_power_threshold` | engineering_threshold | 2.0 | — | No | **ELEVATED** |
| `electrode_pop_sigma` | engineering_threshold | 3.0 | — | No | **ELEVATED** |
| `trial_adequacy` | advisory | Only 1 trial(s) available for condition 'Stimulus/S  4'. Connectivity estimates from very few trials have limited statistical reliability. | — | ⚠️ **YES** | **ELEVATED** |
| `dataset_overview_plot` | informational | generate_dataset_overview_plot available (optional pre-preprocessing diagnostic) | — | No | low |
| `bad_channel_screening` | informational | Not yet performed. Will run in Data Preparation with before/after variance audit. | — | No | low |

### Dataset Overview Plot
*plot tool not invoked for this query*

---

## Case 6: Overview Plot Request + Analysis Plan
- **Scenario ID**: `playground_test_06`
- **Query**: "show me an overview plot of the dataset, then compute PLI and coherence for alpha band on Fp1, Fz during Stimulus/S  4"
- **Goal**: Explicitly requests dataset overview plot before analysis. Supervisor must organically call generate_dataset_overview_plot via ReAct loop, resolve all parameters, and route to gate_1_review.
- **Routed Node**: `clarification_pause`
- **Status**: **FAIL** (Real Data: Overview plot generated, but plan compilation failed (routed to clarification_pause).)
- **Trace Event Count**: 15
- **MLflow Run ID**: `5afd249c30744e8aba48884a49cb45f4`

### Clarification Question Emitted
> ### Compiled Plan for Run ID: `playground_test_06`
> **Status:** ✅ **AUTO-APPROVED** (All scientific axes resolved deterministically)
> 
> ---
> #### **1. Dataset Overview**
> - **File:** `sub-01_eyesclosed_raw.fif`
> - **Sampling Rate (sfreq):** 1000 Hz
> - **Duration:** 361.29 seconds (127 channels, unreferenced)
> - **Overview Plot:** Generated → [View here](outputs/plots/playground_test_06_overview.png)
>   *(Raw signals + PSD for all 127 channels; no metadata leakage.)*
> 
> ---
> #### **2. Scientific Axes**
> | **Axis**       | **User Request**       | **Resolved Value**               | **Validation Status**       |
> |----------------|------------------------|-----------------------------------|-----------------------------|
> | **Frequency**  | Alpha band             | 8–12 Hz (confidence: 1.0)         | ✅ Passed cycle check       |
> | **Channels**   | Fp1, Fz                | `['Fp1', 'Fz']` (confidence: 1.0) | ✅ Exact match               |
> | **Condition**  | Stimulus/S             | **Rejected** → Closest match: `Stimulus/S 4` and `Stimulus/S 11` (2 trials) | ⚠️ **CRITICAL MISMATCH**   |
> | **Metrics**    | PLI, Coherence         | PLI, Coherence                    | ✅ User-specified           |
> 
> ---
> #### **3. Critical Issue: Condition Resolution Failure**
> - **Problem:** The dataset contains **`Stimulus/S 4`** and **`Stimulus/S 11`**, but your request specified **`Stimulus/S`** (ambiguous).
> - **Impact:** Functional connectivity cannot be computed for a non-existent condition.
> - **Action Required:** Clarify the exact condition label(s) from these options:
>   - `Stimulus/S 4` (1 trial)
>   - `Stimulus/S 11` (1 trial)
>   - **OR** specify a different condition if intended.
> 
> ---
> #### **4. Proposed Fix**
> Reply with one of the following to proceed:
> - `Stimulus/S 4` (use only this condition)
> - `Stimulus/S 11` (use only this condition)
> - `Stimulus/S 4, Stimulus/S 11` (use both conditions)
> - **OR** correct the condition label if a typo was intended.
> 
> ---
> #### **5. FYI (Non-Blocking)**
> - The pipeline **automatically** rejected silent assumptions (e.g., guessing `Stimulus/S` → `Stimulus/S 4`).
> - If you intended a different condition (e.g., `EyesClosed`), reply with the **exact** label as shown in the dataset.

### Parameter Manifest
*No parameter manifest generated (guardrail halted execution with clarification question).*

### Dataset Overview Plot
![Overview](../plots/playground_test_06_overview.png)

> **LLM Plot Observation Summary:**
>
> ### Compiled Plan for Run ID: `playground_test_06`
> **Status:** ✅ **AUTO-APPROVED** (All scientific axes resolved deterministically)
> 
> ---
> #### **1. Dataset Overview**
> - **File:** `sub-01_eyesclosed_raw.fif`
> - **Sampling Rate (sfreq):** 1000 Hz
> - **Duration:** 361.29 seconds (127 channels, unreferenced)
> - **Overview Plot:** Generated → [View here](outputs/plots/playground_test_06_overview.png)
>   *(Raw signals + PSD for all 127 channels; no metadata leakage.)*
> 
> ---
> #### **2. Scientific Axes**
> | **Axis**       | **User Request**       | **Resolved Value**               | **Validation Status**       |
> |----------------|------------------------|-----------------------------------|-----------------------------|
> | **Frequency**  | Alpha band             | 8–12 Hz (confidence: 1.0)         | ✅ Passed cycle check       |
> | **Channels**   | Fp1, Fz                | `['Fp1', 'Fz']` (confidence: 1.0) | ✅ Exact match               |
> | **Condition**  | Stimulus/S             | **Rejected** → Closest match: `Stimulus/S 4` and `Stimulus/S 11` (2 trials) | ⚠️ **CRITICAL MISMATCH**   |
> | **Metrics**    | PLI, Coherence         | PLI, Coherence                    | ✅ User-specified           |
> 
> ---
> #### **3. Critical Issue: Condition Resolution Failure**
> - **Problem:** The dataset contains **`Stimulus/S 4`** and **`Stimulus/S 11`**, but your request specified **`Stimulus/S`** (ambiguous).
> - **Impact:** Functional connectivity cannot be computed for a non-existent condition.
> - **Action Required:** Clarify the exact condition label(s) from these options:
>   - `Stimulus/S 4` (1 trial)
>   - `Stimulus/S 11` (1 trial)
>   - **OR** specify a different condition if intended.
> 
> ---
> #### **4. Proposed Fix**
> Reply with one of the following to proceed:
> - `Stimulus/S 4` (use only this condition)
> - `Stimulus/S 11` (use only this condition)
> - `Stimulus/S 4, Stimulus/S 11` (use both conditions)
> - **OR** correct the condition label if a typo was intended.
> 
> ---
> #### **5. FYI (Non-Blocking)**
> - The pipeline **automatically** rejected silent assumptions (e.g., guessing `Stimulus/S` → `Stimulus/S 4`).
> - If you intended a different condition (e.g., `EyesClosed`), reply with the **exact** label as shown in the dataset.

---
