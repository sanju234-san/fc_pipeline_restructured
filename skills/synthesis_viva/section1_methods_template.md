### Methods section template (insert after Scientific Context header)

## Methods

**Data Preparation.** The raw EEG recording was loaded via MNE-Python (Gramfort
et al. 2013). Bad channels were screened by a deterministic variance-audit
using a `bad_channel_variance_threshold` of `1e-15` (flat-channel detection)
combined with an `electrode_pop_sigma` of `3.0` σ from the per-channel mean
variance. A band-pass filter was applied between `{filter_l_freq}` Hz and
`{filter_h_freq}` Hz, followed by re-referencing to `{reference_applied}`.
The continuous data were epoched into `{epoch_count}` epochs of
`{epoch_duration_seconds}` s, locked to the `{condition}` condition label,
with a minimum of `{min_cycles}` cycles per epoch.

**Connectivity analysis.** Functional connectivity was computed using the
MNE-Connectivity library on the preprocessed epochs. The canonical metric set
included the Phase Lag Index (PLI), weighted PLI (wPLI), Imaginary Coherence
(ImCoh), Phase Locking Value (PLV), and Magnitude-Squared Coherence.

**Link classification.** Classified links were binned into: (i) phase-coupled
links using a classification threshold of `τ_phase = 0.20`; (ii) zero-lag or
volume-conduction suspected links using `τ_zerolag = 0.35`; and (iii)
sub-threshold links reported by count only. Classification thresholds are
centralized in `fc_pipeline/config/thresholds.py`.

**HITL and risk gates.** The analysis plan was reviewed at Gate 1
(preflight manifest review). All rows flagged `needs_human_input=True` or
carrying `risk_tier=elevated` required explicit human sign-off before
Data Prep executed.
