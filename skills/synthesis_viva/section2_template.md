### Connectivity section template

## Connectivity Results

The pipeline computed five connectivity metrics for {N_channels} channels (total
{N_pairs} pairs) within the `{freq_band.name}` band (`{freq_band.fmin}–{freq_band.fmax}` Hz),
condition `{condition}`.

**Phase-coupled links (τ ≥ τ_phase = 0.20).** The pipeline classified
{N_phase} channel-pairs as phase-coupled based on a phase-coupling metric value
above the threshold. The most frequent channel pairs (by highest wPLI or PLI)
were:

  - {pair_1}: wPLI = {v1:.3f}
  - {pair_2}: wPLI = {v2:.3f}
  - {pair_3}: wPLI = {v3:.3f}

**Zero-lag / likely volume-conducted links (τ ≥ τ_zerolag = 0.35).** {N_vc}
pairs were classified as zero-lag / likely-volume-conducted based on an
Imaginary-Coherence / Coherence pair ratio consistent with volume conduction.
These pairs should not be interpreted as evidence for directed functional
coupling.

**Sub-threshold links.** {N_below} pairs fell below both thresholds and are
reported only by count.

For the full matrix, see the numerical outputs in:
  - metric_csv_paths: {csvs}
  - heatmap_image_paths: {heatmaps}
  - network_image_paths: {networks}
