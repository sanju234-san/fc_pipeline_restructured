"""Quality gate, coupling classification (tau), and artifact thresholds."""

# Coupling Classification Thresholds (Section 4.2 & Section 6.3)
TAU_PHASE: float = 0.20
TAU_ZEROLAG: float = 0.35

# Supervisor Gate 1 Resolution Cutoff (Section 6.1)
SUPERVISOR_CONFIDENCE_THRESHOLD: float = 0.80

# Scientific Completeness & Quality Gate Thresholds (Section 6.2 & Section 6.4)
N_CYCLES_MIN: int = 3  # Minimum cycles for reliable phase estimation
MIN_CYCLES: int = 3
BAD_CHANNEL_VARIANCE_THRESHOLD: float = 1e-15  # Near-zero variance threshold in V^2

# Artifact Detection Thresholds (Section 6.4)
OCULAR_VARIANCE_RATIO: float = 1.5  # Fronto-polar to posterior variance ratio threshold
MUSCLE_POWER_THRESHOLD: float = 2.0  # High-frequency (>40 Hz) power elevation threshold
ELECTRODE_POP_SIGMA: float = 3.0     # Standard deviations above cross-channel mean variance