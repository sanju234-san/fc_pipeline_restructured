"""Quality gate, coupling classification (tau), and artifact thresholds."""

# Coupling Classification Thresholds (Section 4.2 & Section 6.3)
TAU_PHASE: float = 0.20
TAU_ZEROLAG: float = 0.35

# Supervisor Gate 1 Resolution Cutoff (Section 6.1)
SUPERVISOR_CONFIDENCE_THRESHOLD: float = 0.80

# Scientific Completeness Checks (Gate 1 Advisories)
N_CYCLES_MIN: int = 3  # Minimum cycles for reliable phase estimation