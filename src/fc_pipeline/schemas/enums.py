"""MetricEnum, EvidenceCategoryEnum, VisualTopologyReadEnum, OverallVerdictEnum."""
# Purpose: This file acts as the single source of truth for fixed categorical constants in the pipeline. 
# Specifically, MetricEnum establishes the canonical, lowercase string identifiers for the five 
# electrophysiological connectivity metrics supported by the system.
#The Supervisor agent will use this enum to validate whatever the user asked for in their natural-language prompt, map user terminology (like "phase locking" or "wPLI") to canonical tokens, and populate the AnalysisPlan that drives all downstream computation.
from enum import Enum   #  Imports the base Enum class from Python's standard library enum module.
class MetricEnum(str, Enum):    #  Class definition using Enum for type-safe, fixed string values.    
    PLI = 'pli'        #  Phase Lag Index: Phase synchrony metric robust to volume conduction.
    WPLI = "wpli"      #  Weighted Phase Lag Index: Phase synchrony weighted by coherence.
    IMAGINARY_COHERENCE = "imcoh"  #  Imaginary part of coherence: Measures phase synchrony.
    PLV = "plv"         #  Phase Locking Value: Measures phase synchrony.
    COHERENCE = "coh"     #  Coherence: Measures phase locking in frequency domain.
