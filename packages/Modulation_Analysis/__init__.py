from __future__ import annotations

# Result Dataclasses
from ._results import (
    AMAnalysisResult,
    FMAnalysisResult,
    PMAnalysisResult,
    AMDepthResult,
    FMDeviationResult,
    FMIndexResult,
    PMDeviationResult,
    SymbolRateCandidate,
    SymbolRateResult,
    ConstellationResult,
    PhaseStatesResult,
    FrequencyStatesResult,
    ConstellationFeaturesResult,
    ModulationClassificationResult,
    ModulationIndexResult,
)

# Base & Factory
from .base import (
    BaseModulationMetric,
    ModulationMetricFactory,
)

# 14 Public Modules & Functional Wrappers
from .AM_Analysis import AMAnalysis, compute_am_analysis, am_analysis_summary
from .FM_Analysis import FMAnalysis, compute_fm_analysis, fm_analysis_summary
from .PM_Analysis import PMAnalysis, compute_pm_analysis, pm_analysis_summary
from .AM_Depth import AMDepth, compute_am_depth
from .FM_Deviation import FMDeviation, compute_fm_deviation
from .FM_Index import FMIndex, compute_fm_index
from .PM_Deviation import PMDeviation, compute_pm_deviation
from .Symbol_Rate import SymbolRate, compute_symbol_rate
from .Phase_States import PhaseStates, compute_phase_states
from .Frequency_States import FrequencyStates, compute_frequency_states
from .Constellation_Generation import ConstellationGeneration, compute_constellation_generation
from .Constellation_Features import ConstellationFeatures, compute_constellation_features
from .Modulation_Type_Classification import ModulationTypeClassification, compute_modulation_type_classification
from .Modulation_Index import ModulationIndex, compute_modulation_index

__all__ = [
    # Base / Factory
    "BaseModulationMetric",
    "ModulationMetricFactory",
    # Dataclasses
    "AMAnalysisResult",
    "FMAnalysisResult",
    "PMAnalysisResult",
    "AMDepthResult",
    "FMDeviationResult",
    "FMIndexResult",
    "PMDeviationResult",
    "SymbolRateCandidate",
    "SymbolRateResult",
    "ConstellationResult",
    "PhaseStatesResult",
    "FrequencyStatesResult",
    "ConstellationFeaturesResult",
    "ModulationClassificationResult",
    "ModulationIndexResult",
    # Modules
    "AMAnalysis",
    "FMAnalysis",
    "PMAnalysis",
    "AMDepth",
    "FMDeviation",
    "FMIndex",
    "PMDeviation",
    "SymbolRate",
    "PhaseStates",
    "FrequencyStates",
    "ConstellationGeneration",
    "ConstellationFeatures",
    "ModulationTypeClassification",
    "ModulationIndex",
    # Functional Helpers
    "compute_am_analysis",
    "compute_fm_analysis",
    "compute_pm_analysis",
    "am_analysis_summary",
    "fm_analysis_summary",
    "pm_analysis_summary",
    "compute_am_depth",
    "compute_fm_deviation",
    "compute_fm_index",
    "compute_pm_deviation",
    "compute_symbol_rate",
    "compute_phase_states",
    "compute_frequency_states",
    "compute_constellation_generation",
    "compute_constellation_features",
    "compute_modulation_type_classification",
    "compute_modulation_index",
]
