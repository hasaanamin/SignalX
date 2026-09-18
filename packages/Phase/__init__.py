"""
Phase Analysis Package (Factory & Abstract Base Class Architecture)
Provides an enterprise Factory Pattern with Abstract Base Classes (ABCs) for computing
Instantaneous Phase, Phase Offset, Phase Difference (Interferometry/PDOA), Phase Deviation,
Phase Slope, Phase Jitter, Phase Stability (TDEV), Phase Discontinuity (Symbol Transitions), and Phase Unwrapping.
"""

from .base import BasePhaseMetric, PhaseMetricFactory
from ._common import _as_iq_array, _extract_sample_rate

import importlib

# Dynamic module imports to register all metric classes into the Factory
_inst_p_mod = importlib.import_module(".Instantaneous Phase", package=__name__)
_offset_p_mod = importlib.import_module(".Phase Offset", package=__name__)
_diff_p_mod = importlib.import_module(".Phase Difference", package=__name__)
_dev_p_mod = importlib.import_module(".Phase Deviation", package=__name__)
_slope_p_mod = importlib.import_module(".Phase Slope", package=__name__)
_jitter_p_mod = importlib.import_module(".Phase Jitter", package=__name__)
_stab_p_mod = importlib.import_module(".Phase Stability", package=__name__)
_disc_p_mod = importlib.import_module(".Phase Discontinuity", package=__name__)
_unwrap_p_mod = importlib.import_module(".Phase Unwrapping", package=__name__)

# Exported Concrete Metric Classes
InstantaneousPhaseMetric = _inst_p_mod.InstantaneousPhaseMetric
PhaseOffsetMetric = _offset_p_mod.PhaseOffsetMetric
PhaseDifferenceMetric = _diff_p_mod.PhaseDifferenceMetric
PhaseDeviationMetric = _dev_p_mod.PhaseDeviationMetric
PhaseSlopeMetric = _slope_p_mod.PhaseSlopeMetric
PhaseJitterMetric = _jitter_p_mod.PhaseJitterMetric
PhaseStabilityMetric = _stab_p_mod.PhaseStabilityMetric
PhaseDiscontinuityMetric = _disc_p_mod.PhaseDiscontinuityMetric
PhaseUnwrappingMetric = _unwrap_p_mod.PhaseUnwrappingMetric

# Exported Standalone Functional APIs
instantaneous_phase = _inst_p_mod.instantaneous_phase
phase_offset = _offset_p_mod.phase_offset
phase_difference = _diff_p_mod.phase_difference
pdoa = _diff_p_mod.pdoa
phase_deviation = _dev_p_mod.phase_deviation
phase_slope = _slope_p_mod.phase_slope
phase_jitter = _jitter_p_mod.phase_jitter
phase_stability = _stab_p_mod.phase_stability
phase_discontinuity = _disc_p_mod.phase_discontinuity
phase_unwrapping = _unwrap_p_mod.phase_unwrapping
unwrap_phase = _unwrap_p_mod.unwrap_phase


def phase_summary(iq_data, sample_rate: float | None = None) -> dict:
    """
    Computes a comprehensive phase analysis summary across the input signal.
    """
    fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
    
    offset_val = phase_offset(iq_data, sample_rate=fs)
    dev_dict = phase_deviation(iq_data, sample_rate=fs, mode="all")
    slope_val = phase_slope(iq_data, sample_rate=fs)
    jitter_dict = phase_jitter(iq_data, sample_rate=fs, mode="all")
    stab_dict = phase_stability(iq_data, sample_rate=fs, method="all")
    disc_dict = phase_discontinuity(iq_data, sample_rate=fs)

    return {
        "sample_rate_hz": fs,
        "phase_offset_rad": offset_val,
        "deviation": dev_dict,
        "phase_slope_rad_per_sec": slope_val,
        "jitter": jitter_dict,
        "stability": stab_dict,
        "discontinuities": disc_dict,
    }


__all__ = [
    # Base & Factory
    "BasePhaseMetric",
    "PhaseMetricFactory",
    # Classes
    "InstantaneousPhaseMetric",
    "PhaseOffsetMetric",
    "PhaseDifferenceMetric",
    "PhaseDeviationMetric",
    "PhaseSlopeMetric",
    "PhaseJitterMetric",
    "PhaseStabilityMetric",
    "PhaseDiscontinuityMetric",
    "PhaseUnwrappingMetric",
    # Functions
    "_as_iq_array",
    "_extract_sample_rate",
    "instantaneous_phase",
    "phase_offset",
    "phase_difference",
    "pdoa",
    "phase_deviation",
    "phase_slope",
    "phase_jitter",
    "phase_stability",
    "phase_discontinuity",
    "phase_unwrapping",
    "unwrap_phase",
    "phase_summary",
]
