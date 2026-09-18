"""
Frequency Analysis Package (Factory & Abstract Base Class Architecture)
Provides an enterprise Factory Pattern with Abstract Base Classes (ABCs) for computing
Instantaneous Frequency, Center/RF Frequency, Frequency Offset (CFO), Frequency Deviation,
Frequency Drift, Frequency Stability (Allan Deviation), Frequency Jitter, Frequency Slope, and Radar Chirp Rate.
"""

from .base import BaseFrequencyMetric, FrequencyMetricFactory
from ._common import _as_iq_array, _extract_sample_rate

import importlib

# Dynamic module imports to register all metric classes into the Factory
_inst_f_mod = importlib.import_module(".Instantaneous Frequency", package=__name__)
_center_f_mod = importlib.import_module(".Center Frequency", package=__name__)
_offset_f_mod = importlib.import_module(".Frequency Offset", package=__name__)
_dev_f_mod = importlib.import_module(".Frequency Deviation", package=__name__)
_drift_f_mod = importlib.import_module(".Frequency Drift", package=__name__)
_stab_f_mod = importlib.import_module(".Frequency Stability", package=__name__)
_jitter_f_mod = importlib.import_module(".Frequency Jitter", package=__name__)
_slope_f_mod = importlib.import_module(".Frequency Slope", package=__name__)
_chirp_f_mod = importlib.import_module(".Chirp Rate", package=__name__)

# Exported Concrete Metric Classes
InstantaneousFrequencyMetric = _inst_f_mod.InstantaneousFrequencyMetric
CenterFrequencyMetric = _center_f_mod.CenterFrequencyMetric
FrequencyOffsetMetric = _offset_f_mod.FrequencyOffsetMetric
FrequencyDeviationMetric = _dev_f_mod.FrequencyDeviationMetric
FrequencyDriftMetric = _drift_f_mod.FrequencyDriftMetric
FrequencyStabilityMetric = _stab_f_mod.FrequencyStabilityMetric
FrequencyJitterMetric = _jitter_f_mod.FrequencyJitterMetric
FrequencySlopeMetric = _slope_f_mod.FrequencySlopeMetric
ChirpRateMetric = _chirp_f_mod.ChirpRateMetric

# Exported Standalone Functional APIs
instantaneous_frequency = _inst_f_mod.instantaneous_frequency
center_frequency = _center_f_mod.center_frequency
rf_frequency = _center_f_mod.rf_frequency
frequency_offset = _offset_f_mod.frequency_offset
cfo = _offset_f_mod.cfo
frequency_deviation = _dev_f_mod.frequency_deviation
peak_deviation = _dev_f_mod.peak_deviation
frequency_drift = _drift_f_mod.frequency_drift
drift_rate = _drift_f_mod.drift_rate
frequency_stability = _stab_f_mod.frequency_stability
allan_deviation = _stab_f_mod.allan_deviation
frequency_jitter = _jitter_f_mod.frequency_jitter
frequency_slope = _slope_f_mod.frequency_slope
chirp_rate = _chirp_f_mod.chirp_rate
lfm_chirp_rate = _chirp_f_mod.lfm_chirp_rate


def frequency_summary(iq_data, sample_rate: float | None = None) -> dict:
    """
    Computes a comprehensive frequency analysis summary across the input signal.
    """
    fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
    
    fc = center_frequency(iq_data, sample_rate=fs)
    cfo_val = frequency_offset(iq_data, sample_rate=fs)
    dev_dict = frequency_deviation(iq_data, sample_rate=fs, mode="all")
    drift_dict = frequency_drift(iq_data, sample_rate=fs, return_rate=True)
    stab_dict = frequency_stability(iq_data, sample_rate=fs, method="all")
    jitter_dict = frequency_jitter(iq_data, sample_rate=fs, mode="all")
    slope_val = frequency_slope(iq_data, sample_rate=fs)
    chirp_dict = chirp_rate(iq_data, sample_rate=fs, return_info=True)

    return {
        "sample_rate_hz": fs,
        "center_frequency_hz": fc,
        "frequency_offset_hz": cfo_val,
        "deviation": dev_dict,
        "drift": drift_dict,
        "stability": stab_dict,
        "jitter": jitter_dict,
        "frequency_slope_hz_per_sec": slope_val,
        "chirp": chirp_dict,
    }


__all__ = [
    # Base & Factory
    "BaseFrequencyMetric",
    "FrequencyMetricFactory",
    # Classes
    "InstantaneousFrequencyMetric",
    "CenterFrequencyMetric",
    "FrequencyOffsetMetric",
    "FrequencyDeviationMetric",
    "FrequencyDriftMetric",
    "FrequencyStabilityMetric",
    "FrequencyJitterMetric",
    "FrequencySlopeMetric",
    "ChirpRateMetric",
    # Functions
    "_as_iq_array",
    "_extract_sample_rate",
    "instantaneous_frequency",
    "center_frequency",
    "rf_frequency",
    "frequency_offset",
    "cfo",
    "frequency_deviation",
    "peak_deviation",
    "frequency_drift",
    "drift_rate",
    "frequency_stability",
    "allan_deviation",
    "frequency_jitter",
    "frequency_slope",
    "chirp_rate",
    "lfm_chirp_rate",
    "frequency_summary",
]
