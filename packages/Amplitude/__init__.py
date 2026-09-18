"""
Amplitude and Power Measurement Package (Factory & Abstract Base Class Architecture)
Provides an enterprise Factory Pattern with Abstract Base Classes (ABCs) for computing
instantaneous IQ magnitude, envelope, peak/RMS power, variance, crest factor, PAPR, and dynamic range.
"""

from .base import BaseAmplitudeMetric, AmplitudeMetricFactory
from ._common import _as_iq_array
from ._common import _as_iq_array as normalize_iq_input

import importlib

# Dynamic module imports to register all metric classes into the Factory
_iq_mag = importlib.import_module(".IQ Magnitude", package=__name__)
_env_mag = importlib.import_module(".Envelope Magnitude", package=__name__)
_peak_amp = importlib.import_module(".Peak Amplitude", package=__name__)
_rms = importlib.import_module(".RMS", package=__name__)
_avg_amp = importlib.import_module(".Average Amplitude", package=__name__)
_power = importlib.import_module(".Power", package=__name__)
_peak_pwr = importlib.import_module(".Peak Power", package=__name__)
_avg_pwr = importlib.import_module(".Average Power", package=__name__)
_amp_var = importlib.import_module(".Amplitude Variance", package=__name__)
_crest_fac = importlib.import_module(".Crest Factor", package=__name__)
_papr = importlib.import_module(".PAPR", package=__name__)
_dyn_range = importlib.import_module(".Dynamic Range", package=__name__)

# Exported Concrete Metric Classes
IQMagnitudeMetric = _iq_mag.IQMagnitudeMetric
EnvelopeMagnitudeMetric = _env_mag.EnvelopeMagnitudeMetric
PeakAmplitudeMetric = _peak_amp.PeakAmplitudeMetric
RMSMetric = _rms.RMSMetric
AverageAmplitudeMetric = _avg_amp.AverageAmplitudeMetric
PowerMetric = _power.PowerMetric
PeakPowerMetric = _peak_pwr.PeakPowerMetric
AveragePowerMetric = _avg_pwr.AveragePowerMetric
AmplitudeVarianceMetric = _amp_var.AmplitudeVarianceMetric
CrestFactorMetric = _crest_fac.CrestFactorMetric
PAPRMetric = _papr.PAPRMetric
DynamicRangeMetric = _dyn_range.DynamicRangeMetric

# Exported Standalone Functional APIs
envelope_amplitude = _iq_mag.envelope_amplitude
iq_magnitude = _iq_mag.iq_magnitude
envelope_magnitude = _env_mag.envelope_magnitude
peak_amplitude = _peak_amp.peak_amplitude
rms = _rms.rms
rms_amplitude = _rms.rms_amplitude
average_amplitude = _avg_amp.average_amplitude
power = _power.power
iq_power = _power.iq_power
peak_power = _peak_pwr.peak_power
average_power = _avg_pwr.average_power
amplitude_variance = _amp_var.amplitude_variance
crest_factor = _crest_fac.crest_factor
crest_factor_db = _crest_fac.crest_factor_db
papr = _papr.papr
papr_db = _papr.papr_db
dynamic_range = _dyn_range.dynamic_range
dynamic_range_db = _dyn_range.dynamic_range_db


def amplitude_power_metrics(iq_data) -> dict:
    """Return all common scalar amplitude/power metrics in a single dictionary using the Factory."""
    return {
        "peak_amplitude": peak_amplitude(iq_data),
        "rms": rms(iq_data),
        "average_amplitude": average_amplitude(iq_data),
        "peak_power": peak_power(iq_data),
        "average_power": average_power(iq_data),
        "amplitude_variance": amplitude_variance(iq_data),
        "crest_factor": crest_factor(iq_data),
        "crest_factor_db": crest_factor_db(iq_data),
        "papr": papr(iq_data),
        "papr_db": papr_db(iq_data),
        "dynamic_range_db": dynamic_range_db(iq_data),
    }


__all__ = [
    # Factory & Base
    "BaseAmplitudeMetric",
    "AmplitudeMetricFactory",
    # Classes
    "IQMagnitudeMetric",
    "EnvelopeMagnitudeMetric",
    "PeakAmplitudeMetric",
    "RMSMetric",
    "AverageAmplitudeMetric",
    "PowerMetric",
    "PeakPowerMetric",
    "AveragePowerMetric",
    "AmplitudeVarianceMetric",
    "CrestFactorMetric",
    "PAPRMetric",
    "DynamicRangeMetric",
    # Functions
    "_as_iq_array",
    "normalize_iq_input",
    "envelope_amplitude",
    "iq_magnitude",
    "envelope_magnitude",
    "peak_amplitude",
    "rms",
    "rms_amplitude",
    "average_amplitude",
    "power",
    "iq_power",
    "peak_power",
    "average_power",
    "amplitude_variance",
    "crest_factor",
    "crest_factor_db",
    "papr",
    "papr_db",
    "dynamic_range",
    "dynamic_range_db",
    "amplitude_power_metrics",
]
