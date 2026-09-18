"""
Noise & Signal Fidelity Analysis Package (Factory & Self-Contained Class Architecture)
Provides enterprise Factory Pattern and Abstract Base Classes (ABCs) for computing:
  - Noise Power (linear, dBW, dBm)
  - Noise Variance (sigma_w^2, sigma_I^2, sigma_Q^2, robust MAD)
  - Spectral Noise Floor Density (dBFS/Hz, dBm/Hz)
  - Signal-to-Noise Ratio (SNR) (Spectral Welch, M2M4, Time residual)
  - Signal-to-Interference-plus-Noise Ratio (SINR)
  - Interference / Jammer Power (P_i, spur detection)
  - Phase Noise (L(Delta f) in dBc/Hz, Phase Jitter in deg and ps)
  - Amplitude Noise / Residual AM (% RMS, Amplitude SNR)
  - Signal Quality Metrics (EVM, MER, SINAD, THD, SFDR, ENOB)
  - Clipping Assessment (hard limiting, ADC rail over-range, PAPR loss)
  - Saturation Assessment (soft compression, AM-AM non-linearity, gain compression)
"""

from .base import BaseNoiseMetric, NoiseMetricFactory
from ._common import _as_iq_array, _extract_sample_rate

import importlib

# Dynamic module imports to register all metric classes into the Factory
_power_mod = importlib.import_module(".Noise Power", package=__name__)
_var_mod = importlib.import_module(".Noise Variance", package=__name__)
_floor_mod = importlib.import_module(".Noise Floor", package=__name__)
_snr_mod = importlib.import_module(".SNR", package=__name__)
_sinr_mod = importlib.import_module(".SINR", package=__name__)
_interf_mod = importlib.import_module(".Interference Power", package=__name__)
_phase_noise_mod = importlib.import_module(".Phase Noise", package=__name__)
_amp_noise_mod = importlib.import_module(".Amplitude Noise", package=__name__)
_quality_mod = importlib.import_module(".Signal Quality Metrics", package=__name__)
_clip_mod = importlib.import_module(".Clipping Assessment", package=__name__)
_sat_mod = importlib.import_module(".Saturation Assessment", package=__name__)

# Exported Concrete Metric Classes
NoisePowerMetric = _power_mod.NoisePowerMetric
NoiseVarianceMetric = _var_mod.NoiseVarianceMetric
NoiseFloorMetric = _floor_mod.NoiseFloorMetric
SNRMetric = _snr_mod.SNRMetric
SINRMetric = _sinr_mod.SINRMetric
InterferencePowerMetric = _interf_mod.InterferencePowerMetric
PhaseNoiseMetric = _phase_noise_mod.PhaseNoiseMetric
AmplitudeNoiseMetric = _amp_noise_mod.AmplitudeNoiseMetric
SignalQualityMetrics = _quality_mod.SignalQualityMetrics
ClippingAssessmentMetric = _clip_mod.ClippingAssessmentMetric
SaturationAssessmentMetric = _sat_mod.SaturationAssessmentMetric

# Exported Standalone Functional APIs
noise_power = _power_mod.noise_power
noise_variance = _var_mod.noise_variance
noise_floor = _floor_mod.noise_floor
snr = _snr_mod.snr
sinr = _sinr_mod.sinr
interference_power = _interf_mod.interference_power
phase_noise = _phase_noise_mod.phase_noise
amplitude_noise = _amp_noise_mod.amplitude_noise
signal_quality_metrics = _quality_mod.signal_quality_metrics
clipping_assessment = _clip_mod.clipping_assessment
saturation_assessment = _sat_mod.saturation_assessment


def noise_summary(iq_data, sample_rate: float | None = None) -> dict:
    """
    Computes a comprehensive noise and signal quality summary across the input signal.
    """
    fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
    
    p_noise_res = noise_power(iq_data, sample_rate=fs, as_dict=True)
    var_res = noise_variance(iq_data, sample_rate=fs, as_dict=True)
    floor_res = noise_floor(iq_data, sample_rate=fs, as_dict=True)
    snr_res = snr(iq_data, sample_rate=fs, as_dict=True)
    sinr_res = sinr(iq_data, sample_rate=fs, as_dict=True)
    interf_res = interference_power(iq_data, sample_rate=fs, as_dict=True)
    pnoise_res = phase_noise(iq_data, sample_rate=fs, as_dict=True)
    anoise_res = amplitude_noise(iq_data, sample_rate=fs, as_dict=True)
    quality_res = signal_quality_metrics(iq_data, sample_rate=fs)
    clip_res = clipping_assessment(iq_data, sample_rate=fs)
    sat_res = saturation_assessment(iq_data, sample_rate=fs)

    return {
        "noise_power": p_noise_res,
        "noise_variance": var_res,
        "noise_floor": floor_res,
        "snr": snr_res,
        "sinr": sinr_res,
        "interference": interf_res,
        "phase_noise": pnoise_res,
        "amplitude_noise": anoise_res,
        "signal_quality": quality_res,
        "clipping": clip_res,
        "saturation": sat_res,
    }


__all__ = [
    "BaseNoiseMetric",
    "NoiseMetricFactory",
    "NoisePowerMetric",
    "NoiseVarianceMetric",
    "NoiseFloorMetric",
    "SNRMetric",
    "SINRMetric",
    "InterferencePowerMetric",
    "PhaseNoiseMetric",
    "AmplitudeNoiseMetric",
    "SignalQualityMetrics",
    "ClippingAssessmentMetric",
    "SaturationAssessmentMetric",
    "noise_power",
    "noise_variance",
    "noise_floor",
    "snr",
    "sinr",
    "interference_power",
    "phase_noise",
    "amplitude_noise",
    "signal_quality_metrics",
    "clipping_assessment",
    "saturation_assessment",
    "noise_summary",
]
