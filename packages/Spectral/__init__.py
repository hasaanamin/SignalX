"""
Spectral & Frequency-Domain Analysis Package (Factory & Self-Contained Class Architecture)
Provides enterprise Factory Pattern and Abstract Base Classes (ABCs) for computing:
  - Complex FFT Spectrum (Magnitude, Phase, dBFS, Centered Frequency)
  - Power Spectral Density (PSD) (Welch Averaged Modified Periodogram)
  - High-Resolution Spectrum Estimation (Welch, Burg AR, MUSIC)
  - Power Bandwidth (3-dB, 10-dB, 20-dB Bandwidth)
  - Occupied Bandwidth (99% / Beta% Power OBW)
  - Spectral Peak Detection (Carrier Peak, Prominence, Multi-Peak)
  - Spectral Centroid (First Spectral Moment / Spectral Center of Mass)
  - Spectral Variance & Spectral Spread (Second Central Moment)
  - Spectral Skewness (Third Standardized Spectral Moment)
  - Spectral Kurtosis (Fourth Standardized Moment & Transient SK(f))
  - Spectral Shannon Entropy (Disorder / Complexity)
  - Spectral Flatness / Wiener Entropy (Tonality Factor)
  - Spectral Roll-off Frequency (85% / 95% Energy Cutoff)
  - Sideband Power (USB / LSB Power & Asymmetry Ratio)
"""

from .base import BaseSpectralMetric, SpectralMetricFactory
from ._common import _as_iq_array, _extract_sample_rate

import importlib

# Dynamic module imports to register all metric classes into the Factory
_fft_mod = importlib.import_module(".FFT", package=__name__)
_psd_mod = importlib.import_module(".PSD", package=__name__)
_spec_est_mod = importlib.import_module(".Spectrum Estimation", package=__name__)
_bw_mod = importlib.import_module(".Bandwidth", package=__name__)
_obw_mod = importlib.import_module(".Occupied Bandwidth", package=__name__)
_peak_mod = importlib.import_module(".Spectral Peak", package=__name__)
_centroid_mod = importlib.import_module(".Spectral Centroid", package=__name__)
_var_mod = importlib.import_module(".Spectral Variance", package=__name__)
_skew_mod = importlib.import_module(".Spectral Skewness", package=__name__)
_kurt_mod = importlib.import_module(".Spectral Kurtosis", package=__name__)
_entropy_mod = importlib.import_module(".Spectral Entropy", package=__name__)
_flatness_mod = importlib.import_module(".Spectral Flatness", package=__name__)
_rolloff_mod = importlib.import_module(".Spectral Roll-off", package=__name__)
_sideband_mod = importlib.import_module(".Sideband Power", package=__name__)

# Exported Concrete Metric Classes
FFTSpectrumMetric = _fft_mod.FFTSpectrumMetric
PSDMetric = _psd_mod.PSDMetric
SpectrumEstimationMetric = _spec_est_mod.SpectrumEstimationMetric
BandwidthMetric = _bw_mod.BandwidthMetric
OccupiedBandwidthMetric = _obw_mod.OccupiedBandwidthMetric
SpectralPeakMetric = _peak_mod.SpectralPeakMetric
SpectralCentroidMetric = _centroid_mod.SpectralCentroidMetric
SpectralVarianceMetric = _var_mod.SpectralVarianceMetric
SpectralSkewnessMetric = _skew_mod.SpectralSkewnessMetric
SpectralKurtosisMetric = _kurt_mod.SpectralKurtosisMetric
SpectralEntropyMetric = _entropy_mod.SpectralEntropyMetric
SpectralFlatnessMetric = _flatness_mod.SpectralFlatnessMetric
SpectralRolloffMetric = _rolloff_mod.SpectralRolloffMetric
SidebandPowerMetric = _sideband_mod.SidebandPowerMetric

# Exported Standalone Functional APIs
fft_spectrum = _fft_mod.fft_spectrum
psd = _psd_mod.psd
spectrum_estimation = _spec_est_mod.spectrum_estimation
bandwidth = _bw_mod.bandwidth
occupied_bandwidth = _obw_mod.occupied_bandwidth
spectral_peak = _peak_mod.spectral_peak
spectral_centroid = _centroid_mod.spectral_centroid
spectral_variance = _var_mod.spectral_variance
spectral_skewness = _skew_mod.spectral_skewness
spectral_kurtosis = _kurt_mod.spectral_kurtosis
spectral_entropy = _entropy_mod.spectral_entropy
spectral_flatness = _flatness_mod.spectral_flatness
spectral_rolloff = _rolloff_mod.spectral_rolloff
sideband_power = _sideband_mod.sideband_power


def spectral_summary(iq_data, sample_rate: float | None = None) -> dict:
    """
    Computes a comprehensive summary of spectral and frequency-domain characteristics.
    """
    fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
    
    psd_res = psd(iq_data, sample_rate=fs, as_dict=True)
    bw_res = bandwidth(iq_data, sample_rate=fs, as_dict=True)
    obw_res = occupied_bandwidth(iq_data, sample_rate=fs, as_dict=True)
    peak_res = spectral_peak(iq_data, sample_rate=fs, as_dict=True)
    centroid_res = spectral_centroid(iq_data, sample_rate=fs, as_dict=True)
    var_res = spectral_variance(iq_data, sample_rate=fs, as_dict=True)
    skew_res = spectral_skewness(iq_data, sample_rate=fs, as_dict=True)
    kurt_res = spectral_kurtosis(iq_data, sample_rate=fs, as_dict=True)
    entropy_res = spectral_entropy(iq_data, sample_rate=fs, as_dict=True)
    flatness_res = spectral_flatness(iq_data, sample_rate=fs, as_dict=True)
    rolloff_res = spectral_rolloff(iq_data, sample_rate=fs, as_dict=True)
    sideband_res = sideband_power(iq_data, sample_rate=fs, as_dict=True)

    return {
        "psd": psd_res,
        "bandwidth_3db": bw_res,
        "occupied_bandwidth_99": obw_res,
        "spectral_peak": peak_res,
        "spectral_centroid": centroid_res,
        "spectral_variance": var_res,
        "spectral_skewness": skew_res,
        "spectral_kurtosis": kurt_res,
        "spectral_entropy": entropy_res,
        "spectral_flatness": flatness_res,
        "spectral_rolloff": rolloff_res,
        "sideband_power": sideband_res,
    }


__all__ = [
    "BaseSpectralMetric",
    "SpectralMetricFactory",
    "FFTSpectrumMetric",
    "PSDMetric",
    "SpectrumEstimationMetric",
    "BandwidthMetric",
    "OccupiedBandwidthMetric",
    "SpectralPeakMetric",
    "SpectralCentroidMetric",
    "SpectralVarianceMetric",
    "SpectralSkewnessMetric",
    "SpectralKurtosisMetric",
    "SpectralEntropyMetric",
    "SpectralFlatnessMetric",
    "SpectralRolloffMetric",
    "SidebandPowerMetric",
    "fft_spectrum",
    "psd",
    "spectrum_estimation",
    "bandwidth",
    "occupied_bandwidth",
    "spectral_peak",
    "spectral_centroid",
    "spectral_variance",
    "spectral_skewness",
    "spectral_kurtosis",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_rolloff",
    "sideband_power",
    "spectral_summary",
]
