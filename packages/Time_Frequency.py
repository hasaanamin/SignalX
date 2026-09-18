from __future__ import annotations
import importlib

_pkg = importlib.import_module("packages.Time Frequency")

BaseTimeFrequencyMetric = _pkg.BaseTimeFrequencyMetric
TimeFrequencyMetricFactory = _pkg.TimeFrequencyMetricFactory
TimeFrequencyResult = _pkg.TimeFrequencyResult
WaveletResult = _pkg.WaveletResult
CWTFilterBank = _pkg.CWTFilterBank
calculate_morse_support_samples = _pkg.calculate_morse_support_samples

STFTStage = _pkg.STFTStage
compute_stft = _pkg.compute_stft

SpectrogramMetric = _pkg.SpectrogramMetric
compute_spectrogram = _pkg.compute_spectrogram

FrequencyTrajectoryMetric = _pkg.FrequencyTrajectoryMetric
compute_frequency_trajectory = _pkg.compute_frequency_trajectory

InstantaneousBandwidthMetric = _pkg.InstantaneousBandwidthMetric
compute_instantaneous_bandwidth = _pkg.compute_instantaneous_bandwidth

TimeFrequencyCentroidMetric = _pkg.TimeFrequencyCentroidMetric
compute_time_frequency_centroid = _pkg.compute_time_frequency_centroid

TimeFrequencyEntropyMetric = _pkg.TimeFrequencyEntropyMetric
compute_time_frequency_entropy = _pkg.compute_time_frequency_entropy

WaveletTransformStage = _pkg.WaveletTransformStage
compute_cwt = _pkg.compute_cwt

WaveletCoefficientsMetric = _pkg.WaveletCoefficientsMetric
compute_wavelet_coefficients = _pkg.compute_wavelet_coefficients

WaveletEnergyMetric = _pkg.WaveletEnergyMetric
compute_wavelet_energy = _pkg.compute_wavelet_energy

StreamingSTFT = _pkg.StreamingSTFT
StreamingCWT = _pkg.StreamingCWT

time_frequency_summary = _pkg.time_frequency_summary

__all__ = [
    "BaseTimeFrequencyMetric",
    "TimeFrequencyMetricFactory",
    "TimeFrequencyResult",
    "WaveletResult",
    "CWTFilterBank",
    "STFTStage",
    "compute_stft",
    "SpectrogramMetric",
    "compute_spectrogram",
    "FrequencyTrajectoryMetric",
    "compute_frequency_trajectory",
    "InstantaneousBandwidthMetric",
    "compute_instantaneous_bandwidth",
    "TimeFrequencyCentroidMetric",
    "compute_time_frequency_centroid",
    "TimeFrequencyEntropyMetric",
    "compute_time_frequency_entropy",
    "WaveletTransformStage",
    "compute_cwt",
    "WaveletCoefficientsMetric",
    "compute_wavelet_coefficients",
    "WaveletEnergyMetric",
    "compute_wavelet_energy",
    "StreamingSTFT",
    "StreamingCWT",
    "time_frequency_summary",
]
