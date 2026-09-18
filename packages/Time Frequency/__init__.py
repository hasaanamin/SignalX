from __future__ import annotations
from typing import Any
import numpy as np

from .base import (
    BaseTimeFrequencyMetric,
    TimeFrequencyMetricFactory,
    TimeFrequencyResult,
    WaveletResult,
)
from ._common import (
    CWTFilterBank,
    calculate_morse_support_samples,
    _as_iq_array,
    _extract_sample_rate,
    _compute_stft_core,
    _compute_cwt_core,
)
from .STFT import STFTStage, compute_stft
from .Spectrogram import SpectrogramMetric, compute_spectrogram
from .Frequency_Trajectory import FrequencyTrajectoryMetric, compute_frequency_trajectory
from .Instantaneous_Bandwidth import InstantaneousBandwidthMetric, compute_instantaneous_bandwidth
from .Time_Frequency_Centroid import TimeFrequencyCentroidMetric, compute_time_frequency_centroid
from .Time_Frequency_Entropy import TimeFrequencyEntropyMetric, compute_time_frequency_entropy
from .Wavelet_Transform import WaveletTransformStage, compute_cwt
from .Wavelet_Coefficients import WaveletCoefficientsMetric, compute_wavelet_coefficients
from .Wavelet_Energy import WaveletEnergyMetric, compute_wavelet_energy
from .streaming import StreamingSTFT, StreamingCWT


def time_frequency_summary(
    iq_data: Any,
    *,
    sample_rate: float | None = None,
    window_length: int = 256,
    overlap: float | int = 0.75,
    nfft: int | None = None,
    window: str | np.ndarray = "hann",
    trajectory_mode: str = "tfmoment",
    include_cwt: bool = True,
    wavelet_name: str = "morse",
    voices_per_octave: int = 10,
    gamma: float = 3.0,
    time_bandwidth: float = 60.0,
    min_frequency_hz: float | None = None,
    max_frequency_hz: float | None = None,
    frequency_limits: tuple[float, float] | None = None,
    scales: np.ndarray | None = None,
    normalization: str = "L1",
    frequency_mode: str = "analytic_positive",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Unified High-Performance Time-Frequency Analysis Pipeline.
    Executes the shared STFT engine once and the shared CWT engine once,
    deriving all 9 time-frequency features with zero redundant transform steps:

      1. STFT (Complex Matrix)
      2. Spectrogram (Linear Power & dBFS)
      3. Frequency Trajectory (First Conditional Spectral Moment or Ridge)
      4. Instantaneous Bandwidth (Second Centralized Moment * sqrt(4*pi))
      5. Time-Frequency Centroid (2D Center of Mass)
      6. Time-Frequency Entropy (Instantaneous Shannon Entropy per Frame)
      7. Continuous Wavelet Transform (Morse CWT)
      8. Wavelet Coefficients (Complex Multiscale Matrix)
      9. Wavelet Energy (Scalogram, Scale Energy, Time Energy, Total Energy)

    Parameters:
        iq_data: Complex IQ input array or dictionary.
        sample_rate (Fs): Sampling rate in Hz.
        window_length: STFT window size (default: 256).
        overlap: STFT overlap (default: 0.75 / 75%).
        nfft: STFT FFT size (default: max(256, window_length)).
        window: STFT window type (default: "hann").
        trajectory_mode: 'tfmoment' (default) or 'ridge'.
        include_cwt: If True (default), computes CWT and wavelet metrics.
        wavelet_name: Wavelet family (default: "morse").
        voices_per_octave: CWT scales per octave (default: 10).
        gamma: Morse wavelet gamma (default: 3.0).
        time_bandwidth: Morse time-bandwidth (default: 60.0).
        min_frequency_hz: Optional lower frequency bound in Hz.
        max_frequency_hz: Optional upper frequency bound in Hz.
        frequency_limits: Optional tuple (f_min, f_max).
        scales: Optional pre-specified scales array.
        normalization: CWT normalization ('L1' or 'L2').
        frequency_mode: CWT frequency mode ('analytic_positive' or 'two_sided').

    Returns:
        Comprehensive dictionary containing all computed transforms, matrices, profiles, and summary metrics.
    """
    fs = _extract_sample_rate(data=iq_data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
    z = _as_iq_array(iq_data)

    # 1. Common STFT Engine
    stft_res = _compute_stft_core(
        z,
        sample_rate=fs,
        window_length=window_length,
        overlap=overlap,
        nfft=nfft,
        window=window,
    )

    # 2. Downstream STFT Metrics (derived from stft_res with zero duplicate FFTs)
    freq_traj_metric = FrequencyTrajectoryMetric()
    freq_traj = freq_traj_metric.compute(stft_res, trajectory_mode=trajectory_mode, as_dict=True)

    ibw_metric = InstantaneousBandwidthMetric()
    ibw = ibw_metric.compute(stft_res, as_dict=True)

    tf_centroid_metric = TimeFrequencyCentroidMetric()
    tf_centroid = tf_centroid_metric.compute(stft_res, as_dict=True)

    tf_entropy_metric = TimeFrequencyEntropyMetric()
    tf_entropy = tf_entropy_metric.compute(stft_res, as_dict=True)

    out: dict[str, Any] = {
        "sample_rate_hz": fs,
        "num_samples": len(z),
        "stft": {
            "complex": stft_res.stft_complex,
            "frequencies_hz": stft_res.frequencies_hz,
            "times_sec": stft_res.times_sec,
            "num_frames": len(stft_res.times_sec),
            "num_bins": len(stft_res.frequencies_hz),
            "window_length": stft_res.window_length,
            "hop_samples": stft_res.hop_samples,
            "nfft": stft_res.nfft,
        },
        "spectrogram": {
            "power_linear": stft_res.power_spectrogram,
            "power_db": stft_res.spectrogram_db,
            "frequencies_hz": stft_res.frequencies_hz,
            "times_sec": stft_res.times_sec,
        },
        "frequency_trajectory": freq_traj,
        "instantaneous_bandwidth": ibw,
        "time_frequency_centroid": tf_centroid,
        "time_frequency_entropy": tf_entropy,
    }

    # 3. Common CWT Engine & Wavelet Metrics
    if include_cwt:
        wav_res = _compute_cwt_core(
            z,
            sample_rate=fs,
            wavelet_name=wavelet_name,
            voices_per_octave=voices_per_octave,
            gamma=gamma,
            time_bandwidth=time_bandwidth,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
            frequency_limits=frequency_limits,
            scales=scales,
            normalization=normalization,
            frequency_mode=frequency_mode,
        )

        out["wavelet_transform"] = {
            "coefficients": wav_res.coefficients,
            "frequencies_hz": wav_res.frequencies_hz,
            "scales": wav_res.scales,
            "times_sec": wav_res.times_sec,
            "wavelet_name": wav_res.wavelet_name,
            "voices_per_octave": wav_res.voices_per_octave,
            "gamma": wav_res.gamma,
            "time_bandwidth": wav_res.time_bandwidth,
            "normalization": wav_res.normalization,
            "frequency_mode": wav_res.frequency_mode,
        }
        out["wavelet_coefficients"] = wav_res.coefficients
        out["wavelet_energy"] = {
            "coefficient_energy_map": wav_res.coefficient_energy,
            "energy_per_scale": wav_res.energy_per_scale,
            "energy_vs_time": wav_res.energy_vs_time,
            "total_energy": wav_res.total_energy,
            "frequencies_hz": wav_res.frequencies_hz,
            "scales": wav_res.scales,
            "times_sec": wav_res.times_sec,
        }

    return out



__all__ = [
    "BaseTimeFrequencyMetric",
    "TimeFrequencyMetricFactory",
    "TimeFrequencyResult",
    "WaveletResult",
    "CWTFilterBank",
    "calculate_morse_support_samples",
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
