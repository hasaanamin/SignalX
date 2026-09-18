from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

from typing import Any
import numpy as np
from .base import BaseTimeFrequencyMetric, TimeFrequencyMetricFactory, WaveletResult
from ._common import _extract_sample_rate, _compute_cwt_core


@TimeFrequencyMetricFactory.register(["wavelet_coefficients", "cwt_coefficients", "wavelet_coefs", "cwt_coefs"])
class WaveletCoefficientsMetric(BaseTimeFrequencyMetric):
    """
    Wavelet Coefficients Stage.
    Exposes and formats the complex CWT coefficient matrix W[s, n] across scales and time.

    Accepts:
      - Raw complex IQ (runs common CWT once)
      - Precomputed WaveletResult (reuses precalculated CWT coefficients with zero duplicate transform steps)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "wavelet_coefficients"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
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
        as_dict: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | dict[str, Any]:
        """
        Extracts or computes complex CWT coefficients.

        Parameters:
            data: Complex IQ array or existing WaveletResult.
            sample_rate (Fs): Sampling rate in Hz.
            wavelet_name: Wavelet type ('morse').
            voices_per_octave: Voices per octave.
            gamma: Morse gamma.
            time_bandwidth: Morse time-bandwidth.
            min_frequency_hz: Optional lower frequency bound in Hz.
            max_frequency_hz: Optional upper frequency bound in Hz.
            frequency_limits: Optional tuple (f_min, f_max).
            scales: Optional pre-specified scales array.
            normalization: 'L1' or 'L2'.
            frequency_mode: 'analytic_positive' or 'two_sided'.
            as_dict: If True, returns dictionary with coefficients, scales, frequencies, and times.

        Returns:
            2D complex ndarray [S, N] or dictionary.
        """
        if isinstance(data, WaveletResult):
            wav_res = data
        elif isinstance(data, dict) and "coefficients" in data and "frequencies_hz" in data:
            coeffs = np.asarray(data["coefficients"], dtype=np.complex128)
            freqs = np.asarray(data["frequencies_hz"], dtype=np.float64)
            scales = np.asarray(data.get("scales", np.arange(len(freqs))), dtype=np.float64)
            times = np.asarray(data.get("times_sec", np.arange(coeffs.shape[1])), dtype=np.float64)
            if as_dict:
                return {
                    "coefficients": coeffs,
                    "frequencies_hz": freqs,
                    "scales": scales,
                    "times_sec": times,
                    "sample_rate": data.get("sample_rate", 1.0),
                }
            return coeffs
        else:
            fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
            wav_res = _compute_cwt_core(
                data,
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

        if as_dict:
            return {
                "coefficients": wav_res.coefficients,
                "frequencies_hz": wav_res.frequencies_hz,
                "scales": wav_res.scales,
                "times_sec": wav_res.times_sec,
                "sample_rate": wav_res.sample_rate,
                "wavelet_name": wav_res.wavelet_name,
                "voices_per_octave": wav_res.voices_per_octave,
            }

        return wav_res.coefficients


def compute_wavelet_coefficients(
    data: Any,
    *,
    sample_rate: float | None = None,
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
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:
    """
    Convenience functional API for wavelet coefficients extraction.
    """
    metric = WaveletCoefficientsMetric()
    return metric.compute(
        data,
        sample_rate=sample_rate,
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
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  WaveletCoefficientsMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = WaveletCoefficientsMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
