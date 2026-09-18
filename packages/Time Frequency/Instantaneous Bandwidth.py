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
import math
import numpy as np
from .base import BaseTimeFrequencyMetric, TimeFrequencyMetricFactory, TimeFrequencyResult
from ._common import _extract_sample_rate, _compute_stft_core


@TimeFrequencyMetricFactory.register(["instantaneous_bandwidth", "instbw", "instantaneous_bw", "ibw"])
class InstantaneousBandwidthMetric(BaseTimeFrequencyMetric):
    """
    Instantaneous Bandwidth Metric matching MATLAB `instbw` definition.
    Calculates the second centralized conditional spectral moment of the time-frequency distribution.

    MATLAB Formulation:
      f_inst[m] = sum_k (f_k * P[k, m]) / sum_k P[k, m]
      sigma_f^2[m] = sum_k ((f_k - f_inst[m])^2 * P[k, m]) / sum_k P[k, m]
      sigma_f[m] = sqrt(sigma_f^2[m])
      IBW_MATLAB[m] = sqrt(4 * pi) * sigma_f[m]

    Parameters:
      - scale_factor: Multiplier applied to sigma_f (default: sqrt(4*pi) approx 3.5449 matching MATLAB instbw default).
                      Set scale_factor=1.0 to obtain the unscaled standard deviation sigma_f.

    Accepts:
      - Raw complex IQ (runs common STFT once)
      - Precomputed TimeFrequencyResult (reuses linear power spectrogram with zero duplicate FFTs)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "instantaneous_bandwidth"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
        window_length: int = 256,
        overlap: float | int = 0.75,
        nfft: int | None = None,
        window: str | np.ndarray = "hann",
        scale_factor: float | None = None,
        as_dict: bool = False,
        eps: float = 1e-18,
        **kwargs: Any,
    ) -> np.ndarray | dict[str, Any]:
        """
        Computes instantaneous bandwidth across time frames.

        Parameters:
            data: Complex IQ array or precomputed TimeFrequencyResult.
            sample_rate (Fs): Sampling rate in Hz.
            window_length: Window length in samples.
            overlap: Overlap fraction or samples.
            nfft: FFT size.
            window: Window function.
            scale_factor: Bandwidth scaling factor (default: sqrt(4*pi) for MATLAB compatibility).
            as_dict: If True, returns dictionary with bandwidth, standard deviation, and timestamps.

        Returns:
            1D array of instantaneous bandwidth in Hz (or canonical dictionary).
        """
        if isinstance(data, TimeFrequencyResult):
            tf_res = data
        elif isinstance(data, dict) and "power_spectrogram" in data and "frequencies_hz" in data:
            p_linear = np.asarray(data["power_spectrogram"], dtype=np.float64)
            freqs = np.asarray(data["frequencies_hz"], dtype=np.float64)
            times = np.asarray(data.get("times_sec", np.arange(p_linear.shape[1])), dtype=np.float64)
            fs = float(data.get("sample_rate", 1.0))
            tf_res = None
        else:
            fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
            tf_res = _compute_stft_core(
                data,
                sample_rate=fs,
                window_length=window_length,
                overlap=overlap,
                nfft=nfft,
                window=window,
                eps=eps,
            )

        if tf_res is not None:
            p_linear = tf_res.power_spectrogram
            freqs = tf_res.frequencies_hz
            times = tf_res.times_sec
            fs = tf_res.sample_rate

        if p_linear.size == 0 or p_linear.shape[1] == 0:
            empty_arr = np.empty(0, dtype=np.float64)
            return {
                "instantaneous_bandwidth_hz": empty_arr,
                "frequency_std_hz": empty_arr,
                "times_sec": times,
                "scale_factor": scale_factor,
            } if as_dict else empty_arr

        # Determine scale factor (default: sqrt(4*pi))
        c_scale = float(scale_factor) if scale_factor is not None else math.sqrt(4.0 * math.pi)

        # 1. First moment (instantaneous frequency)
        den = np.sum(p_linear, axis=0) + eps
        num = np.sum(freqs[:, None] * p_linear, axis=0)
        f_inst = num / den  # Shape: [M_frames]

        # 2. Second centralized moment (variance)
        diff_f = freqs[:, None] - f_inst[None, :]  # Shape: [K_bins, M_frames]
        var_f = np.sum((diff_f ** 2) * p_linear, axis=0) / den
        sigma_f = np.sqrt(np.maximum(0.0, var_f))

        # 3. Scaled instantaneous bandwidth
        ibw = c_scale * sigma_f

        if as_dict:
            return {
                "instantaneous_bandwidth_hz": ibw,
                "frequency_std_hz": sigma_f,
                "instantaneous_frequency_hz": f_inst,
                "times_sec": times,
                "sample_rate": fs,
                "scale_factor": c_scale,
                "mean_bandwidth_hz": float(np.mean(ibw)) if len(ibw) > 0 else 0.0,
                "min_bandwidth_hz": float(np.min(ibw)) if len(ibw) > 0 else 0.0,
                "max_bandwidth_hz": float(np.max(ibw)) if len(ibw) > 0 else 0.0,
            }

        return ibw


def compute_instantaneous_bandwidth(
    data: Any,
    *,
    sample_rate: float | None = None,
    window_length: int = 256,
    overlap: float | int = 0.75,
    nfft: int | None = None,
    window: str | np.ndarray = "hann",
    scale_factor: float | None = None,
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:

    """
    Convenience functional API for instantaneous bandwidth computation.
    """
    metric = InstantaneousBandwidthMetric()
    return metric.compute(
        data,
        sample_rate=sample_rate,
        window_length=window_length,
        overlap=overlap,
        nfft=nfft,
        window=window,
        scale_factor=scale_factor,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  InstantaneousBandwidthMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = InstantaneousBandwidthMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
