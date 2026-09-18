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
from .base import BaseTimeFrequencyMetric, TimeFrequencyMetricFactory, TimeFrequencyResult
from ._common import _extract_sample_rate, _compute_stft_core


@TimeFrequencyMetricFactory.register(["frequency_trajectory", "instfreq", "instantaneous_frequency", "freq_trajectory", "frequency_track"])
class FrequencyTrajectoryMetric(BaseTimeFrequencyMetric):
    """
    Frequency Trajectory Metric matching MATLAB `instfreq` definition.
    Calculates the first noncentralized conditional spectral moment of the time-frequency distribution.

    MATLAB Formulation:
      f_inst[m] = sum_k (f_k * P[k, m]) / sum_k P[k, m]
      Computed over centered frequency grid [-Fs/2, Fs/2) for complex IQ signals.

    Modes:
      - 'tfmoment' (default): MATLAB first conditional spectral moment.
      - 'ridge': Peak spectral tracking via argmax_k(P[k, m]).

    Accepts:
      - Raw complex IQ (runs common STFT once)
      - Precomputed TimeFrequencyResult (reuses linear power spectrogram with zero duplicate FFTs)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "frequency_trajectory"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
        window_length: int = 256,
        overlap: float | int = 0.75,
        nfft: int | None = None,
        window: str | np.ndarray = "hann",
        trajectory_mode: str = "tfmoment",
        as_dict: bool = False,
        eps: float = 1e-18,
        **kwargs: Any,
    ) -> np.ndarray | dict[str, Any]:
        """
        Computes frequency trajectory across time frames.

        Parameters:
            data: Complex IQ array or precomputed TimeFrequencyResult.
            sample_rate (Fs): Sampling rate in Hz.
            window_length: Window length in samples.
            overlap: Overlap fraction or samples.
            nfft: FFT size.
            window: Window function.
            trajectory_mode: 'tfmoment' (default, MATLAB first moment) or 'ridge' (argmax peak).
            as_dict: If True, returns dictionary with trajectory, timestamps, and frequencies.

        Returns:
            1D array of instantaneous frequencies in Hz or dictionary.
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
            empty_traj = np.empty(0, dtype=np.float64)
            return {"frequency_trajectory_hz": empty_traj, "times_sec": times, "mode": trajectory_mode} if as_dict else empty_traj

        mode_clean = str(trajectory_mode).lower().strip().replace("-", "_")
        if mode_clean in ("ridge", "peak", "argmax"):
            # Ridge tracking via spectral maximum
            max_bin_indices = np.argmax(p_linear, axis=0)
            f_inst = freqs[max_bin_indices]
        else:
            # MATLAB default: First conditional spectral moment
            den = np.sum(p_linear, axis=0) + eps
            num = np.sum(freqs[:, None] * p_linear, axis=0)
            f_inst = num / den

        if as_dict:
            return {
                "frequency_trajectory_hz": f_inst,
                "times_sec": times,
                "sample_rate": fs,
                "trajectory_mode": mode_clean,
                "mean_frequency_hz": float(np.mean(f_inst)) if len(f_inst) > 0 else 0.0,
                "min_frequency_hz": float(np.min(f_inst)) if len(f_inst) > 0 else 0.0,
                "max_frequency_hz": float(np.max(f_inst)) if len(f_inst) > 0 else 0.0,
            }

        return f_inst


def compute_frequency_trajectory(
    data: Any,
    *,
    sample_rate: float | None = None,
    window_length: int = 256,
    overlap: float | int = 0.75,
    nfft: int | None = None,
    window: str | np.ndarray = "hann",
    trajectory_mode: str = "tfmoment",
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:

    """
    Convenience functional API for frequency trajectory computation.
    """
    metric = FrequencyTrajectoryMetric()
    return metric.compute(
        data,
        sample_rate=sample_rate,
        window_length=window_length,
        overlap=overlap,
        nfft=nfft,
        window=window,
        trajectory_mode=trajectory_mode,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  FrequencyTrajectoryMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FrequencyTrajectoryMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
