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


@TimeFrequencyMetricFactory.register(["time_frequency_centroid", "tf_centroid", "time_frequency_center", "tf_center_of_mass"])
class TimeFrequencyCentroidMetric(BaseTimeFrequencyMetric):
    """
    Time-Frequency Centroid Metric.
    Calculates the 2-D center of mass of the linear power spectrogram.

    Mathematical Definition:
      Total TF Energy: E = sum_m sum_k P[k, m]
      Time Centroid:   t_c = sum_m sum_k (t_m * P[k, m]) / E
      Freq Centroid:   f_c = sum_m sum_k (f_k * P[k, m]) / E

    Returns:
      Dictionary containing:
        - 'time_centroid_sec': t_c (in seconds)
        - 'frequency_centroid_hz': f_c (in Hz)
        - 'total_energy': E (linear integrated energy)

    Accepts:
      - Raw complex IQ (runs common STFT once)
      - Precomputed TimeFrequencyResult (reuses linear power spectrogram with zero duplicate FFTs)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "time_frequency_centroid"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
        window_length: int = 256,
        overlap: float | int = 0.75,
        nfft: int | None = None,
        window: str | np.ndarray = "hann",
        as_dict: bool = True,
        eps: float = 1e-18,
        **kwargs: Any,
    ) -> dict[str, Any] | tuple[float, float]:
        """
        Computes 2D time-frequency centroid (t_c, f_c).

        Parameters:
            data: Complex IQ array or precomputed TimeFrequencyResult.
            sample_rate (Fs): Sampling rate in Hz.
            window_length: Window length in samples.
            overlap: Overlap fraction or samples.
            nfft: FFT size.
            window: Window function.
            as_dict: If True, returns dictionary. If False, returns (time_centroid_sec, frequency_centroid_hz).

        Returns:
            Dictionary with centroid metrics or (t_c, f_c) tuple.
        """
        if isinstance(data, TimeFrequencyResult):
            tf_res = data
        elif isinstance(data, dict) and "power_spectrogram" in data and "frequencies_hz" in data:
            p_linear = np.asarray(data["power_spectrogram"], dtype=np.float64)
            freqs = np.asarray(data["frequencies_hz"], dtype=np.float64)
            times = np.asarray(data.get("times_sec", np.arange(p_linear.shape[1])), dtype=np.float64)
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

        if p_linear.size == 0 or p_linear.shape[1] == 0:
            res_dict = {
                "time_centroid_sec": 0.0,
                "frequency_centroid_hz": 0.0,
                "total_energy": 0.0,
            }
            return res_dict if as_dict else (0.0, 0.0)

        # Total energy E = sum_{k, m} P[k, m]
        total_e = float(np.sum(p_linear))
        den = total_e if total_e > eps else eps

        # Time centroid: sum_{m, k} (t_m * P[k, m]) / E
        # sum_k P[k, m] is the marginal power vs time
        time_marginal = np.sum(p_linear, axis=0)  # Shape: [M_frames]
        tc = float(np.sum(times * time_marginal) / den)

        # Frequency centroid: sum_{m, k} (f_k * P[k, m]) / E
        # sum_m P[k, m] is the marginal power vs frequency
        freq_marginal = np.sum(p_linear, axis=1)  # Shape: [K_bins]
        fc = float(np.sum(freqs * freq_marginal) / den)

        if as_dict:
            return {
                "time_centroid_sec": tc,
                "frequency_centroid_hz": fc,
                "total_energy": total_e,
            }

        return (tc, fc)


def compute_time_frequency_centroid(
    data: Any,
    *,
    sample_rate: float | None = None,
    window_length: int = 256,
    overlap: float | int = 0.75,
    nfft: int | None = None,
    window: str | np.ndarray = "hann",
    as_dict: bool = True,
    **kwargs: Any,
) -> dict[str, Any] | tuple[float, float]:

    """
    Convenience functional API for time-frequency centroid computation.
    """
    metric = TimeFrequencyCentroidMetric()
    return metric.compute(
        data,
        sample_rate=sample_rate,
        window_length=window_length,
        overlap=overlap,
        nfft=nfft,
        window=window,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  TimeFrequencyCentroidMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = TimeFrequencyCentroidMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
