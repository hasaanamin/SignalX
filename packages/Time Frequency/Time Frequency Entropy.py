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


@TimeFrequencyMetricFactory.register(["time_frequency_entropy", "tf_entropy", "spectrogram_entropy", "spectral_entropy_tf"])
class TimeFrequencyEntropyMetric(BaseTimeFrequencyMetric):
    """
    Time-Frequency Entropy Metric following MATLAB `spectralEntropy` definition.
    Normalizes spectral power into a probability distribution per frame and computes Shannon entropy.

    MATLAB Formulation:
      p[k, m] = P[k, m] / sum_j P[j, m]
      H[m] = -sum_k p[k, m] * log2(p[k, m])
      H_N[m] = H[m] / log2(N_f)  (normalized form in [0, 1])
      where N_f is the number of frequency bins.

    Returns:
      - 'tf_entropy_normalized': H_N[m] (in [0, 1])
      - 'tf_entropy_raw': H[m] (in bits)
      - 'times_sec': timestamps associated with STFT frames

    Accepts:
      - Raw complex IQ (runs common STFT once)
      - Precomputed TimeFrequencyResult (reuses linear power spectrogram with zero duplicate FFTs)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "time_frequency_entropy"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
        window_length: int = 256,
        overlap: float | int = 0.75,
        nfft: int | None = None,
        window: str | np.ndarray = "hann",
        normalized: bool = True,
        as_dict: bool = False,
        eps: float = 1e-18,
        **kwargs: Any,
    ) -> np.ndarray | dict[str, Any]:
        """
        Computes time-frequency Shannon entropy per STFT frame.

        Parameters:
            data: Complex IQ array or precomputed TimeFrequencyResult.
            sample_rate (Fs): Sampling rate in Hz.
            window_length: Window length in samples.
            overlap: Overlap fraction or samples.
            nfft: FFT size.
            window: Window function.
            normalized: If True (default), returns normalized entropy in [0, 1]. If False, returns raw entropy in bits.
            as_dict: If True, returns dictionary with normalized & raw entropy, times, and statistics.

        Returns:
            1D array of instantaneous entropy or canonical dictionary.
        """
        if isinstance(data, TimeFrequencyResult):
            tf_res = data
        elif isinstance(data, dict) and "power_spectrogram" in data and "frequencies_hz" in data:
            p_linear = np.asarray(data["power_spectrogram"], dtype=np.float64)
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
            times = tf_res.times_sec

        if p_linear.size == 0 or p_linear.shape[1] == 0:
            empty_arr = np.empty(0, dtype=np.float64)
            return {
                "tf_entropy_normalized": empty_arr,
                "tf_entropy_raw": empty_arr,
                "times_sec": times,
            } if as_dict else empty_arr

        n_bins = p_linear.shape[0]
        max_entropy = math.log2(n_bins) if n_bins > 1 else 1.0

        # Normalization into probability distribution per frame: p[k, m] = P[k, m] / sum_j P[j, m]
        sum_p = np.sum(p_linear, axis=0, keepdims=True) + eps
        prob = p_linear / sum_p

        # Shannon entropy: H[m] = -sum_k p[k, m] * log2(p[k, m] + eps)
        h_raw = -np.sum(prob * np.log2(np.maximum(1e-300, prob)), axis=0)

        # Normalized entropy: H_N[m] = H[m] / log2(N_f), bounded in [0, 1]
        h_norm = np.clip(h_raw / max_entropy, 0.0, 1.0)

        if as_dict:
            return {
                "tf_entropy_normalized": h_norm,
                "tf_entropy_raw": h_raw,
                "times_sec": times,
                "mean_normalized_entropy": float(np.mean(h_norm)) if len(h_norm) > 0 else 0.0,
                "min_normalized_entropy": float(np.min(h_norm)) if len(h_norm) > 0 else 0.0,
                "max_normalized_entropy": float(np.max(h_norm)) if len(h_norm) > 0 else 0.0,
                "num_frequency_bins": n_bins,
            }

        return h_norm if normalized else h_raw


def compute_time_frequency_entropy(
    data: Any,
    *,
    sample_rate: float | None = None,
    window_length: int = 256,
    overlap: float | int = 0.75,
    nfft: int | None = None,
    window: str | np.ndarray = "hann",
    normalized: bool = True,
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:

    """
    Convenience functional API for time-frequency Shannon entropy computation.
    """
    metric = TimeFrequencyEntropyMetric()
    return metric.compute(
        data,
        sample_rate=sample_rate,
        window_length=window_length,
        overlap=overlap,
        nfft=nfft,
        window=window,
        normalized=normalized,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  TimeFrequencyEntropyMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = TimeFrequencyEntropyMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
