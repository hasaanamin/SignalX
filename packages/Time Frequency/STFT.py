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
from ._common import _as_iq_array, _extract_sample_rate, _compute_stft_core


@TimeFrequencyMetricFactory.register(["stft", "short_time_fourier_transform", "short_time_ft", "discrete_stft"])
class STFTStage(BaseTimeFrequencyMetric):
    """
    Short-Time Fourier Transform (STFT) Stage matching MATLAB `stft` definition.
    Computes a two-sided centered spectrum [-Fs/2, Fs/2) for complex IQ with window center timestamps.

    MATLAB Formulation:
      X_m(f) = sum_{n=-inf}^inf x[n] g[n - mR] e^{-j 2 pi f n}
      Discrete: X[k, m] = sum_{r=0}^{M-1} x[mR + r] w[r] e^{-j 2 pi k r / N_fft}
      Frequency bins: f_k = (k - N_fft/2) * (Fs / N_fft) via fftshift
      Timestamps: t_m = (n_{start, m} + (M - 1)/2) / Fs

    Returns:
      TimeFrequencyResult container with:
        - stft_complex [K, M_frames] (complex128)
        - power_spectrogram [K, M_frames] (|X|^2)
        - spectrogram_db [K, M_frames] (10*log10(P + eps))
        - frequencies_hz [K]
        - times_sec [M_frames]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "stft"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
        window_length: int = 256,
        overlap: float | int = 0.75,
        nfft: int | None = None,
        window: str | np.ndarray = "hann",
        sample_offset: int = 0,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> TimeFrequencyResult | dict[str, Any] | np.ndarray:
        """
        Executes STFT on complex IQ data.

        Parameters:
            data: Complex IQ input array, list, tuple, or dictionary.
            sample_rate (Fs): Sampling rate in Hz.
            window_length (M): Window length in samples (default: 256).
            overlap (L): Overlap fraction in [0, 1) or integer overlap samples (default: 0.75 / 75%).
            nfft (N_fft): FFT size (>= window_length, default: max(window_length, 256)).
            window: Window function name ("hann", "hamming", "blackman", "flattop", "rect") or array.
            sample_offset: Global sample offset for timestamp alignment in streaming/slicing.
            as_dict: If True, returns canonical dictionary. If False, returns TimeFrequencyResult.

        Returns:
            TimeFrequencyResult (or dict if as_dict=True).
        """
        if isinstance(data, TimeFrequencyResult):
            return data.to_dict() if as_dict else data

        fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
        res = _compute_stft_core(
            data,
            sample_rate=fs,
            window_length=window_length,
            overlap=overlap,
            nfft=nfft,
            window=window,
            sample_offset=sample_offset,
        )
        return res.to_dict() if as_dict else res


def compute_stft(
    iq_data: Any,
    *,
    sample_rate: float | None = None,
    window_length: int = 256,
    overlap: float | int = 0.75,
    nfft: int | None = None,
    window: str | np.ndarray = "hann",
    sample_offset: int = 0,
    as_dict: bool = False,
    **kwargs: Any,
) -> TimeFrequencyResult | dict[str, Any]:

    """
    Convenience functional API to compute discrete STFT on complex IQ data.
    """
    stage = STFTStage()
    return stage.compute(
        iq_data,
        sample_rate=sample_rate,
        window_length=window_length,
        overlap=overlap,
        nfft=nfft,
        window=window,
        sample_offset=sample_offset,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  STFTStage: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = STFTStage()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
