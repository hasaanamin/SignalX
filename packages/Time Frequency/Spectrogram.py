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


@TimeFrequencyMetricFactory.register(["spectrogram", "spectrogram_generation", "power_spectrogram", "pspectrogram"])
class SpectrogramMetric(BaseTimeFrequencyMetric):
    """
    Spectrogram Generation Stage matching MATLAB `spectrogram` definition.
    Calculates the magnitude squared of the STFT: P[k, m] = |X[k, m]|^2.

    Mathematical Definition:
      Linear Power Spectrogram: P[k, m] = |X[k, m]|^2  (used for all downstream mathematical calculations)
      Visualization Spectrogram: P_dB[k, m] = 10 * log10(P[k, m] + eps)

    Accepts:
      - Raw complex IQ (executes common STFT engine once)
      - Precomputed TimeFrequencyResult (reuses precalculated STFT with zero duplicate FFTs)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "spectrogram"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
        window_length: int = 256,
        overlap: float | int = 0.75,
        nfft: int | None = None,
        window: str | np.ndarray = "hann",
        mode: str = "linear",
        as_dict: bool = False,
        eps: float = 1e-18,
        **kwargs: Any,
    ) -> np.ndarray | dict[str, Any]:
        """
        Computes or extracts the spectrogram.

        Parameters:
            data: Complex IQ array, canonical dictionary, or precomputed TimeFrequencyResult.
            sample_rate (Fs): Sampling rate in Hz.
            window_length: Window length in samples.
            overlap: Overlap fraction or integer samples.
            nfft: FFT size.
            window: Window function name.
            mode: 'linear' (P = |X|^2), 'db' / 'dbfs' (10*log10(P + eps)), or 'both' / 'all'.
            as_dict: If True, returns dictionary with frequencies, times, and power matrices.

        Returns:
            2D spectrogram matrix (linear or dB) or canonical dictionary.
        """
        if isinstance(data, TimeFrequencyResult):
            tf_res = data
        elif isinstance(data, dict) and "power_spectrogram" in data and "frequencies_hz" in data:
            # Precomputed dictionary
            p_linear = np.asarray(data["power_spectrogram"], dtype=np.float64)
            p_db = data.get("spectrogram_db", 10.0 * np.log10(p_linear + eps))
            freqs = np.asarray(data["frequencies_hz"], dtype=np.float64)
            times = np.asarray(data.get("times_sec", np.arange(p_linear.shape[1])), dtype=np.float64)
            if as_dict:
                return {
                    "power_spectrogram": p_linear,
                    "spectrogram_db": p_db,
                    "frequencies_hz": freqs,
                    "times_sec": times,
                    "sample_rate": data.get("sample_rate", 1.0),
                }
            mode_clean = str(mode).lower().strip()
            return p_db if mode_clean in ("db", "dbfs", "log") else p_linear
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

        mode_clean = str(mode).lower().strip()
        if as_dict or mode_clean in ("both", "all", "dict"):
            return {
                "power_spectrogram": tf_res.power_spectrogram,
                "spectrogram_db": tf_res.spectrogram_db,
                "frequencies_hz": tf_res.frequencies_hz,
                "times_sec": tf_res.times_sec,
                "sample_rate": tf_res.sample_rate,
                "hop_samples": tf_res.hop_samples,
                "window_length": tf_res.window_length,
                "nfft": tf_res.nfft,
            }

        if mode_clean in ("db", "dbfs", "log"):
            return tf_res.spectrogram_db

        return tf_res.power_spectrogram


def compute_spectrogram(
    data: Any,
    *,
    sample_rate: float | None = None,
    window_length: int = 256,
    overlap: float | int = 0.75,
    nfft: int | None = None,
    window: str | np.ndarray = "hann",
    mode: str = "linear",
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:

    """
    Convenience functional API for spectrogram computation.
    """
    metric = SpectrogramMetric()
    return metric.compute(
        data,
        sample_rate=sample_rate,
        window_length=window_length,
        overlap=overlap,
        nfft=nfft,
        window=window,
        mode=mode,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  SpectrogramMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SpectrogramMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
