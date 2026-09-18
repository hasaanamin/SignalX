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
from .base import BaseSpectralMetric, SpectralMetricFactory
from ._common import _as_iq_array, _extract_sample_rate, _create_window


@SpectralMetricFactory.register(["fft", "fft_spectrum", "fourier_transform", "complex_fft", "spectrum_fft"])
class FFTSpectrumMetric(BaseSpectralMetric):
    """
    Computes Complex Fast Fourier Transform (FFT) and Spectral Components:
      - Centered frequency grid [-fs/2, fs/2) via fftshift.
      - Complex spectral bins X[k].
      - Magnitude spectrum (linear and dBFS / normalized dB).
      - Phase spectrum (wrapped [-pi, pi] and unwrapped).
      - Power spectrum |X[k]|^2.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "fft"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        nfft: int | None = None,
        window: str = "hann",
        normalize: str = "standard",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"frequencies": np.array([]), "spectrum": np.array([])} if as_dict else np.array([])

        if nfft is None:
            nfft = n

        # Apply window
        if window and window.lower() not in ("none", "rect", "rectangular", "boxcar"):
            win = _create_window(window, n)
            z_win = z * win
            scale = np.sum(win)
        else:
            z_win = z
            scale = float(n)

        # Complex FFT with centered zero frequency
        fft_raw = np.fft.fft(z_win, n=nfft)
        fft_centered = np.fft.fftshift(fft_raw)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

        if normalize == "magnitude":
            # Normalizes peak magnitude to true sinusoidal amplitude
            fft_norm = fft_centered / max(1e-18, scale / 2.0)
        elif normalize == "power":
            # Normalizes total power
            fft_norm = fft_centered / np.sqrt(max(1e-18, nfft * (scale / n)))
        else:  # standard
            fft_norm = fft_centered / scale

        magnitude_linear = np.abs(fft_norm)
        max_mag = np.max(magnitude_linear) if magnitude_linear.size > 0 else 1.0
        magnitude_db = 20.0 * np.log10(np.maximum(1e-12, magnitude_linear / max(1e-12, max_mag)))
        phase_rad = np.angle(fft_centered)
        phase_deg = np.rad2deg(phase_rad)

        if as_dict:
            return {
                "frequencies": freqs,
                "spectrum_complex": fft_norm,
                "magnitude_linear": magnitude_linear,
                "magnitude_db": magnitude_db,
                "phase_rad": phase_rad,
                "phase_deg": phase_deg,
                "power_spectrum": np.abs(fft_norm) ** 2,
                "sample_rate": fs,
                "nfft": nfft,
                "window": window,
                "resolution_hz": fs / nfft,
            }
        return fft_norm


def fft_spectrum(
    iq_data: Any,
    sample_rate: float | None = None,
    nfft: int | None = None,
    window: str = "hann",
    normalize: str = "standard",
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:
    """
    Functional DSP wrapper for Complex FFT computation.
    """
    metric = FFTSpectrumMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, nfft=nfft, window=window, normalize=normalize, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  FFTSpectrumMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FFTSpectrumMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
