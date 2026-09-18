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
from ._common import _as_iq_array, _extract_sample_rate, _compute_welch_psd, _create_window


@SpectralMetricFactory.register(["spectral_kurtosis", "spectral_peakedness", "spectral_kurt", "pkurtosis"])
class SpectralKurtosisMetric(BaseSpectralMetric):
    """
    Computes Spectral Kurtosis:
      1. Fourth Standardized Spectral Moment (Global Peakedness):
         kappa = sum((f - f_c)^4 * S(f)) / (sigma_f^4 * sum(S(f)))
      2. Time-Frequency Transient Spectral Kurtosis SK(f) (Antoni's formulation for transient detection):
         SK(f) = <|X(t, f)|^4> / <|X(t, f)|^2>^2 - 2
    MATLAB Reference: `spectralKurtosis(x, fs)`, `pkurtosis(x, fs)`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "spectral_kurtosis"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        nperseg: int | None = None,
        window: str = "hann",
        mode: str = "moment",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"spectral_kurtosis": 0.0} if as_dict else 0.0

        if mode == "transient":
            # Short-time STFT Spectral Kurtosis vector across frequency bins
            nseg = min(n, 256)
            nseg = max(16, 1 << (nseg.bit_length() - 1))
            noverlap = nseg // 2
            win = _create_window(window, nseg)
            step = max(1, nseg - noverlap)

            frames = []
            for start in range(0, n - nseg + 1, step):
                seg = z[start : start + nseg] * win
                frames.append(np.fft.fftshift(np.fft.fft(seg)))

            if len(frames) > 1:
                stft_mat = np.array(frames)  # [num_frames, nseg]
                power_mat = np.abs(stft_mat) ** 2
                mean_p2 = np.mean(power_mat ** 2, axis=0)
                p_mean = np.mean(power_mat, axis=0)
                # SK(f) = <S^2> / <S>^2 - 2
                sk_vector = (mean_p2 / np.maximum(1e-18, p_mean ** 2)) - 2.0
                freqs = np.fft.fftshift(np.fft.fftfreq(nseg, d=1.0 / fs))
                global_kurt = float(np.mean(sk_vector))
            else:
                sk_vector = np.zeros(nseg)
                freqs = np.fft.fftshift(np.fft.fftfreq(nseg, d=1.0 / fs))
                global_kurt = 0.0
        else:
            # Standard Fourth Spectral Moment
            freqs, psd = _compute_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
            total_pwr = np.sum(psd)

            if total_pwr <= 0:
                return {"spectral_kurtosis": 0.0} if as_dict else 0.0

            fc = np.sum(freqs * psd) / total_pwr
            var_hz2 = np.sum(((freqs - fc) ** 2) * psd) / total_pwr
            sigma_f = np.sqrt(max(1e-24, var_hz2))

            m4 = np.sum(((freqs - fc) ** 4) * psd) / total_pwr
            global_kurt = float(m4 / (sigma_f ** 4))
            sk_vector = None

        if as_dict:
            res = {
                "spectral_kurtosis": global_kurt,
                "excess_spectral_kurtosis": float(global_kurt - 3.0),
                "mode": mode,
                "sample_rate": fs,
            }
            if sk_vector is not None:
                res["sk_profile"] = sk_vector
                res["frequencies"] = freqs
            return res

        return global_kurt


def spectral_kurtosis(
    iq_data: Any,
    sample_rate: float | None = None,
    nperseg: int | None = None,
    window: str = "hann",
    mode: str = "moment",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for Spectral Kurtosis computation.
    """
    metric = SpectralKurtosisMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, nperseg=nperseg, window=window, mode=mode, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SpectralKurtosisMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SpectralKurtosisMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
