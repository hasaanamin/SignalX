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
from ._common import _as_iq_array, _extract_sample_rate, _compute_welch_psd


@SpectralMetricFactory.register(["spectral_skewness", "spectral_asymmetry", "spectral_skew"])
class SpectralSkewnessMetric(BaseSpectralMetric):
    """
    Computes Spectral Skewness (Third Standardized Spectral Moment):
      gamma_1 = sum((f - f_c)^3 * S(f)) / (sigma_f^3 * sum(S(f)))
    MATLAB Reference: `spectralSkewness(x, fs)`.
    Quantifies the asymmetry of the power spectrum around the centroid.
      - gamma_1 > 0: energy skewed towards higher frequencies.
      - gamma_1 < 0: energy skewed towards lower frequencies.
      - gamma_1 ~ 0: symmetric spectrum.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "spectral_skewness"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        nperseg: int | None = None,
        window: str = "hann",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"spectral_skewness": 0.0} if as_dict else 0.0

        freqs, psd = _compute_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
        total_pwr = np.sum(psd)

        if total_pwr <= 0:
            return {"spectral_skewness": 0.0} if as_dict else 0.0

        # Centroid mu_1
        fc = np.sum(freqs * psd) / total_pwr
        # Variance mu_2
        var_hz2 = np.sum(((freqs - fc) ** 2) * psd) / total_pwr
        sigma_f = np.sqrt(max(1e-24, var_hz2))

        # Skewness mu_3 / sigma^3
        m3 = np.sum(((freqs - fc) ** 3) * psd) / total_pwr
        skew = float(m3 / (sigma_f ** 3))

        if as_dict:
            return {
                "spectral_skewness": skew,
                "spectral_centroid_hz": float(fc),
                "spectral_spread_hz": float(sigma_f),
                "sample_rate": fs,
            }
        return skew


def spectral_skewness(
    iq_data: Any,
    sample_rate: float | None = None,
    nperseg: int | None = None,
    window: str = "hann",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for Spectral Skewness computation.
    """
    metric = SpectralSkewnessMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, nperseg=nperseg, window=window, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SpectralSkewnessMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SpectralSkewnessMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
