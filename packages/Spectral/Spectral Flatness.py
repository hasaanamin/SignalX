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


@SpectralMetricFactory.register(["spectral_flatness", "wiener_entropy", "tonality_factor", "sfm", "spectral_flatness_measure"])
class SpectralFlatnessMetric(BaseSpectralMetric):
    """
    Computes Spectral Flatness Measure (SFM) / Wiener Entropy / Tonality Factor:
      SFM = exp( (1/N) * sum(ln(S[k])) ) / ( (1/N) * sum(S[k]) )
    MATLAB Reference: `spectralFlatness(x, fs)`.
    Quantifies how tone-like vs noise-like the spectrum is:
      - SFM ~ 1.0 (0 dB): Flat White Noise
      - SFM ~ 0.0 (-inf dB): Pure Sinusoid / Discrete Line Spectrum
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "spectral_flatness"

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
            return {"spectral_flatness_linear": 0.0, "spectral_flatness_db": -np.inf} if as_dict else 0.0

        freqs, psd = _compute_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
        if psd.size == 0:
            return {"spectral_flatness_linear": 0.0, "spectral_flatness_db": -np.inf} if as_dict else 0.0

        # Arithmetic Mean
        arithmetic_mean = float(np.mean(psd))
        if arithmetic_mean <= 0:
            return {"spectral_flatness_linear": 0.0, "spectral_flatness_db": -np.inf} if as_dict else 0.0

        # Geometric Mean via log-sum
        log_psd = np.log(np.maximum(1e-24, psd))
        geometric_mean = float(np.exp(np.mean(log_psd)))

        sfm_linear = float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))
        sfm_db = float(10.0 * np.log10(max(1e-12, sfm_linear)))

        if as_dict:
            return {
                "spectral_flatness_linear": sfm_linear,
                "spectral_flatness_db": sfm_db,
                "geometric_mean": geometric_mean,
                "arithmetic_mean": arithmetic_mean,
                "tonality_factor": float(1.0 - sfm_linear),
                "sample_rate": fs,
            }
        return sfm_linear


def spectral_flatness(
    iq_data: Any,
    sample_rate: float | None = None,
    nperseg: int | None = None,
    window: str = "hann",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for Spectral Flatness computation.
    """
    metric = SpectralFlatnessMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, nperseg=nperseg, window=window, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SpectralFlatnessMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SpectralFlatnessMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
