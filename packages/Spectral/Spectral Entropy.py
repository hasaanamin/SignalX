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


@SpectralMetricFactory.register(["spectral_entropy", "shannon_spectral_entropy", "spectral_information_entropy", "entropy_spectrum"])
class SpectralEntropyMetric(BaseSpectralMetric):
    """
    Computes Shannon Information Spectral Entropy:
      P[k] = S[k] / sum(S[m])
      H = -sum(P[k] * log2(P[k]))
      H_norm = H / log2(N) in [0.0, 1.0]
    MATLAB Reference: `spectralEntropy(x, fs)`.
    Measures spectral disorder / complexity:
      - Narrowband / pure carrier tone: H_norm ~ 0.0
      - Flat White Gaussian Noise: H_norm ~ 1.0
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "spectral_entropy"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        nperseg: int | None = None,
        window: str = "hann",
        normalize: bool = True,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"normalized_entropy": 0.0, "raw_entropy_bits": 0.0} if as_dict else 0.0

        freqs, psd = _compute_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
        total_pwr = np.sum(psd)

        if total_pwr <= 0 or psd.size <= 1:
            return {"normalized_entropy": 0.0, "raw_entropy_bits": 0.0} if as_dict else 0.0

        # Normalized probability mass function
        p_k = psd / total_pwr
        # Mask zero / negative values to avoid log(0)
        valid = p_k > 1e-24
        p_valid = p_k[valid]

        # Raw Shannon Entropy in bits
        h_bits = float(-np.sum(p_valid * np.log2(p_valid)))
        # Maximum possible entropy for N discrete bins
        h_max = float(np.log2(len(psd)))
        h_norm = float(np.clip(h_bits / max(1e-12, h_max), 0.0, 1.0))

        if as_dict:
            return {
                "normalized_entropy": h_norm,
                "raw_entropy_bits": h_bits,
                "max_entropy_bits": h_max,
                "num_bins": len(psd),
                "sample_rate": fs,
            }
        return h_norm if normalize else h_bits


def spectral_entropy(
    iq_data: Any,
    sample_rate: float | None = None,
    nperseg: int | None = None,
    window: str = "hann",
    normalize: bool = True,
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for Spectral Entropy computation.
    """
    metric = SpectralEntropyMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, nperseg=nperseg, window=window, normalize=normalize, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SpectralEntropyMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SpectralEntropyMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
