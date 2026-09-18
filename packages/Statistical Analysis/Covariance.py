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

import numpy as np
from .base import BaseStatisticalAnalyzer, StatisticalAnalysisFactory
from ._common import _as_array


@StatisticalAnalysisFactory.register(["covariance", "cov", "iq_covariance"])
class CovarianceAnalyzer(BaseStatisticalAnalyzer):
    """
    Calculates Population Covariance between two signals or between I and Q channels:
      Cov(I, Q) = mean((I - mean(I)) * (Q - mean(Q))) matching MATLAB population covariance.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes CovarianceAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up CovarianceAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "covariance"

    def compute(self, data, y=None, *, sample_covariance: bool = False, matrix: bool = False) -> float | np.ndarray:
        arr_x = _as_array(data)
        ddof = 1 if sample_covariance else 0

        if y is not None:
            arr_y = _as_array(y)
            if arr_x.size != arr_y.size:
                raise ValueError(f"Length mismatch: x={arr_x.size}, y={arr_y.size}")
            n = arr_x.size
            if ddof == 1 and n < 2:
                return float("nan")

            mu_x = np.mean(arr_x)
            mu_y = np.mean(arr_y)
            cov = np.sum((arr_x - mu_x) * np.conj(arr_y - mu_y)) / (n - ddof)
            return float(np.real(cov)) if not np.iscomplexobj(cov) else cov

        if np.iscomplexobj(arr_x):
            i = np.real(arr_x)
            q = np.imag(arr_x)
            if matrix:
                return np.cov(i, q, ddof=ddof)
            if ddof == 0:
                return float(np.mean((i - np.mean(i)) * (q - np.mean(q))))
            return float(np.sum((i - np.mean(i)) * (q - np.mean(q))) / max(1, arr_x.size - 1))

        return float(np.var(arr_x, ddof=ddof))


_instance = CovarianceAnalyzer()
covariance = _instance.compute


def iq_covariance_matrix(iq_data, *, sample_covariance: bool = False) -> np.ndarray:
    """Computes the 2x2 I/Q covariance matrix for imbalance detection."""
    return _instance.compute(iq_data, sample_covariance=sample_covariance, matrix=True)


__all__ = ["CovarianceAnalyzer", "covariance", "iq_covariance_matrix"]


def main() -> None:
    print("=" * 60)
    print("  CovarianceAnalyzer: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    analyzer = CovarianceAnalyzer()
    result = analyzer.compute(sample_iq)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
