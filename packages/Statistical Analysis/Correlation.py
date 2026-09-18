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


@StatisticalAnalysisFactory.register(["correlation", "correlation_coefficient", "corr"])
class CorrelationAnalyzer(BaseStatisticalAnalyzer):
    """
    Calculates Pearson Correlation Coefficient, Cross-Correlation, and Auto-Correlation.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes CorrelationAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up CorrelationAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "correlation"

    def compute(self, data, y=None) -> float | complex:
        arr_x = _as_array(data)

        if y is not None:
            arr_y = _as_array(y)
            if arr_x.size != arr_y.size:
                raise ValueError("Length mismatch between x and y")
            std_x = np.std(arr_x)
            std_y = np.std(arr_y)
            if std_x == 0 or std_y == 0:
                return 0.0
            cov = np.mean((arr_x - np.mean(arr_x)) * np.conj(arr_y - np.mean(arr_y)))
            return float(np.real(cov / (std_x * std_y))) if not np.iscomplexobj(cov) else cov / (std_x * std_y)

        if np.iscomplexobj(arr_x):
            i = np.real(arr_x)
            q = np.imag(arr_x)
            std_i = np.std(i)
            std_q = np.std(q)
            if std_i == 0 or std_q == 0:
                return 0.0
            return float(np.mean((i - np.mean(i)) * (q - np.mean(q))) / (std_i * std_q))

        return 1.0


_instance = CorrelationAnalyzer()
correlation_coefficient = _instance.compute


def cross_correlation(x, y, *, mode: str = "full", normalize: bool = True) -> np.ndarray:
    """Computes Cross-Correlation sequence R_xy[k]."""
    arr_x = _as_array(x)
    arr_y = _as_array(y)
    corr = np.correlate(arr_x, arr_y, mode=mode)
    if normalize:
        norm_factor = np.sqrt(np.sum(np.abs(arr_x) ** 2) * np.sum(np.abs(arr_y) ** 2))
        if norm_factor > 0:
            corr = corr / norm_factor
    return corr


def auto_correlation(x, *, max_lag: int | None = None, normalize: bool = True) -> np.ndarray:
    """Computes Auto-Correlation sequence R_xx[k]."""
    arr_x = _as_array(x)
    corr = np.correlate(arr_x, arr_x, mode="full")
    center = len(corr) // 2
    r_xx = corr[center:]
    if normalize and r_xx[0] > 0:
        r_xx = r_xx / r_xx[0]
    if max_lag is not None:
        r_xx = r_xx[:int(max_lag)]
    return r_xx


__all__ = ["CorrelationAnalyzer", "correlation_coefficient", "cross_correlation", "auto_correlation"]


def main() -> None:
    print("=" * 60)
    print("  CorrelationAnalyzer: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    analyzer = CorrelationAnalyzer()
    result = analyzer.compute(sample_iq)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
