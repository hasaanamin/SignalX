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
from ._common import _as_real_array


@StatisticalAnalysisFactory.register(["distribution", "distribution_analysis", "analyze_distribution"])
class DistributionAnalyzer(BaseStatisticalAnalyzer):
    """Performs statistical distribution analysis on signal amplitudes and values."""

    def __init__(self, **kwargs) -> None:
        """Constructor."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor."""
        super().__del__()

    @property
    def name(self) -> str:
        return "distribution_analysis"

    def compute(self, data, *, bins: int = 50, mode: str = "auto") -> dict:
        x = _as_real_array(data, mode=mode)
        n = int(x.size)
        if n == 0:
            return {}

        mu = float(np.mean(x))
        var = float(np.var(x))
        std = float(np.std(x))
        min_val = float(np.min(x))
        max_val = float(np.max(x))

        quantiles = {
            "p01": float(np.percentile(x, 1)),
            "p05": float(np.percentile(x, 5)),
            "p25_q1": float(np.percentile(x, 25)),
            "median_p50": float(np.median(x)),
            "p75_q3": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
        }
        iqr = quantiles["p75_q3"] - quantiles["p25_q1"]

        counts, bin_edges = np.histogram(x, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        sorted_x = np.sort(x)
        cdf_y = np.arange(1, n + 1) / n

        if std > 0:
            skew = float(np.mean(((x - mu) / std) ** 3))
            kurt = float(np.mean(((x - mu) / std) ** 4) - 3.0)
        else:
            skew = 0.0
            kurt = 0.0

        return {
            "sample_count": n,
            "mean": mu,
            "variance": var,
            "std": std,
            "min": min_val,
            "max": max_val,
            "iqr": float(iqr),
            "skewness": skew,
            "excess_kurtosis": kurt,
            "quantiles": quantiles,
            "pdf": {
                "density": counts.tolist(),
                "bin_centers": bin_centers.tolist(),
                "bin_edges": bin_edges.tolist(),
            },
            "cdf": {
                "x": sorted_x[::max(1, n // 100)].tolist(),
                "y": cdf_y[::max(1, n // 100)].tolist(),
            }
        }


_instance = DistributionAnalyzer()
distribution_analysis = _instance.compute
analyze_distribution = _instance.compute

__all__ = ["DistributionAnalyzer", "distribution_analysis", "analyze_distribution"]


def main() -> None:
    print("=" * 60)
    print("  DistributionAnalyzer: Demonstration")
    print("=" * 60)
    data = np.random.randn(1024)
    analyzer = DistributionAnalyzer()
    result = analyzer.compute(data)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
