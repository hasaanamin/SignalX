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


@StatisticalAnalysisFactory.register(["skewness", "skew"])
class SkewnessAnalyzer(BaseStatisticalAnalyzer):
    """
    Calculates Fisher-Pearson skewness (3rd standardized moment):
      skewness = (1/N * sum((x - mu)^3)) / (sigma^3)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes SkewnessAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up SkewnessAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "skewness"

    def compute(self, data, *, bias: bool = True, mode: str = "auto") -> float:
        x = _as_real_array(data, mode=mode)
        n = x.size
        if n < 3:
            return float("nan")

        mu = np.mean(x)
        m2 = np.mean((x - mu) ** 2)
        m3 = np.mean((x - mu) ** 3)

        if m2 == 0.0:
            return 0.0

        val = m3 / (m2 ** 1.5)

        if not bias and n > 2:
            val = val * np.sqrt(n * (n - 1)) / (n - 2)

        return float(val)


# Standalone functional API wrapper
_instance = SkewnessAnalyzer()
skewness = _instance.compute

__all__ = ["SkewnessAnalyzer", "skewness"]


def main() -> None:
    print("=" * 60)
    print("  SkewnessAnalyzer: Demonstration")
    print("=" * 60)
    data = np.random.randn(1024)
    analyzer = SkewnessAnalyzer()
    result = analyzer.compute(data)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
