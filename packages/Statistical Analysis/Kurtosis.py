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


@StatisticalAnalysisFactory.register(["kurtosis", "excess_kurtosis", "kurt"])
class KurtosisAnalyzer(BaseStatisticalAnalyzer):
    """
    Calculates Kurtosis (4th standardized moment) and Excess Kurtosis:
      kurtosis = (1/N * sum((x - mu)^4)) / (sigma^4)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes KurtosisAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up KurtosisAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "kurtosis"

    def compute(self, data, *, excess: bool = True, bias: bool = True, mode: str = "auto") -> float:
        x = _as_real_array(data, mode=mode)
        n = x.size
        if n < 4:
            return float("nan")

        mu = np.mean(x)
        m2 = np.mean((x - mu) ** 2)
        m4 = np.mean((x - mu) ** 4)

        if m2 == 0.0:
            return 0.0

        raw_kurt = m4 / (m2 ** 2)

        if not bias:
            unbiased_excess = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * raw_kurt - 3 * (n - 1))
            return float(unbiased_excess if excess else unbiased_excess + 3.0)

        return float(raw_kurt - 3.0 if excess else raw_kurt)


# Standalone functional API wrapper
_instance = KurtosisAnalyzer()
kurtosis = _instance.compute
excess_kurtosis = _instance.compute

__all__ = ["KurtosisAnalyzer", "kurtosis", "excess_kurtosis"]


def main() -> None:
    print("=" * 60)
    print("  KurtosisAnalyzer: Demonstration")
    print("=" * 60)
    data = np.random.randn(1024)
    analyzer = KurtosisAnalyzer()
    result = analyzer.compute(data)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
