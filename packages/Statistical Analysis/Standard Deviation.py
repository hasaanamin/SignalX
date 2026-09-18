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
from .Variance import variance


@StatisticalAnalysisFactory.register(["standard_deviation", "std", "signal_std"])
class StandardDeviationAnalyzer(BaseStatisticalAnalyzer):
    """
    Computes standard deviation sigma = sqrt(variance).
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes StandardDeviationAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up StandardDeviationAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "standard_deviation"

    def compute(self, data, *, sample_variance: bool = False, mode: str = "auto") -> float:
        var_val = variance(data, sample_variance=sample_variance, mode=mode)
        if np.isnan(var_val) or var_val < 0:
            return float("nan")
        return float(np.sqrt(var_val))


# Standalone functional API wrapper
_instance = StandardDeviationAnalyzer()
standard_deviation = _instance.compute
std = _instance.compute
signal_std = _instance.compute

__all__ = ["StandardDeviationAnalyzer", "standard_deviation", "std", "signal_std"]


def main() -> None:
    print("=" * 60)
    print("  StandardDeviationAnalyzer: Demonstration")
    print("=" * 60)
    data = np.random.randn(1024)
    analyzer = StandardDeviationAnalyzer()
    result = analyzer.compute(data)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
