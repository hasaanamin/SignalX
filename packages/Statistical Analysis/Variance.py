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


@StatisticalAnalysisFactory.register(["variance", "signal_variance", "var"])
class VarianceAnalyzer(BaseStatisticalAnalyzer):
    """
    Computes signal variance:
      - Real signals: Var(x) = (1 / (N - ddof)) * sum((x_i - mean(x))^2)
      - Complex IQ:
          * mode='complex' (or 'auto'): total complex variance = Var(I) + Var(Q)
          * mode='magnitude': envelope variance = Var(|z|)
          * mode='real': Var(I)
          * mode='imag': Var(Q)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes VarianceAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up VarianceAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "variance"

    def compute(self, data, *, sample_variance: bool = False, mode: str = "auto") -> float:
        x = _as_array(data)
        ddof = 1 if sample_variance else 0
        if ddof == 1 and x.size < 2:
            return float("nan")

        m = str(mode).lower()

        if np.iscomplexobj(x):
            if m in ("complex", "auto"):
                mu = np.mean(x)
                diff_sq = np.abs(x - mu) ** 2
                return float(np.sum(diff_sq) / (x.size - ddof))
            elif m in ("magnitude", "abs"):
                return float(np.var(np.abs(x), ddof=ddof))
            elif m in ("real", "i"):
                return float(np.var(np.real(x), ddof=ddof))
            elif m in ("imag", "q"):
                return float(np.var(np.imag(x), ddof=ddof))
            else:
                raise ValueError(f"Unknown mode: {mode}")

        return float(np.var(x, ddof=ddof))


# Standalone functional API wrapper
_instance = VarianceAnalyzer()
variance = _instance.compute
signal_variance = _instance.compute

__all__ = ["VarianceAnalyzer", "variance", "signal_variance"]


def main() -> None:
    print("=" * 60)
    print("  VarianceAnalyzer: Demonstration")
    print("=" * 60)
    data = np.random.randn(1024)
    analyzer = VarianceAnalyzer()
    result = analyzer.compute(data)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
