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


@StatisticalAnalysisFactory.register(["mean", "signal_mean"])
class MeanAnalyzer(BaseStatisticalAnalyzer):
    """
    Computes arithmetic mean:
      - Real signals: mean(x) = (1/N) * sum(x_i)
      - Complex IQ:
          * mode='complex' (or 'auto'): complex mean = mean(I) + 1j * mean(Q)
          * mode='magnitude': mean envelope = mean(|x|)
          * mode='real': mean(I)
          * mode='imag': mean(Q)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes MeanAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up MeanAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "mean"

    def compute(self, data, *, mode: str = "auto") -> float | complex:
        x = _as_array(data)
        m = str(mode).lower()

        if np.iscomplexobj(x):
            if m in ("complex", "auto"):
                return complex(np.mean(x))
            elif m in ("magnitude", "abs"):
                return float(np.mean(np.abs(x)))
            elif m in ("real", "i"):
                return float(np.mean(np.real(x)))
            elif m in ("imag", "q"):
                return float(np.mean(np.imag(x)))
            else:
                raise ValueError(f"Unknown mode: {mode}")

        return float(np.mean(x))


# Standalone functional API wrapper
_instance = MeanAnalyzer()
mean = _instance.compute
signal_mean = _instance.compute

__all__ = ["MeanAnalyzer", "mean", "signal_mean"]


def main() -> None:
    print("=" * 60)
    print("  MeanAnalyzer: Demonstration")
    print("=" * 60)
    data = np.random.randn(1024)
    analyzer = MeanAnalyzer()
    result = analyzer.compute(data)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
