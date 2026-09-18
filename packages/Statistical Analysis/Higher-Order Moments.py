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


@StatisticalAnalysisFactory.register(["higher_order_moments", "higher_order_cumulants", "cumulants", "hoc"])
class HigherOrderMomentsAnalyzer(BaseStatisticalAnalyzer):
    """
    Computes Higher-Order Moments (HOM) and Cumulants (HOC):
      - C20 = M20 = E[z^2]
      - C21 = M21 = E[|z|^2]
      - C40 = M40 - 3 * (M20)^2
      - C41 = M41 - 3 * M20 * M21
      - C42 = M42 - |M20|^2 - 2 * (M21)^2
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes HigherOrderMomentsAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up HigherOrderMomentsAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "higher_order_moments"

    def compute(self, data, **kwargs) -> dict[str, complex]:
        z = _as_array(data)
        z0 = z - np.mean(z)
        rms_val = np.sqrt(np.mean(np.abs(z0) ** 2))
        if rms_val > 0:
            z0 = z0 / rms_val

        z_conj = np.conj(z0)

        m20 = np.mean(z0 ** 2)
        m21 = np.mean(np.abs(z0) ** 2)
        m40 = np.mean(z0 ** 4)
        m41 = np.mean((z0 ** 3) * z_conj)
        m42 = np.mean((z0 ** 2) * (z_conj ** 2))

        c20 = m20
        c21 = m21
        c40 = m40 - 3.0 * (m20 ** 2)
        c41 = m41 - 3.0 * m20 * m21
        c42 = m42 - np.abs(m20) ** 2 - 2.0 * (m21 ** 2)

        return {
            "C20": complex(c20),
            "C21": complex(c21),
            "C40": complex(c40),
            "C41": complex(c41),
            "C42": complex(c42),
        }


_instance = HigherOrderMomentsAnalyzer()
higher_order_cumulants = _instance.compute
cumulants = _instance.compute


def raw_moment(data, order: int = 1) -> float | complex:
    """Computes raw moment m_k = (1/N) * sum(x^k)."""
    x = _as_array(data)
    return np.mean(x ** int(order))


def central_moment(data, order: int = 2) -> float | complex:
    """Computes central moment mu_k = (1/N) * sum((x - mu)^k)."""
    x = _as_array(data)
    mu = np.mean(x)
    return np.mean((x - mu) ** int(order))


__all__ = ["HigherOrderMomentsAnalyzer", "higher_order_cumulants", "cumulants", "raw_moment", "central_moment"]


def main() -> None:
    print("=" * 60)
    print("  HigherOrderMomentsAnalyzer: Demonstration")
    print("=" * 60)
    data = np.random.randn(1024)
    analyzer = HigherOrderMomentsAnalyzer()
    result = analyzer.compute(data)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
