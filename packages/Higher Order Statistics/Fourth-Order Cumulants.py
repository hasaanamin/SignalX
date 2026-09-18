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
from .base import BaseHigherOrderMetric, HigherOrderStatisticsFactory
from ._common import _prepare_zero_mean_iq


@HigherOrderStatisticsFactory.register(["fourth_order_cumulants", "c4x", "4th_order_cumulants"])
class FourthOrderCumulantsMetric(BaseHigherOrderMetric):
    """
    Computes the complete set of Fourth-Order Cumulants:
      - C40 = M40 - 3 * (M20)^2
      - C41 = M41 - 3 * M20 * M21
      - C42 = M42 - |M20|^2 - 2 * (M21)^2
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes FourthOrderCumulantsMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up FourthOrderCumulantsMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "fourth_order_cumulants"

    def compute(self, iq_data, *, normalize: bool = False, **kwargs) -> dict[str, complex]:
        # Step 1: Center signal to zero mean and obtain average power
        z_zero_mean, power_sigma2 = _prepare_zero_mean_iq(iq_data, normalize=normalize)
        z_conj = np.conj(z_zero_mean)
        
        # Step 2: 2nd-order moments
        m20 = np.mean(z_zero_mean ** 2)
        m21 = np.mean(np.abs(z_zero_mean) ** 2)
        
        # Step 3: 4th-order moments
        m40 = np.mean(z_zero_mean ** 4)
        m41 = np.mean((z_zero_mean ** 3) * z_conj)
        m42 = np.mean(np.abs(z_zero_mean) ** 4)
        
        # Step 4: Compute 4th-order cumulants
        c40 = m40 - 3.0 * (m20 ** 2)
        c41 = m41 - 3.0 * m20 * m21
        c42 = m42 - (np.abs(m20) ** 2) - 2.0 * (m21 ** 2)
        
        return {
            "C40": complex(c40),
            "C41": complex(c41),
            "C42": complex(c42),
        }


_instance = FourthOrderCumulantsMetric()
fourth_order_cumulants = _instance.compute

__all__ = ["FourthOrderCumulantsMetric", "fourth_order_cumulants"]


def main() -> None:
    print("=" * 60)
    print("  FourthOrderCumulantsMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FourthOrderCumulantsMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
