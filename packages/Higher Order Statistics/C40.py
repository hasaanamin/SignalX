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


@HigherOrderStatisticsFactory.register(["c40", "cumulant_40", "c_40"])
class C40Metric(BaseHigherOrderMetric):
    """
    Computes Fourth-Order Unconjugated Cumulant C40:
      C40 = Cum(z, z, z, z) = M40 - 3 * (M20)^2 = E[z_0^4] - 3 * (E[z_0^2])^2
      
    Key discriminator for BPSK vs QPSK vs 8-PSK:
      - BPSK: |C40| = -2
      - QPSK / 4-QAM: |C40| = 1
      - 8-PSK: |C40| = 0
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes C40Metric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up C40Metric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "c40"

    def compute(self, iq_data, *, normalize: bool = False, **kwargs) -> complex:
        # Step 1: Center signal to zero mean and obtain average power
        z_zero_mean, power_sigma2 = _prepare_zero_mean_iq(iq_data, normalize=normalize)
        
        # Step 2: Calculate 2nd-order moment M20 = E[z_0^2]
        m20 = np.mean(z_zero_mean ** 2)
        
        # Step 3: Calculate 4th-order unconjugated moment M40 = E[z_0^4]
        m40 = np.mean(z_zero_mean ** 4)
        
        # Step 4: Compute 4th-order cumulant: C40 = M40 - 3 * (M20)^2
        c40_val = complex(m40 - 3.0 * (m20 ** 2))
        
        return c40_val


_instance = C40Metric()
c40 = _instance.compute

__all__ = ["C40Metric", "c40"]


def main() -> None:
    print("=" * 60)
    print("  C40Metric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = C40Metric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
