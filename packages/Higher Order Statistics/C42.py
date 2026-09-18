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


@HigherOrderStatisticsFactory.register(["c42", "cumulant_42", "c_42"])
class C42Metric(BaseHigherOrderMetric):
    """
    Computes Fourth-Order Double-Conjugated Cumulant C42 (Kurtosis Cumulant):
      C42 = Cum(z, z, z*, z*) = M42 - |M20|^2 - 2 * (M21)^2
      
    Theoretical Normalized C42 values:
      - Gaussian Noise: C42 = 0
      - BPSK / QPSK: C42 = -1.0
      - 8-PSK: C42 = -1.0
      - 16-QAM: C42 = -0.68
      - 64-QAM: C42 = -0.619
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes C42Metric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up C42Metric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "c42"

    def compute(self, iq_data, *, normalize: bool = False, **kwargs) -> complex:
        # Step 1: Center signal to zero mean and obtain average power
        z_zero_mean, power_sigma2 = _prepare_zero_mean_iq(iq_data, normalize=normalize)
        z_conj = np.conj(z_zero_mean)
        
        # Step 2: 2nd-order moments M20 and M21
        m20 = np.mean(z_zero_mean ** 2)
        m21 = np.mean(np.abs(z_zero_mean) ** 2)
        
        # Step 3: 4th-order moment M42 = E[|z_0|^4]
        m42 = np.mean((np.abs(z_zero_mean) ** 4))
        
        # Step 4: Compute cumulant C42 = M42 - |M20|^2 - 2 * (M21)^2
        c42_val = complex(m42 - (np.abs(m20) ** 2) - 2.0 * (m21 ** 2))
        
        return c42_val


_instance = C42Metric()
c42 = _instance.compute

__all__ = ["C42Metric", "c42"]


def main() -> None:
    print("=" * 60)
    print("  C42Metric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = C42Metric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
