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


@HigherOrderStatisticsFactory.register(["c21", "cumulant_21", "c_21"])
class C21Metric(BaseHigherOrderMetric):
    """
    Computes Second-Order Conjugated Cumulant C21 (Average Signal Power / Variance):
      C21 = Cum(z, z*) = E[|z_0|^2] = M21
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes C21Metric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up C21Metric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "c21"

    def compute(self, iq_data, *, normalize: bool = False, **kwargs) -> complex:
        # Step 1: Center signal to zero mean and obtain average power
        z_zero_mean, power_sigma2 = _prepare_zero_mean_iq(iq_data, normalize=normalize)
        
        # Step 2: Calculate squared magnitude |z_0|^2 = z_0 * conj(z_0)
        power_samples = np.abs(z_zero_mean) ** 2
        
        # Step 3: Compute expectation / mean: C21 = E[|z_0|^2]
        c21_val = complex(np.mean(power_samples))
        
        return c21_val


_instance = C21Metric()
c21 = _instance.compute

__all__ = ["C21Metric", "c21"]


def main() -> None:
    print("=" * 60)
    print("  C21Metric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = C21Metric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
