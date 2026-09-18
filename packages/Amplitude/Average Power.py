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
from .base import BaseAmplitudeMetric, AmplitudeMetricFactory
from ._common import _as_iq_array


@AmplitudeMetricFactory.register(["average_power", "mean_power", "avg_power"])
class AveragePowerMetric(BaseAmplitudeMetric):
    """
    Computes average normalized IQ power:
      Average_Power = mean(|z[n]|^2) = 1/N * sum(I[n]^2 + Q[n]^2)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes AveragePowerMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up AveragePowerMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "average_power"

    def compute(self, iq_data, **kwargs) -> float:
        z = _as_iq_array(iq_data)
        
        # Step 1: Instantaneous power array P[n] = I^2 + Q^2 = |z[n]|^2
        instantaneous_power = (np.real(z) ** 2) + (np.imag(z) ** 2)
        
        # Step 2: Average power is the mean of instantaneous power
        avg_pwr = float(np.mean(instantaneous_power))
        
        return avg_pwr


_instance = AveragePowerMetric()
average_power = _instance.compute

__all__ = ["AveragePowerMetric", "average_power"]


def main() -> None:
    print("=" * 60)
    print(f"  AveragePowerMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = AveragePowerMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
