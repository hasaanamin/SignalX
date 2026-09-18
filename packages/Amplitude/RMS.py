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


@AmplitudeMetricFactory.register(["rms", "rms_amplitude", "root_mean_square"])
class RMSMetric(BaseAmplitudeMetric):
    """
    Computes Complex-Envelope Root Mean Square (RMS) amplitude:
      RMS = sqrt(mean(|z[n]|^2)) = sqrt(1/N * sum(I[n]^2 + Q[n]^2))
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes RMSMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up RMSMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "rms"

    def compute(self, iq_data, **kwargs) -> float:
        z = _as_iq_array(iq_data)
        
        # Step 1: Calculate instantaneous power |z[n]|^2 = I^2 + Q^2
        power_samples = np.abs(z) ** 2
        
        # Step 2: Calculate mean power (average of squared values)
        mean_power = np.mean(power_samples)
        
        # Step 3: Root Mean Square = square root of mean power
        rms_val = float(np.sqrt(mean_power))
        
        return rms_val


_instance = RMSMetric()
rms = _instance.compute
rms_amplitude = _instance.compute

__all__ = ["RMSMetric", "rms", "rms_amplitude"]


def main() -> None:
    print("=" * 60)
    print(f"  RMSMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = RMSMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
