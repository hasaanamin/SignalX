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


@AmplitudeMetricFactory.register(["peak_amplitude", "peak_mag", "max_amplitude"])
class PeakAmplitudeMetric(BaseAmplitudeMetric):
    """
    Computes maximum peak envelope amplitude:
      Peak_Amplitude = max(|z[n]|) = max(sqrt(I[n]^2 + Q[n]^2))
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes PeakAmplitudeMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up PeakAmplitudeMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "peak_amplitude"

    def compute(self, iq_data, **kwargs) -> float:
        z = _as_iq_array(iq_data)
        
        # Step 1: Calculate instantaneous envelope amplitude
        envelope = np.abs(z)
        
        # Step 2: Peak amplitude is the maximum envelope value
        peak_val = float(np.max(envelope))
        
        return peak_val


_instance = PeakAmplitudeMetric()
peak_amplitude = _instance.compute

__all__ = ["PeakAmplitudeMetric", "peak_amplitude"]


def main() -> None:
    print("=" * 60)
    print(f"  PeakAmplitudeMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = PeakAmplitudeMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
