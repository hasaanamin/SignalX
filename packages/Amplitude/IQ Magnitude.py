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


@AmplitudeMetricFactory.register(["iq_magnitude", "envelope_amplitude", "magnitude"])
class IQMagnitudeMetric(BaseAmplitudeMetric):
    """
    Computes instantaneous complex IQ magnitude / envelope:
      A[n] = sqrt(I[n]^2 + Q[n]^2) = |z[n]|
      Mean Magnitude: mean(|z[n]|) matching MATLAB mean(abs(z))
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes IQMagnitudeMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up IQMagnitudeMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "iq_magnitude"

    def compute(self, iq_data, *, return_mean: bool = False, **kwargs) -> np.ndarray | float:
        z = _as_iq_array(iq_data)
        
        # Step 1: Real (I) and Imaginary (Q) components
        i_samples = np.real(z)
        q_samples = np.imag(z)
        
        # Step 2: Instantaneous power / squared sum I^2 + Q^2
        squared_sum = (i_samples ** 2) + (q_samples ** 2)
        
        # Step 3: Complex magnitude envelope sqrt(I^2 + Q^2) = |z[n]|
        magnitude_envelope = np.sqrt(squared_sum)
        
        if return_mean:
            return float(np.mean(magnitude_envelope))

        return magnitude_envelope


_instance = IQMagnitudeMetric()
iq_magnitude = _instance.compute
envelope_amplitude = _instance.compute
mean_iq_magnitude = lambda iq_data, **kw: _instance.compute(iq_data, return_mean=True, **kw)

__all__ = ["IQMagnitudeMetric", "iq_magnitude", "envelope_amplitude", "mean_iq_magnitude"]


def main() -> None:
    print("=" * 60)
    print(f"  IQMagnitudeMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = IQMagnitudeMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
