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


@AmplitudeMetricFactory.register(["amplitude_variance", "envelope_variance", "amp_variance"])
class AmplitudeVarianceMetric(BaseAmplitudeMetric):
    """
    Computes variance of envelope amplitude |x[n]|:
      Var(A) = 1/(N - ddof) * sum((A[n] - mean(A))^2)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes AmplitudeVarianceMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up AmplitudeVarianceMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "amplitude_variance"

    def compute(self, iq_data, *, sample_variance: bool = False, **kwargs) -> float:
        z = _as_iq_array(iq_data)
        
        # Step 1: Calculate instantaneous envelope amplitude A[n] = |z[n]|
        envelope = np.abs(z)
        
        # Step 2: Degrees of freedom (0 for population variance, 1 for sample variance)
        ddof = 1 if sample_variance else 0
        if ddof == 1 and envelope.size < 2:
            return float("nan")
        
        # Step 3: Mean envelope amplitude
        mean_amp = np.mean(envelope)
        
        # Step 4: Squared deviations from the mean
        squared_deviations = (envelope - mean_amp) ** 2
        
        # Step 5: Variance = sum(squared_deviations) / (N - ddof)
        amp_variance = float(np.sum(squared_deviations) / (envelope.size - ddof))
        
        return amp_variance


_instance = AmplitudeVarianceMetric()
amplitude_variance = _instance.compute

__all__ = ["AmplitudeVarianceMetric", "amplitude_variance"]


def main() -> None:
    print("=" * 60)
    print(f"  AmplitudeVarianceMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = AmplitudeVarianceMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
