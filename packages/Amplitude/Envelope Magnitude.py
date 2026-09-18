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


@AmplitudeMetricFactory.register(["envelope_magnitude", "envelope"])
class EnvelopeMagnitudeMetric(BaseAmplitudeMetric):
    """
    Computes instantaneous envelope magnitude |x[n]|.
    Supports complex IQ data as well as real 1-D signals (via Hilbert transform).
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes EnvelopeMagnitudeMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up EnvelopeMagnitudeMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "envelope_magnitude"

    def compute(self, signal, **kwargs) -> np.ndarray:
        try:
            z = _as_iq_array(signal)
            i_samples = np.real(z)
            q_samples = np.imag(z)
            squared_sum = (i_samples ** 2) + (q_samples ** 2)
            envelope = np.sqrt(squared_sum)
            return envelope
        except (TypeError, ValueError):
            from scipy.signal import hilbert
            sig = np.asarray(signal, dtype=np.float64).reshape(-1)
            if sig.size == 0:
                raise ValueError("Signal must contain at least one sample")
            analytic_signal = hilbert(sig)
            envelope = np.abs(analytic_signal)
            return envelope


_instance = EnvelopeMagnitudeMetric()
envelope_magnitude = _instance.compute
envelope_amplitude = _instance.compute

__all__ = ["EnvelopeMagnitudeMetric", "envelope_magnitude", "envelope_amplitude"]


def main() -> None:
    print("=" * 60)
    print(f"  EnvelopeMagnitudeMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = EnvelopeMagnitudeMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
