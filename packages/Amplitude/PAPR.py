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


@AmplitudeMetricFactory.register(["papr", "papr_db", "peak_to_average_power_ratio"])
class PAPRMetric(BaseAmplitudeMetric):
    """
    Computes Peak-to-Average Power Ratio (PAPR):
      Linear: PAPR_linear = max(|z[n]|^2) / mean(|z[n]|^2) = P_peak / P_avg
      dB:     PAPR_dB = 10 * log10(PAPR_linear)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes PAPRMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up PAPRMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "papr"

    def compute(self, iq_data, *, db: bool = False, mode: str = "default", **kwargs) -> float | dict:
        z = _as_iq_array(iq_data)
        
        # Step 1: Instantaneous power array P[n] = |z[n]|^2 = I^2 + Q^2
        power_samples = (np.real(z) ** 2) + (np.imag(z) ** 2)
        
        # Step 2: Peak instantaneous power
        peak_power_val = float(np.max(power_samples))
        
        # Step 3: Average power
        avg_power_val = float(np.mean(power_samples))
        
        # Step 4: Handle zero signal / silence
        if avg_power_val == 0.0:
            papr_linear = float("nan")
            papr_db_val = float("-inf")
        else:
            papr_linear = float(peak_power_val / avg_power_val)
            papr_db_val = float(10.0 * np.log10(papr_linear)) if papr_linear > 0 else float("-inf")
        
        m = str(mode).lower()
        if m in ("all", "dict", "full"):
            return {
                "papr": papr_linear,
                "papr_linear": papr_linear,
                "papr_db": papr_db_val,
                "peak_power": peak_power_val,
                "average_power": avg_power_val,
            }

        # Step 5: Convert to decibels if requested: 10 * log10(P_peak / P_avg)
        if db:
            return papr_db_val
            
        return papr_linear


_instance = PAPRMetric()
papr = _instance.compute


def papr_db(iq_data) -> float:
    """Computes Peak-to-Average Power Ratio (PAPR) in decibels (dB)."""
    return _instance.compute(iq_data, db=True)


__all__ = ["PAPRMetric", "papr", "papr_db"]


def main() -> None:
    print("=" * 60)
    print(f"  PAPRMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = PAPRMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
