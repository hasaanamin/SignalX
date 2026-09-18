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


@AmplitudeMetricFactory.register(["crest_factor", "crest_factor_db", "cf"])
class CrestFactorMetric(BaseAmplitudeMetric):
    """
    Computes Crest Factor = Peak Amplitude / RMS Amplitude:
      CF_linear = max(|z[n]|) / sqrt(mean(|z[n]|^2))
      CF_dB = 20 * log10(CF_linear)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes CrestFactorMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up CrestFactorMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "crest_factor"

    def compute(self, iq_data, *, db: bool = False, **kwargs) -> float:
        z = _as_iq_array(iq_data)
        
        # Step 1: Calculate instantaneous envelope amplitude |z[n]|
        envelope = np.abs(z)
        
        # Step 2: Calculate peak amplitude
        peak_amp = float(np.max(envelope))
        
        # Step 3: Calculate RMS amplitude = sqrt(mean(|z|^2))
        power_samples = envelope ** 2
        mean_power = np.mean(power_samples)
        rms_amp = float(np.sqrt(mean_power))
        
        # Step 4: Handle zero signal / silence
        if rms_amp == 0.0:
            return float("nan") if not db else float("-inf")
        
        # Step 5: Linear Crest Factor ratio
        cf_linear = peak_amp / rms_amp
        
        # Step 6: Convert to decibels if requested: 20 * log10(peak / rms)
        if db:
            cf_db = float(20.0 * np.log10(cf_linear))
            return cf_db
            
        return float(cf_linear)


_instance = CrestFactorMetric()
crest_factor = _instance.compute


def crest_factor_db(iq_data) -> float:
    """Computes Crest Factor in decibels (dB)."""
    return _instance.compute(iq_data, db=True)


__all__ = ["CrestFactorMetric", "crest_factor", "crest_factor_db"]


def main() -> None:
    print("=" * 60)
    print(f"  CrestFactorMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = CrestFactorMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
