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


@AmplitudeMetricFactory.register(["dynamic_range", "dynamic_range_db", "dr"])
class DynamicRangeMetric(BaseAmplitudeMetric):
    """
    Computes observed amplitude dynamic range:
      Dynamic_Range_linear = Peak_Amplitude / Floor_Amplitude
      Dynamic_Range_dB = 20 * log10(Dynamic_Range_linear)
      
    Modes:
      - mode='min_nonzero': floor is smallest positive non-zero envelope sample.
      - mode='percentile' : floor is the requested positive envelope percentile (e.g. 1%).
      - mode='floor'      : floor is explicit floor_amplitude parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes DynamicRangeMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up DynamicRangeMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "dynamic_range"

    def compute(
        self,
        iq_data,
        *,
        mode: str = "min_nonzero",
        percentile: float = 1.0,
        floor_amplitude: float | None = None,
        db: bool = True,
        **kwargs,
    ) -> float:
        z = _as_iq_array(iq_data)
        
        # Step 1: Calculate instantaneous envelope amplitude |z[n]|
        envelope = np.abs(z)
        
        # Step 2: Calculate peak amplitude
        peak_amp = float(np.max(envelope))
        if peak_amp == 0.0:
            return float("nan") if not db else float("-inf")
        
        # Step 3: Determine floor amplitude based on mode
        m = str(mode).lower()
        if m in ("min_nonzero", "min"):
            positive_samples = envelope[envelope > 0.0]
            if positive_samples.size == 0:
                return float("nan") if not db else float("-inf")
            floor_amp = float(np.min(positive_samples))
        elif m in ("percentile", "pct"):
            pct = float(percentile)
            if not 0.0 <= pct <= 100.0:
                raise ValueError("percentile must be between 0 and 100")
            positive_samples = envelope[envelope > 0.0]
            if positive_samples.size == 0:
                return float("nan") if not db else float("-inf")
            floor_amp = float(np.percentile(positive_samples, pct))
        elif m == "floor":
            if floor_amplitude is None:
                raise ValueError("floor_amplitude is required when mode='floor'")
            floor_amp = float(floor_amplitude)
            if not np.isfinite(floor_amp) or floor_amp <= 0:
                raise ValueError("floor_amplitude must be positive and finite")
        else:
            raise ValueError("mode must be 'min_nonzero', 'percentile', or 'floor'")
        
        if floor_amp == 0.0:
            return float("nan") if not db else float("inf")
        
        # Step 4: Calculate linear dynamic range ratio = peak / floor
        dr_linear = peak_amp / floor_amp
        
        # Step 5: Convert to decibels: 20 * log10(peak / floor)
        if db:
            dr_db = float(20.0 * np.log10(dr_linear))
            return dr_db
            
        return float(dr_linear)


_instance = DynamicRangeMetric()
dynamic_range = _instance.compute


def dynamic_range_db(iq_data) -> float:
    """Computes Dynamic Range in decibels (dB)."""
    return _instance.compute(iq_data, db=True)


__all__ = ["DynamicRangeMetric", "dynamic_range", "dynamic_range_db"]


def main() -> None:
    print("=" * 60)
    print(f"  DynamicRangeMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = DynamicRangeMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
