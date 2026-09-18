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
from .base import BasePhaseMetric, PhaseMetricFactory
from ._common import _as_iq_array, _extract_sample_rate


@PhaseMetricFactory.register(["phase_deviation", "peak_phase_deviation", "pm_deviation"])
class PhaseDeviationMetric(BasePhaseMetric):
    """
    Computes Phase Deviation relative to the linear carrier trajectory:
      - phi = unwrap(angle(z))
      - trend = polyfit(t, phi, 1)
      - deviation = phi - trend
      - Peak Phase Deviation: max(|deviation|)
      - RMS Phase Deviation: sqrt(mean(deviation^2))
      - STD Phase Deviation: std(deviation)
      - Peak-to-Peak Phase Deviation: max(deviation) - min(deviation)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes PhaseDeviationMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up PhaseDeviationMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "phase_deviation"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        mode: str = "all",
        deg: bool = False,
        **kwargs,
    ) -> float | dict:
        # Step 1: Format IQ array and sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n < 2:
            return 0.0 if mode != "all" else {
                "peak_phase_deviation": 0.0,
                "rms_phase_deviation": 0.0,
                "std_phase_deviation": 0.0,
                "peak_to_peak_phase_deviation": 0.0,
                "unit": "degrees" if deg else "radians",
            }

        # Step 2: Unwrapped phase trajectory and time vector
        phi = np.unwrap(np.angle(z))
        t = np.arange(n) / fs

        # Step 3: Linear trend fit: trend = polyfit(t, phi, 1)
        slope, intercept = np.polyfit(t, phi, 1)
        trend = slope * t + intercept
        
        # Step 4: Detrended phase deviation
        deviation = phi - trend

        # Step 5: Separate quantities
        peak_dev = float(np.max(np.abs(deviation)))
        rms_dev = float(np.sqrt(np.mean(deviation ** 2)))
        std_dev = float(np.std(deviation))
        pp_dev = float(np.max(deviation) - np.min(deviation))

        if deg:
            peak_dev = float(np.degrees(peak_dev))
            rms_dev = float(np.degrees(rms_dev))
            std_dev = float(np.degrees(std_dev))
            pp_dev = float(np.degrees(pp_dev))

        m = str(mode).lower()
        if m in ("peak", "max"):
            return peak_dev
        elif m in ("rms", "root_mean_square"):
            return rms_dev
        elif m in ("std", "standard_deviation"):
            return std_dev
        elif m in ("peak_to_peak", "pp", "p-p", "ptp"):
            return pp_dev
        elif m in ("all", "dict", "full"):
            return {
                "peak_phase_deviation": peak_dev,
                "rms_phase_deviation": rms_dev,
                "std_phase_deviation": std_dev,
                "peak_to_peak_phase_deviation": pp_dev,
                "unit": "degrees" if deg else "radians",
            }
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'peak', 'rms', 'std', 'peak_to_peak', or 'all'.")


_instance = PhaseDeviationMetric()
phase_deviation = _instance.compute

__all__ = ["PhaseDeviationMetric", "phase_deviation"]


def main() -> None:
    print("=" * 60)
    print("  PhaseDeviationMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PhaseDeviationMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
