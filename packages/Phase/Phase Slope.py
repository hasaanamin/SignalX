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
from ._common import _as_iq_array


@PhaseMetricFactory.register(["phase_slope", "phase_rate_of_change", "dphi_dt"])
class PhaseSlopeMetric(BasePhaseMetric):
    """
    Computes Phase Slope / Linear Rate of Change:
      d(phi) / dt = slope from polyfit(t, unwrap(angle(z)), 1) where t = arange(N) / Fs
      
    Requires the actual sample rate Fs.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes PhaseSlopeMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up PhaseSlopeMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "phase_slope"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        unit: str = "rad_per_sec",
        **kwargs,
    ) -> float:
        # Step 1: Format IQ array
        z = _as_iq_array(iq_data)
        n = z.size
        if n < 2:
            return 0.0

        # Step 2: Require actual sample rate Fs (do not silently assume Fs=1)
        fs = sample_rate
        if fs is None and isinstance(iq_data, dict):
            fs = iq_data.get("sample_rate") or iq_data.get("sampling_rate") or iq_data.get("fs")
        if fs is None:
            raise ValueError("sample_rate is required for phase slope calculation")
        fs = float(fs)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("sample_rate must be a positive finite number")

        # Step 3: Unwrapped phase against time grid t = arange(N) / Fs
        phi = np.unwrap(np.angle(z))
        t = np.arange(n) / fs

        # Step 4: Slope from linear fit of unwrapped phase against t
        slope_rad_per_s, _ = np.polyfit(t, phi, 1)
        slope_val = float(slope_rad_per_s)

        # Step 5: Unit conversion
        u = str(unit).lower()
        if u in ("rad_per_sec", "rad/s", "rad"):
            return slope_val
        elif u in ("deg_per_sec", "deg/s", "degrees_per_sec"):
            return float(np.degrees(slope_val))
        elif u in ("hz", "frequency", "cycles_per_sec"):
            return float(slope_val / (2.0 * np.pi))
        else:
            raise ValueError(f"Unknown unit '{unit}'. Use 'rad_per_sec', 'deg_per_sec', or 'hz'.")


_instance = PhaseSlopeMetric()
phase_slope = _instance.compute

__all__ = ["PhaseSlopeMetric", "phase_slope"]


def main() -> None:
    print("=" * 60)
    print("  PhaseSlopeMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PhaseSlopeMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
