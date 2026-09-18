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
from .base import BaseFrequencyMetric, FrequencyMetricFactory
from ._common import _as_iq_array, _extract_sample_rate


@FrequencyMetricFactory.register(["frequency_slope", "freq_slope", "f_slope", "frequency_derivative"])
class FrequencySlopeMetric(BaseFrequencyMetric):
    """
    Computes Frequency Slope / Linear Rate of Change matching MATLAB reference:
      1. Computes instantaneous frequency with NaN first sample:
         f_inst[0] = NaN, f_inst[1:] = (Fs / (2*pi)) * angle(z[1:] * conj(z[:-1]))
      2. Fits linear trend to finite valid samples:
         slope, _ = polyfit(t[valid], f_inst[valid], 1)
      3. Returns slope in Hz / s.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes FrequencySlopeMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up FrequencySlopeMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "frequency_slope"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        unit: str = "hz_per_sec",
        **kwargs,
    ) -> float:
        # Step 1: Format IQ array and extract required sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=None)
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError("A valid positive sample_rate (Fs) is required for frequency slope calculation.")

        n = z.size
        if n < 3:
            return 0.0

        # Step 2: Instantaneous frequency with NaN first sample
        phase_step = np.angle(z[1:] * np.conj(z[:-1]))
        f_inst = np.empty(n, dtype=np.float64)
        f_inst[0] = np.nan
        f_inst[1:] = (fs / (2.0 * np.pi)) * phase_step

        t = np.arange(n, dtype=np.float64) / fs
        valid = np.isfinite(f_inst)
        t_valid = t[valid]
        f_valid = f_inst[valid]

        if t_valid.size < 2:
            return 0.0

        # Step 3: Slope from linear fit on valid samples
        slope_hz_per_s, _ = np.polyfit(t_valid, f_valid, 1)
        slope_val = float(slope_hz_per_s)

        # Step 4: Unit conversion
        u = str(unit).lower()
        if u in ("hz_per_sec", "hz/s", "hz"):
            return slope_val
        elif u in ("khz_per_ms", "khz/ms"):
            return slope_val * 1e-6
        elif u in ("mhz_per_us", "mhz/us", "mhz/microsecond"):
            return slope_val * 1e-12
        elif u in ("mhz_per_sec", "mhz/s"):
            return slope_val * 1e-6
        else:
            raise ValueError(f"Unknown unit '{unit}'. Use 'hz_per_sec', 'khz_per_ms', 'mhz_per_us', or 'mhz_per_sec'.")


_instance = FrequencySlopeMetric()
frequency_slope = _instance.compute

__all__ = ["FrequencySlopeMetric", "frequency_slope"]


def main() -> None:
    print("=" * 60)
    print("  FrequencySlopeMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FrequencySlopeMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
