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


@FrequencyMetricFactory.register(["frequency_drift", "drift_rate", "freq_drift"])
class FrequencyDriftMetric(BaseFrequencyMetric):
    """
    Computes Frequency Drift matching MATLAB reference:
      1. Computes instantaneous frequency with NaN first sample:
         f_inst[0] = NaN, f_inst[1:] = (Fs / (2*pi)) * angle(z[1:] * conj(z[:-1]))
      2. Fits linear trend to finite valid samples only:
         polyfit(t[valid], f_inst[valid], 1)
      3. Total drift = f_fit[-1] - f_fit[0]
      4. Drift rate = slope (Hz/s)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes FrequencyDriftMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up FrequencyDriftMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "frequency_drift"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        return_rate: bool = False,
        mode: str = "drift",
        **kwargs,
    ) -> float | dict:
        # Step 1: Format IQ array and extract required sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=None)
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError("A valid positive sample_rate (Fs) is required for frequency drift calculation.")

        n = z.size
        if n < 3:
            return 0.0

        # Step 2: Instantaneous frequency with first sample excluded as NaN
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

        # Step 3: Linear fit over valid samples only
        slope, intercept = np.polyfit(t_valid, f_valid, 1)

        # Step 4: Fitted endpoints and drift
        f_fit_start = float(intercept + slope * t_valid[0])
        f_fit_end = float(intercept + slope * t_valid[-1])
        total_drift_hz = float(f_fit_end - f_fit_start)
        drift_rate_hz_per_sec = float(slope)

        out_mode = str(mode).lower()
        if return_rate or out_mode in ("all", "dict", "full"):
            return {
                "total_drift_hz": total_drift_hz,
                "drift_rate_hz_per_sec": drift_rate_hz_per_sec,
                "duration_seconds": float(t_valid[-1] - t_valid[0]),
                "start_freq_hz": f_fit_start,
                "end_freq_hz": f_fit_end,
                "unit": "Hz",
            }

        return total_drift_hz


_instance = FrequencyDriftMetric()
frequency_drift = _instance.compute
drift_rate = lambda iq_data, **kw: _instance.compute(iq_data, return_rate=True, **kw)

__all__ = ["FrequencyDriftMetric", "frequency_drift", "drift_rate"]


def main() -> None:
    print("=" * 60)
    print("  FrequencyDriftMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FrequencyDriftMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
