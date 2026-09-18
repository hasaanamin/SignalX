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


@FrequencyMetricFactory.register(["chirp_rate", "lfm_chirp_rate", "chirp_slope", "lfm_slope"])
class ChirpRateMetric(BaseFrequencyMetric):
    """
    Computes Linear Frequency Modulation (LFM) Radar Chirp Rate matching MATLAB reference:
      1. Direct quadratic fit to unwrapped instantaneous phase:
         p = polyfit(t, unwrap(angle(z)), 2)
         where phi(t) = a*t^2 + b*t + c
      2. chirp_rate = a / pi (Hz / s)
      
    Also estimates Chirp Bandwidth (Hz), Pulse Duration (s), and Chirp Direction (+1 Up, -1 Down).
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes ChirpRateMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up ChirpRateMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "chirp_rate"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        return_info: bool = False,
        mode: str = "chirp_rate",
        **kwargs,
    ) -> float | dict:
        # Step 1: Format IQ array and extract required sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=None)
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError("A valid positive sample_rate (Fs) is required for chirp rate calculation.")

        n = z.size
        if n < 4:
            return 0.0 if not return_info else {
                "chirp_rate_hz_per_sec": 0.0,
                "chirp_bandwidth_hz": 0.0,
                "pulse_duration_seconds": 0.0,
                "direction": "CW",
            }

        # Step 2: Unwrapped instantaneous phase and time vector
        phi = np.unwrap(np.angle(z))
        t = np.arange(n, dtype=np.float64) / fs
        pulse_duration = float(t[-1] - t[0])

        # Step 3: Quadratic fit to phase: phi(t) = a*t^2 + b*t + c
        p = np.polyfit(t, phi, 2)
        a = float(p[0])
        b = float(p[1])
        c = float(p[2])

        # Step 4: Chirp rate mu = a / pi (Hz / s)
        chirp_rate_hz_per_sec = float(a / np.pi)

        # Step 5: Parameters
        f_start = float(b / (2.0 * np.pi))
        f_stop = float(f_start + chirp_rate_hz_per_sec * pulse_duration)
        bandwidth_hz = float(abs(f_stop - f_start))

        if abs(chirp_rate_hz_per_sec) < 1e-3:
            direction = "CW"
        elif chirp_rate_hz_per_sec > 0:
            direction = "UP_CHIRP"
        else:
            direction = "DOWN_CHIRP"

        out_mode = str(mode).lower()
        if return_info or out_mode in ("all", "dict", "full"):
            return {
                "chirp_rate_hz_per_sec": chirp_rate_hz_per_sec,
                "chirp_bandwidth_hz": bandwidth_hz,
                "pulse_duration_seconds": pulse_duration,
                "start_freq_hz": f_start,
                "stop_freq_hz": f_stop,
                "direction": direction,
                "direction_sign": 1 if chirp_rate_hz_per_sec > 0 else (-1 if chirp_rate_hz_per_sec < 0 else 0),
                "poly_a": a,
                "poly_b": b,
                "poly_c": c,
                "unit": "Hz/s",
            }

        return chirp_rate_hz_per_sec


_instance = ChirpRateMetric()
chirp_rate = _instance.compute
lfm_chirp_rate = _instance.compute

__all__ = ["ChirpRateMetric", "chirp_rate", "lfm_chirp_rate"]


def main() -> None:
    print("=" * 60)
    print("  ChirpRateMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = ChirpRateMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
