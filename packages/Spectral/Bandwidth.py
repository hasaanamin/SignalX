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

from typing import Any
import numpy as np
from .base import BaseSpectralMetric, SpectralMetricFactory
from ._common import _as_iq_array, _extract_sample_rate, _compute_welch_psd


@SpectralMetricFactory.register(["bandwidth", "power_bandwidth", "powerbw", "x_db_bandwidth", "3db_bandwidth"])
class BandwidthMetric(BaseSpectralMetric):
    """
    Computes X-dB (default 3-dB, 10-dB, 20-dB) Power Bandwidth:
      - Finds peak spectral power P_max.
      - Finds lower frequency f_low and upper frequency f_high where P(f) drops by x_db (default 3.0 dB).
      - BW = f_high - f_low (Hz).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "bandwidth"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        x_db: float = 3.0,
        nperseg: int | None = None,
        window: str = "hann",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"bandwidth_hz": 0.0, "f_low_hz": 0.0, "f_high_hz": 0.0} if as_dict else 0.0

        freqs, psd = _compute_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
        if psd.size == 0 or np.max(psd) <= 0:
            return {"bandwidth_hz": 0.0, "f_low_hz": 0.0, "f_high_hz": 0.0} if as_dict else 0.0

        peak_idx = int(np.argmax(psd))
        peak_pwr = psd[peak_idx]
        target_pwr = peak_pwr * (10.0 ** (-abs(x_db) / 10.0))

        # 1. Find lower and upper frequency boundaries where PSD >= target_pwr (MATLAB powerbw)
        above_mask = psd >= target_pwr
        if not np.any(above_mask):
            return {"bandwidth_hz": 0.0, "f_low_hz": float(freqs[peak_idx]), "f_high_hz": float(freqs[peak_idx])} if as_dict else 0.0

        indices = np.where(above_mask)[0]
        i_low = indices[0]
        i_high = indices[-1]

        # Interpolate exact lower cutoff f_low
        if i_low > 0:
            p1, p2 = psd[i_low - 1], psd[i_low]
            f1, f2 = freqs[i_low - 1], freqs[i_low]
            denom = p2 - p1
            frac = (target_pwr - p1) / denom if abs(denom) > 1e-24 else 0.0
            f_low = float(f1 + frac * (f2 - f1))
        else:
            f_low = float(freqs[i_low])

        # Interpolate exact upper cutoff f_high
        if i_high < len(psd) - 1:
            p1, p2 = psd[i_high], psd[i_high + 1]
            f1, f2 = freqs[i_high], freqs[i_high + 1]
            denom = p2 - p1
            frac = (target_pwr - p1) / denom if abs(denom) > 1e-24 else 0.0
            f_high = float(f1 + frac * (f2 - f1))
        else:
            f_high = float(freqs[i_high])

        bw_hz = float(max(0.0, f_high - f_low))

        if as_dict:
            return {
                "bandwidth_hz": bw_hz,
                "f_low_hz": float(f_low),
                "f_high_hz": float(f_high),
                "peak_freq_hz": float(freqs[peak_idx]),
                "peak_power_linear": float(peak_pwr),
                "peak_power_dbfs": float(10.0 * np.log10(max(1e-24, peak_pwr))),
                "threshold_db": float(x_db),
                "sample_rate": fs,
            }
        return bw_hz


def bandwidth(
    iq_data: Any,
    sample_rate: float | None = None,
    x_db: float = 3.0,
    nperseg: int | None = None,
    window: str = "hann",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for X-dB Power Bandwidth estimation.
    """
    metric = BandwidthMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, x_db=x_db, nperseg=nperseg, window=window, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  BandwidthMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = BandwidthMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
