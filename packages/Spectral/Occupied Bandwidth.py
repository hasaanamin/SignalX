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


@SpectralMetricFactory.register(["occupied_bandwidth", "obw", "occupied_bw", "99_percent_bandwidth", "power_obw"])
class OccupiedBandwidthMetric(BaseSpectralMetric):
    """
    Computes Occupied Bandwidth (OBW) containing beta% (default 99.0%) of total integrated power.
    MATLAB Reference: `obw(x, fs)`.
    Integrates the power spectral density to find the lower frequency f_low and upper frequency f_high
    such that (100 - beta)/2 percent of total power lies below f_low and above f_high.
    OBW = f_high - f_low (Hz).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "occupied_bandwidth"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        percent_power: float = 99.0,
        nperseg: int | None = None,
        window: str = "hann",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"occupied_bandwidth_hz": 0.0, "f_low_hz": 0.0, "f_high_hz": 0.0} if as_dict else 0.0

        freqs, psd = _compute_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
        df = fs / len(psd)

        # Cumulative power distribution (CDF)
        cum_power = np.cumsum(psd * df)
        total_power = cum_power[-1]

        if total_power <= 0:
            return {"occupied_bandwidth_hz": 0.0, "f_low_hz": 0.0, "f_high_hz": 0.0} if as_dict else 0.0

        # Percent bounds
        beta = np.clip(percent_power, 1.0, 99.99)
        tail_pct = (100.0 - beta) / 2.0
        p_low_target = total_power * (tail_pct / 100.0)
        p_high_target = total_power * ((100.0 - tail_pct) / 100.0)

        # Interpolate exact lower boundary f_low
        f_low = float(np.interp(p_low_target, cum_power, freqs))
        # Interpolate exact upper boundary f_high
        f_high = float(np.interp(p_high_target, cum_power, freqs))

        obw_hz = float(max(0.0, f_high - f_low))
        contained_power = float(np.interp(f_high, freqs, cum_power) - np.interp(f_low, freqs, cum_power))

        if as_dict:
            return {
                "occupied_bandwidth_hz": obw_hz,
                "f_low_hz": f_low,
                "f_high_hz": f_high,
                "percent_power": beta,
                "total_power_linear": float(total_power),
                "contained_power_linear": contained_power,
                "fraction_contained": float(contained_power / total_power),
                "center_frequency_hz": float((f_low + f_high) / 2.0),
                "sample_rate": fs,
            }
        return obw_hz


def occupied_bandwidth(
    iq_data: Any,
    sample_rate: float | None = None,
    percent_power: float = 99.0,
    nperseg: int | None = None,
    window: str = "hann",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for Occupied Bandwidth (OBW) calculation.
    """
    metric = OccupiedBandwidthMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, percent_power=percent_power, nperseg=nperseg, window=window, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  OccupiedBandwidthMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = OccupiedBandwidthMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
