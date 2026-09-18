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
from .base import BaseNoiseMetric, NoiseMetricFactory
from ._common import _as_iq_array, _extract_sample_rate, _compute_psd


@NoiseMetricFactory.register(["noise_floor", "spectral_noise_floor", "noise_density", "n0"])
class NoiseFloorMetric(BaseNoiseMetric):
    """
    Computes spectral Noise Floor and Noise Spectral Density N_0:
      - Average Noise Floor Density in dBFS/Hz or dBm/Hz.
      - Noise Floor level across FFT spectrum bins.
      - Dynamic running noise floor array over time.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "noise_floor"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        nperseg: int | None = None,
        unit: str = "dbfs_hz",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"noise_floor_dbfs_hz": -np.inf, "noise_floor_linear_hz": 0.0} if as_dict else -np.inf

        freqs, psd = _compute_psd(z, fs=fs, nperseg=nperseg)

        # Robust noise floor baseline across spectrum (median of lower 50% spectral bins)
        p50 = np.percentile(psd, 50)
        noise_bins = psd[psd <= p50]
        if noise_bins.size > 0:
            n0_linear_hz = float(np.median(noise_bins))
        else:
            n0_linear_hz = float(np.median(psd))

        n0_linear_hz = max(1e-24, n0_linear_hz)
        n0_dbfs_hz = float(10.0 * np.log10(n0_linear_hz))
        n0_dbm_hz = float(n0_dbfs_hz + 30.0)

        # Average noise floor per FFT bin
        df = fs / len(psd)
        bin_noise_power_linear = n0_linear_hz * df
        bin_noise_power_db = float(10.0 * np.log10(max(1e-24, bin_noise_power_linear)))

        if as_dict:
            return {
                "noise_floor_dbfs_hz": n0_dbfs_hz,
                "noise_floor_dbm_hz": n0_dbm_hz,
                "noise_floor_linear_hz": n0_linear_hz,
                "bin_noise_power_db": bin_noise_power_db,
                "bin_bandwidth_hz": df,
                "sample_rate": fs,
                "frequencies": freqs,
                "psd": psd,
            }
        return n0_dbfs_hz if unit.lower().startswith("db") else n0_linear_hz


def noise_floor(
    iq_data: Any,
    sample_rate: float | None = None,
    nperseg: int | None = None,
    unit: str = "dbfs_hz",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for noise floor estimation.
    """
    metric = NoiseFloorMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, nperseg=nperseg, unit=unit, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  NoiseFloorMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = NoiseFloorMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
