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


@SpectralMetricFactory.register(["psd", "power_spectral_density", "welch_psd", "pwelch", "spectral_density"])
class PSDMetric(BaseSpectralMetric):
    """
    Computes Power Spectral Density (PSD) using Welch's averaged modified periodogram method.
    MATLAB Reference: `pwelch`, `periodogram`.
    Supports:
      - Units: 'w_hz' (Watts/Hz), 'dbfs_hz' (dBFS/Hz), 'dbm_hz' (dBm/Hz), 'db_norm' (normalized peak 0 dB).
      - Configurable segment size (nperseg), overlap (noverlap), and window function.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "psd"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        nperseg: int | None = None,
        noverlap: int | None = None,
        window: str = "hann",
        unit: str = "dbfs_hz",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"frequencies": np.array([]), "psd": np.array([])} if as_dict else np.array([])

        freqs, psd_linear = _compute_welch_psd(z, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
        df = fs / len(psd_linear)
        total_integrated_power = float(np.sum(psd_linear) * df)

        # Scale according to requested unit
        unit_clean = unit.lower().strip()
        if unit_clean == "dbm_hz":
            psd_out = 10.0 * np.log10(np.maximum(1e-24, psd_linear)) + 30.0
        elif unit_clean == "db_norm":
            max_p = np.max(psd_linear) if psd_linear.size > 0 else 1.0
            psd_out = 10.0 * np.log10(np.maximum(1e-18, psd_linear / max(1e-18, max_p)))
        elif unit_clean.startswith("db"):
            psd_out = 10.0 * np.log10(np.maximum(1e-24, psd_linear))
        else:  # 'w_hz' or 'linear'
            psd_out = psd_linear

        if as_dict:
            return {
                "frequencies": freqs,
                "psd": psd_out,
                "psd_linear_w_hz": psd_linear,
                "psd_dbfs_hz": 10.0 * np.log10(np.maximum(1e-24, psd_linear)),
                "total_power_linear": total_integrated_power,
                "total_power_dbw": float(10.0 * np.log10(max(1e-18, total_integrated_power))),
                "bin_bandwidth_hz": df,
                "sample_rate": fs,
                "unit": unit,
            }
        return psd_out


def psd(
    iq_data: Any,
    sample_rate: float | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    unit: str = "dbfs_hz",
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:
    """
    Functional DSP wrapper for Power Spectral Density (PSD) calculation.
    """
    metric = PSDMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap, window=window, unit=unit, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  PSDMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PSDMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
