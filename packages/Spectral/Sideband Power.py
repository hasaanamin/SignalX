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


@SpectralMetricFactory.register(["sideband_power", "usb_lsb_power", "sideband_asymmetry", "sidebands"])
class SidebandPowerMetric(BaseSpectralMetric):
    """
    Computes Sideband Power, Upper Sideband (USB), Lower Sideband (LSB), and Asymmetry:
      - Upper Sideband Power P_usb = int_{f_c + guard}^{f_high} S(f) df
      - Lower Sideband Power P_lsb = int_{f_low}^{f_c - guard} S(f) df
      - Carrier Power P_carrier
      - Sideband Asymmetry Ratio (USB / LSB in dB)
      - Sideband-to-Carrier Ratio (in dBc)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "sideband_power"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        carrier_freq_hz: float | None = None,
        carrier_guard_bins: int = 5,
        nperseg: int | None = None,
        window: str = "hann",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"usb_power_linear": 0.0, "lsb_power_linear": 0.0} if as_dict else 0.0

        freqs, psd = _compute_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
        df = fs / len(psd)

        # 1. Identify carrier center frequency
        if carrier_freq_hz is not None:
            carrier_idx = int(np.argmin(np.abs(freqs - carrier_freq_hz)))
            fc = float(carrier_freq_hz)
        else:
            carrier_idx = int(np.argmax(psd))
            fc = float(freqs[carrier_idx])

        # Carrier mainlobe guard
        k_low = max(0, carrier_idx - carrier_guard_bins)
        k_high = min(len(psd), carrier_idx + carrier_guard_bins + 1)

        p_carrier = float(np.sum(psd[k_low:k_high]) * df)
        p_lsb = float(np.sum(psd[:k_low]) * df)
        p_usb = float(np.sum(psd[k_high:]) * df)
        p_total_sideband = p_lsb + p_usb

        # Ratios
        asymmetry_db = float(10.0 * np.log10(max(1e-18, p_usb) / max(1e-18, p_lsb)))
        sideband_to_carrier_dbc = float(10.0 * np.log10(max(1e-18, p_total_sideband) / max(1e-18, p_carrier)))

        if as_dict:
            return {
                "usb_power_linear": p_usb,
                "usb_power_dbw": float(10.0 * np.log10(max(1e-18, p_usb))),
                "lsb_power_linear": p_lsb,
                "lsb_power_dbw": float(10.0 * np.log10(max(1e-18, p_lsb))),
                "carrier_power_linear": p_carrier,
                "carrier_power_dbw": float(10.0 * np.log10(max(1e-18, p_carrier))),
                "total_sideband_power_linear": p_total_sideband,
                "sideband_asymmetry_db": asymmetry_db,
                "sideband_to_carrier_dbc": sideband_to_carrier_dbc,
                "carrier_frequency_hz": fc,
                "sample_rate": fs,
            }
        return p_total_sideband


def sideband_power(
    iq_data: Any,
    sample_rate: float | None = None,
    carrier_freq_hz: float | None = None,
    carrier_guard_bins: int = 5,
    nperseg: int | None = None,
    window: str = "hann",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for Sideband Power computation.
    """
    metric = SidebandPowerMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, carrier_freq_hz=carrier_freq_hz, carrier_guard_bins=carrier_guard_bins, nperseg=nperseg, window=window, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SidebandPowerMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SidebandPowerMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
