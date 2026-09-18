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


@NoiseMetricFactory.register(["sinr", "signal_to_interference_noise_ratio", "sinr_db", "sinr_estimator"])
class SINRMetric(BaseNoiseMetric):
    """
    Computes Signal-to-Interference-plus-Noise Ratio (SINR):
      SINR = P_signal / (P_interference + P_noise)
      SINR_dB = 10 * log10(P_signal / (P_interference + P_noise))
    Separates discrete interference tones / spurs from the flat background noise floor.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "sinr"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        guard_bins: int = 5,
        spur_threshold_db: float = 6.0,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"sinr_db": -np.inf, "p_signal": 0.0, "p_interference": 0.0, "p_noise": 0.0} if as_dict else -np.inf

        total_pwr = float(np.mean(np.abs(z) ** 2))
        freqs, psd = _compute_psd(z, fs=fs)
        df = fs / len(psd)

        # 1. Identify Fundamental Peak
        peak_idx = int(np.argmax(psd))
        f_fund = float(freqs[peak_idx])

        # Integrate fundamental signal mainlobe
        k_min = max(0, peak_idx - guard_bins)
        k_max = min(len(psd), peak_idx + guard_bins + 1)
        p_signal = float(np.sum(psd[k_min:k_max]) * df)
        p_signal = min(total_pwr, max(0.0, p_signal))

        # 2. Mask fundamental to analyze non-signal spectral bins
        mask = np.ones(len(psd), dtype=bool)
        mask[k_min:k_max] = False
        non_signal_psd = psd[mask]

        if non_signal_psd.size > 0:
            # Median noise floor
            n0_med = float(np.median(non_signal_psd))
            spur_th = n0_med * (10.0 ** (spur_threshold_db / 10.0))

            # Interfering spurs: bins exceeding spur_th
            is_spur = non_signal_psd > spur_th
            p_interf = float(np.sum(non_signal_psd[is_spur]) * df)
            p_noise = float(np.sum(non_signal_psd[~is_spur]) * df)
        else:
            p_interf = 0.0
            p_noise = max(1e-18, total_pwr - p_signal)

        p_noise = max(1e-18, p_noise)
        p_in = max(1e-18, p_interf + p_noise)
        sinr_lin = max(1e-12, p_signal / p_in)
        sinr_db = float(10.0 * np.log10(sinr_lin))

        sir_lin = p_signal / max(1e-18, p_interf) if p_interf > 0 else np.inf
        sir_db = float(10.0 * np.log10(sir_lin)) if p_interf > 0 else 999.0
        snr_lin = p_signal / p_noise
        snr_db = float(10.0 * np.log10(max(1e-12, snr_lin)))

        if as_dict:
            return {
                "sinr_db": sinr_db,
                "sinr_linear": sinr_lin,
                "sir_db": sir_db,
                "snr_db": snr_db,
                "signal_power_linear": p_signal,
                "interference_power_linear": p_interf,
                "noise_power_linear": p_noise,
                "total_power_linear": total_pwr,
                "fundamental_freq_hz": f_fund,
                "sample_rate": fs,
            }
        return sinr_db


def sinr(
    iq_data: Any,
    sample_rate: float | None = None,
    guard_bins: int = 5,
    spur_threshold_db: float = 6.0,
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for SINR calculation.
    """
    metric = SINRMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, guard_bins=guard_bins, spur_threshold_db=spur_threshold_db, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SINRMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SINRMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
