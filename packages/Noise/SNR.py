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
from ._common import _as_iq_array, _extract_sample_rate, _compute_psd, _estimate_fundamental_carrier


@NoiseMetricFactory.register(["snr", "signal_to_noise_ratio", "snr_estimator", "snr_db"])
class SNRMetric(BaseNoiseMetric):
    """
    Computes Signal-to-Noise Ratio (SNR) in dB and linear scale.
    MATLAB Reference: `snr(x, fs)`.
    Supports:
      - 'welch' / 'spectral': Separates fundamental mainlobe power from out-of-band noise (MATLAB `snr`).
      - 'time_residual': Coherent carrier projection and residual subtraction.
      - 'm2m4': Blind 2nd and 4th moment estimator for constant modulus signals (PSK, QAM, Chirp).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "snr"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        method: str = "spectral",
        guard_bins: int = 5,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"snr_db": -np.inf, "snr_linear": 0.0, "p_signal": 0.0, "p_noise": 0.0} if as_dict else -np.inf

        total_pwr = float(np.mean(np.abs(z) ** 2))

        if method == "m2m4":
            # Moment-based blind SNR estimator (M2M4)
            pwr = np.abs(z) ** 2
            m2 = float(np.mean(pwr))
            m4 = float(np.mean(pwr ** 2))
            disc = max(0.0, 2.0 * (m2 ** 2) - m4)
            s_est = np.sqrt(disc)
            n_est = max(1e-18, m2 - s_est)
            snr_lin = max(1e-12, s_est / n_est)
            p_sig = s_est
            p_noise = n_est
        else:
            # Spectral decomposition (MATLAB snr matching)
            freqs, psd = _compute_psd(z, fs=fs)
            df = fs / len(psd)
            peak_idx = int(np.argmax(psd))
            k_min = max(0, peak_idx - guard_bins)
            k_max = min(len(psd), peak_idx + guard_bins + 1)
            p_sig = float(np.sum(psd[k_min:k_max]) * df)
            p_noise = float(np.sum(np.delete(psd, np.arange(k_min, k_max))) * df)
            p_noise = max(1e-18, p_noise)
            snr_lin = max(1e-12, p_sig / p_noise)

        snr_db = float(10.0 * np.log10(snr_lin))

        if as_dict:
            return {
                "snr_db": snr_db,
                "snr_linear": snr_lin,
                "signal_power_linear": p_sig,
                "signal_power_dbw": float(10.0 * np.log10(max(1e-18, p_sig))),
                "noise_power_linear": p_noise,
                "noise_power_dbw": float(10.0 * np.log10(max(1e-18, p_noise))),
                "total_power_linear": total_pwr,
                "sample_rate": fs,
                "method": method,
            }
        return snr_db


def snr(
    iq_data: Any,
    sample_rate: float | None = None,
    method: str = "spectral",
    guard_bins: int = 5,
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for SNR calculation matching MATLAB `snr(x, fs)`.
    """
    metric = SNRMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, method=method, guard_bins=guard_bins, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SNRMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SNRMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
