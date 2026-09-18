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


@NoiseMetricFactory.register(["noise_power", "p_noise", "noise_power_estimator", "noisepower"])
class NoisePowerMetric(BaseNoiseMetric):
    """
    Computes total integrated Noise Power (P_n) for complex baseband IQ signals.
    Supports:
      - 'spectral': Out-of-band spectral integration excluding the fundamental signal peak.
      - 'time_residual': Reconstruction residual variance: mean(|z[n] - s_fund[n]|^2).
      - 'robust_quantile': Lower 35% trimmed exponential quantile scaling for unknown dense signals.
      - 'direct': Direct sample power mean(|w[n]|^2) when applied to pure noise segments.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "noise_power"

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
            return {"noise_power_linear": 0.0, "noise_power_dbw": -np.inf, "noise_power_dbm": -np.inf} if as_dict else 0.0

        p_total = float(np.mean(np.abs(z) ** 2))

        if method == "robust_quantile":
            pwr = np.abs(z) ** 2
            p35 = np.percentile(pwr, 35)
            low_slice = pwr[pwr <= p35]
            if low_slice.size > 0:
                p_noise = float(np.median(low_slice) / 0.1923)
            else:
                p_noise = float(np.median(pwr) / 0.69315)
        elif method == "time_residual":
            _, _, noise_res, _ = _estimate_fundamental_carrier(z, fs=fs, guard_bins=guard_bins)
            p_noise = float(np.mean(np.abs(noise_res) ** 2))
        elif method == "direct":
            p_noise = p_total
        else:  # "spectral" (default MATLAB matching)
            _, p_sig, noise_res, _ = _estimate_fundamental_carrier(z, fs=fs, guard_bins=guard_bins)
            p_noise = max(1e-18, p_total - p_sig)
            if p_noise <= 1e-18:
                p_noise = float(np.mean(np.abs(noise_res) ** 2))

        p_noise = max(1e-18, min(p_total, p_noise))
        p_noise_dbw = float(10.0 * np.log10(p_noise))
        p_noise_dbm = float(p_noise_dbw + 30.0)

        if as_dict:
            return {
                "noise_power_linear": p_noise,
                "noise_power_dbw": p_noise_dbw,
                "noise_power_dbm": p_noise_dbm,
                "total_power_linear": p_total,
                "total_power_dbw": float(10.0 * np.log10(max(1e-18, p_total))),
                "sample_rate": fs,
                "method": method,
            }
        return p_noise


def noise_power(
    iq_data: Any,
    sample_rate: float | None = None,
    method: str = "spectral",
    guard_bins: int = 5,
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for noise power estimation.
    """
    metric = NoisePowerMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, method=method, guard_bins=guard_bins, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  NoisePowerMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = NoisePowerMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
