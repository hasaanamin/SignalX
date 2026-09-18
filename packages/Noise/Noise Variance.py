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
from ._common import _as_iq_array, _extract_sample_rate, _estimate_fundamental_carrier


@NoiseMetricFactory.register(["noise_variance", "variance_noise", "sigma_noise", "noise_var"])
class NoiseVarianceMetric(BaseNoiseMetric):
    """
    Computes complex noise variance:
      - Total variance: sigma_w^2 = E[|w - mu_w|^2] = sigma_I^2 + sigma_Q^2
      - In-phase variance: sigma_I^2 = Var(Re(w))
      - Quadrature variance: sigma_Q^2 = Var(Im(w))
      - Robust MAD variance: sigma_mad^2 = (1.4826 * MAD)^2
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "noise_variance"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        method: str = "residual",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"noise_variance": 0.0, "var_i": 0.0, "var_q": 0.0, "sigma": 0.0} if as_dict else 0.0

        if method == "direct":
            w = z
        else:
            _, _, w, _ = _estimate_fundamental_carrier(z, fs=fs)

        i_ch = np.real(w)
        q_ch = np.imag(w)

        var_i = float(np.var(i_ch))
        var_q = float(np.var(q_ch))
        var_total = float(var_i + var_q)
        sigma = float(np.sqrt(max(0.0, var_total)))

        # Robust MAD estimate (Median Absolute Deviation)
        mad_i = float(np.median(np.abs(i_ch - np.median(i_ch))))
        mad_q = float(np.median(np.abs(q_ch - np.median(q_ch))))
        sigma_mad_i = 1.4826 * mad_i
        sigma_mad_q = 1.4826 * mad_q
        var_mad = float((sigma_mad_i ** 2) + (sigma_mad_q ** 2))

        if as_dict:
            return {
                "noise_variance": var_total,
                "sigma_total": sigma,
                "variance_i": var_i,
                "variance_q": var_q,
                "sigma_i": float(np.sqrt(max(0.0, var_i))),
                "sigma_q": float(np.sqrt(max(0.0, var_q))),
                "robust_mad_variance": var_mad,
                "mean_offset_i": float(np.mean(i_ch)),
                "mean_offset_q": float(np.mean(q_ch)),
                "iq_balance_ratio": float(var_i / max(1e-18, var_q)),
            }
        return var_total


def noise_variance(
    iq_data: Any,
    sample_rate: float | None = None,
    method: str = "residual",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for noise variance calculation.
    """
    metric = NoiseVarianceMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, method=method, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  NoiseVarianceMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = NoiseVarianceMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
