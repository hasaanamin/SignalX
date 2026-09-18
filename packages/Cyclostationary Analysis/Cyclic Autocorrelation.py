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
from .base import BaseCyclostationaryMetric, CyclostationaryMetricFactory
from ._common import _as_iq_array, _extract_sample_rate


@CyclostationaryMetricFactory.register([
    "cyclic_autocorrelation",
    "cyclic_correlation",
    "cyclic_autocorr",
    "caf",
    "cyclic_acf"
])
class CyclicAutocorrelationMetric(BaseCyclostationaryMetric):
    """
    Cyclic Autocorrelation Function (CAF).
    Measures the correlation between frequency-shifted versions of the signal across lag tau:
      R_x^alpha[ell] = E{ x[n+ell] x*[n] e^{-j 2pi alpha (n + ell/2) / F_s} }
    Finite-sample symmetric time-average estimate:
      hat{R}_x^alpha[ell] = (1 / (N - |ell|)) sum_{n=0}^{N - |ell| - 1} x[n+ell] x*[n] e^{-j 2pi alpha (n + ell/2) / F_s}
    
    Phase center convention:
      For ell >= 0: y[n] = x[n+ell] x*[n], evaluated at midpoint n + ell/2
      For ell = -a < 0: y[n] = x[n] x*[n+a], evaluated at physical midpoint n + a/2
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "cyclic_autocorrelation"

    def compute(self, data: Any, **kwargs: Any) -> np.ndarray | dict[str, Any]:
        """
        Computes the cyclic autocorrelation function over lags -max_lag <= ell <= max_lag.

        Parameters:
            data: Complex IQ array x[n].
            alpha: Target cyclic frequency in Hz (floating-point, default: 0.0).
            max_lag (L): Maximum lag in samples (default: 64).
            sample_rate (Fs): Sampling frequency in Hz (default: 1.0).
            conjugate: If True, computes conjugate CAF E{ x[n+ell] x[n] e^{-j 2pi alpha n} } (for BPSK/DSB).
            as_dict: If True, returns comprehensive dictionary.

        Returns:
            1D complex array of R_x^alpha[ell] (or dictionary if as_dict=True).
        """
        params = {**self._config, **kwargs}
        z = _as_iq_array(data)
        fs = _extract_sample_rate(params, default=1.0)
        alpha = float(params.get("alpha", 0.0))
        max_lag = int(params.get("max_lag", params.get("L", params.get("lags", 64))))
        conjugate = bool(params.get("conjugate", False))

        n_samples = z.size
        if n_samples == 0:
            empty = np.array([], dtype=np.complex128)
            return {"r_alpha": empty, "lags": np.array([])} if params.get("as_dict") else empty

        max_lag = min(max_lag, n_samples - 1)
        lags = np.arange(-max_lag, max_lag + 1, dtype=np.int32)
        r_alpha = np.zeros(len(lags), dtype=np.complex128)

        # Time vector n = 0 ... N-1
        n_vec = np.arange(n_samples, dtype=np.float64)
        carrier = np.exp(-1j * 2.0 * np.pi * alpha * n_vec / fs)

        for idx, ell in enumerate(lags):
            if ell >= 0:
                x_shift = z[ell:]
                x_base = z[: n_samples - ell]
                carrier_slice = carrier[: n_samples - ell]
                phase_mid = np.exp(-1j * np.pi * alpha * ell / fs)
            else:
                abs_ell = abs(ell)
                x_shift = z[: n_samples - abs_ell]
                x_base = z[abs_ell:]
                carrier_slice = carrier[: n_samples - abs_ell]
                # Correct physical midpoint for ell = -a is n + a/2.0
                phase_mid = np.exp(-1j * np.pi * alpha * abs_ell / fs)

            if conjugate:
                product = x_shift * (x_base * carrier_slice) * phase_mid
            else:
                product = x_shift * (np.conj(x_base) * carrier_slice) * phase_mid

            norm_factor = max(1, len(product))
            r_alpha[idx] = np.sum(product) / norm_factor

        if params.get("as_dict", False):
            mag = np.abs(r_alpha)
            phase = np.angle(r_alpha)
            return {
                "r_alpha": r_alpha,
                "magnitude": mag,
                "phase_rad": phase,
                "lags": lags,
                "lags_sec": lags / fs,
                "alpha_hz": alpha,
                "max_lag": max_lag,
                "peak_magnitude": float(np.max(mag)) if mag.size > 0 else 0.0,
                "zero_lag_value": complex(r_alpha[max_lag]) if len(r_alpha) > max_lag else 0.0,
                "sample_rate_hz": fs,
                "conjugate": conjugate,
            }

        return r_alpha


def cyclic_autocorrelation(
    data: Any,
    alpha: float = 0.0,
    max_lag: int = 64,
    sample_rate: float = 1.0,
    as_dict: bool = False,
    **kwargs: Any
) -> np.ndarray | dict[str, Any]:
    """Functional wrapper for CyclicAutocorrelationMetric."""
    metric = CyclicAutocorrelationMetric(alpha=alpha, max_lag=max_lag, sample_rate=sample_rate, as_dict=as_dict, **kwargs)
    return metric.compute(data)


def main() -> None:
    print("=" * 60)
    print("  CyclicAutocorrelationMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = CyclicAutocorrelationMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
