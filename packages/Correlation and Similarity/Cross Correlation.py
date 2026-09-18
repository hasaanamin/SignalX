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

from .base import BaseCorrelationMetric, CorrelationMetricFactory, CorrelationResult
from ._common import _extract_sample_rate, _compute_xcorr_core


@CorrelationMetricFactory.register(["cross_correlation", "crosscorr", "xcorr", "cross_corr"])
class CrossCorrelationMetric(BaseCorrelationMetric):
    """
    Deterministic Cross-Correlation Stage matching MATLAB `xcorr(x, y)`.

    Mathematical Formulation:
      For complex IQ signals x[n] and y[n]:
        R_xy[m] = sum_n x[n+m] * y^*[n],   m >= 0
        R_xy[-m] = R_yx^*[m],              m < 0

    Normalization Modes (scaleopt):
      - 'coeff' (default): rho_xy[m] = R_xy[m] / sqrt(R_xx[0] * R_yy[0])  in [-1, 1]
      - 'none': Raw deterministic cross-correlation
      - 'biased': (1 / max(Nx, Ny)) * R_xy[m]
      - 'unbiased': (1 / (max(Nx, Ny) - |m|)) * R_xy[m]

    Computation Methods:
      - 'auto' (default): Uses direct convolution for small N, FFT for N >= 64
      - 'fft': Frequency-domain IFFT(X(f) * Y^*(f))
      - 'direct': Time-domain discrete summation
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "cross_correlation"

    def compute(
        self,
        data: Any,
        y: Any = None,
        *,
        sample_rate: float | None = None,
        maxlag: int | None = None,
        scaleopt: str = "coeff",
        method: str = "auto",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> CorrelationResult | dict[str, Any]:
        """
        Computes cross-correlation between two complex IQ signals.

        Parameters:
            data: First complex IQ signal x[n] (or canonical dictionary / pulse object).
            y: Second complex IQ signal y[n]. If None, computes autocorrelation.
            sample_rate (Fs): Sampling rate in Hz.
            maxlag: Maximum integer lag to compute (default: max(Nx, Ny) - 1).
            scaleopt: Normalization mode ('coeff', 'none', 'biased', 'unbiased').
            method: Calculation method ('auto', 'fft', 'direct').
            as_dict: If True, returns canonical dictionary. If False, returns CorrelationResult.

        Returns:
            CorrelationResult (or dict if as_dict=True).
        """
        fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
        res = _compute_xcorr_core(
            data,
            y,
            maxlag=maxlag,
            scaleopt=scaleopt,
            method=method,
            sample_rate=fs,
        )
        return res.to_dict() if as_dict else res


def compute_cross_correlation(
    x: Any,
    y: Any = None,
    *,
    sample_rate: float | None = None,
    maxlag: int | None = None,
    scaleopt: str = "coeff",
    method: str = "auto",
    as_dict: bool = False,
    **kwargs: Any,
) -> CorrelationResult | dict[str, Any]:
    """
    Convenience functional API for cross-correlation computation matching MATLAB `xcorr(x, y)`.
    """
    metric = CrossCorrelationMetric()
    return metric.compute(
        x,
        y,
        sample_rate=sample_rate,
        maxlag=maxlag,
        scaleopt=scaleopt,
        method=method,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  CrossCorrelationMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    s1 = np.exp(1j * 2 * np.pi * 50e3 * t)
    s2 = np.roll(s1, 10) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = CrossCorrelationMetric()
    result = metric.compute(s1, s2, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
