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


@CorrelationMetricFactory.register(["autocorrelation", "autocorr", "xcorr_auto", "auto_correlation"])
class AutocorrelationMetric(BaseCorrelationMetric):
    """
    Deterministic Autocorrelation Stage matching MATLAB `xcorr(x)`.

    Mathematical Formulation:
      For complex IQ signal x[n], n = 0 ... N - 1:
        r_x[h] = sum_{n=0}^{N-h-1} x^*[n] * x[n+h],   h >= 0
        r_x[-h] = r_x^*[h]                             h < 0

    Normalization Modes (scaleopt):
      - 'coeff' (default): r_{xx, coeff}[h] = r_xx[h] / r_xx[0],  r_{xx, coeff}[0] = 1.0
      - 'none': Raw deterministic autocorrelation
      - 'biased': (1 / N) * r_xx[h]
      - 'unbiased': (1 / (N - |h|)) * r_xx[h]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "autocorrelation"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
        maxlag: int | None = None,
        scaleopt: str = "coeff",
        method: str = "auto",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> CorrelationResult | dict[str, Any]:
        """
        Computes autocorrelation on complex IQ data.

        Parameters:
            data: Complex IQ array or canonical dictionary.
            sample_rate (Fs): Sampling rate in Hz.
            maxlag: Maximum integer lag to compute (default: N - 1).
            scaleopt: Normalization mode ('coeff', 'none', 'biased', 'unbiased').
            method: Calculation method ('auto', 'fft', 'direct').
            as_dict: If True, returns canonical dictionary. If False, returns CorrelationResult.

        Returns:
            CorrelationResult (or dict if as_dict=True).
        """
        fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
        res = _compute_xcorr_core(
            data,
            None,
            maxlag=maxlag,
            scaleopt=scaleopt,
            method=method,
            sample_rate=fs,
        )
        return res.to_dict() if as_dict else res


def compute_autocorrelation(
    iq_data: Any,
    *,
    sample_rate: float | None = None,
    maxlag: int | None = None,
    scaleopt: str = "coeff",
    method: str = "auto",
    as_dict: bool = False,
    **kwargs: Any,
) -> CorrelationResult | dict[str, Any]:
    """
    Convenience functional API for autocorrelation computation matching MATLAB `xcorr(x)`.
    """
    metric = AutocorrelationMetric()
    return metric.compute(
        iq_data,
        sample_rate=sample_rate,
        maxlag=maxlag,
        scaleopt=scaleopt,
        method=method,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  AutocorrelationMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    metric = AutocorrelationMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
