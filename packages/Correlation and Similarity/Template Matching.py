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

from .base import BaseCorrelationMetric, CorrelationMetricFactory, TemplateMatchResult
from ._common import _as_iq_array, _extract_sample_rate, _compute_findsignal_core, _compute_xcorr_core


@CorrelationMetricFactory.register(["template_matching", "findsignal", "template_match", "signal_matching"])
class TemplateMatchingMetric(BaseCorrelationMetric):
    """
    Template Matching Stage matching MATLAB `findsignal(data, signal)`.

    MATLAB Mathematical Formulation:
      For template s[n], n = 0 ... M - 1, and candidate location i in data x[n]:
        - 'squared' (default): D[i] = sum_{n=0}^{M-1} |x[i+n] - s[n]|^2
        - 'euclidean':         D[i] = sqrt(sum_{n=0}^{M-1} |x[i+n] - s[n]|^2)
        - 'absolute':          D[i] = sum_{n=0}^{M-1} |x[i+n] - s[n]|

      Optimal Location:
        i_best = argmin_i D[i]
        i_stop = i_best + M - 1

    Normalization Modes:
      - 'none' (default): Raw complex sample values
      - 'center': Mean centered: x_i - mu(x_i), s - mu(s)
      - 'power':  L2 unit power normalized: x_i / ||x_i||_2, s / ||s||_2
      - 'zscore': Standardized: (x_i - mu) / sigma

    Alternative Method:
      - 'normalized_correlation': Standard RF matched filter normalized cross-correlation peak.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "template_matching"

    def compute(
        self,
        data: Any,
        template: Any,
        *,
        sample_rate: float | None = None,
        template_method: str = "matlab_findsignal",
        metric: str = "squared",
        normalization: str = "none",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> TemplateMatchResult | dict[str, Any]:
        """
        Locates the best matching occurrence of a template waveform within data.

        Parameters:
            data: Complex IQ data stream/buffer x[n].
            template: Complex IQ reference template waveform s[n].
            sample_rate (Fs): Sampling rate in Hz.
            template_method: 'matlab_findsignal' (default, distance minimization) or 'normalized_correlation'.
            metric: Distance metric ('squared', 'euclidean', 'absolute').
            normalization: Pre-normalization ('none', 'center', 'power', 'zscore').
            as_dict: If True, returns canonical dictionary. If False, returns TemplateMatchResult.

        Returns:
            TemplateMatchResult (or dict if as_dict=True).
        """
        fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
        method_clean = str(template_method).lower().strip().replace("-", "_")

        if method_clean in ("normalized_correlation", "correlation", "matched_filter", "xcorr"):
            # Conventional RF matched-waveform correlation
            x_arr = _as_iq_array(data)
            s_arr = _as_iq_array(template)
            corr_res = _compute_xcorr_core(x_arr, s_arr, scaleopt="coeff", sample_rate=fs)

            # In xcorr(x, s), the peak lag m_p corresponds to the start offset of s in x
            # Since R_xs[m] = sum_n x[n+m] s^*[n], s starts at x[m_p] when m_p >= 0
            best_start = max(0, corr_res.peak_lag_samples)
            m_len = len(s_arr)
            best_stop = min(len(x_arr) - 1, best_start + m_len - 1)

            res = TemplateMatchResult(
                best_start_sample=best_start,
                best_stop_sample=best_stop,
                best_start_sec=float(best_start / fs),
                best_stop_sec=float(best_stop / fs),
                minimum_distance=float(1.0 - corr_res.peak_magnitude),  # 1 - peak correlation
                distance_profile=1.0 - corr_res.correlation_magnitude,
                matched_segment=x_arr[best_start : best_stop + 1].copy(),
                template_method="normalized_correlation",
                metric="correlation_distance",
                normalization="coeff",
                sample_rate=fs,
            )
        else:
            # Default MATLAB findsignal distance minimization
            res = _compute_findsignal_core(
                data,
                template,
                metric=metric,
                normalization=normalization,
                sample_rate=fs,
            )

        return res.to_dict() if as_dict else res


def compute_template_matching(
    data: Any,
    template: Any,
    *,
    sample_rate: float | None = None,
    template_method: str = "matlab_findsignal",
    metric: str = "squared",
    normalization: str = "none",
    as_dict: bool = False,
    **kwargs: Any,
) -> TemplateMatchResult | dict[str, Any]:
    """
    Convenience functional API for template matching matching MATLAB `findsignal`.
    """
    stage = TemplateMatchingMetric()
    return stage.compute(
        data,
        template,
        sample_rate=sample_rate,
        template_method=template_method,
        metric=metric,
        normalization=normalization,
        as_dict=as_dict,
        **kwargs,
    )


def findsignal(
    data: Any,
    template: Any,
    *,
    sample_rate: float | None = None,
    metric: str = "squared",
    normalization: str = "none",
    **kwargs: Any,
) -> tuple[int, int]:
    """
    Exact MATLAB `findsignal(data, signal)` wrapper returning (i_best, i_stop).
    """
    res = compute_template_matching(
        data,
        template,
        sample_rate=sample_rate,
        template_method="matlab_findsignal",
        metric=metric,
        normalization=normalization,
        as_dict=False,
        **kwargs,
    )
    return (res.best_start_sample, res.best_stop_sample)


def main() -> None:
    print("=" * 60)
    print("  TemplateMatchingMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    s1 = np.exp(1j * 2 * np.pi * 50e3 * t)
    s2 = np.roll(s1, 10) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = TemplateMatchingMetric()
    result = metric.compute(s1, s2, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
