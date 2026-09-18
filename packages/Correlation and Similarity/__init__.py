from __future__ import annotations
from typing import Any
import numpy as np

from .base import (
    BaseCorrelationMetric,
    CorrelationMetricFactory,
    CorrelationResult,
    InterPulseCorrelationResult,
    TemplateMatchResult,
    SequenceSimilarityResult,
)
from ._common import (
    _as_iq_array,
    _extract_sample_rate,
    _compute_xcorr_core,
    _compute_findsignal_core,
    _compute_dtw_core,
)
from .Autocorrelation import AutocorrelationMetric, compute_autocorrelation
from .Cross_Correlation import CrossCorrelationMetric, compute_cross_correlation
from .Lag_Analysis import LagAnalysisMetric, compute_lag_analysis, finddelay
from .Waveform_Similarity import WaveformSimilarityMetric, compute_waveform_similarity
from .Inter_Pulse_Correlation import InterPulseCorrelationMetric, compute_inter_pulse_correlation
from .Template_Matching import TemplateMatchingMetric, compute_template_matching, findsignal
from .Sequence_Similarity import SequenceSimilarityMetric, compute_sequence_similarity, dtw
from .streaming import (
    StreamingCorrelation,
    StreamingTemplateMatcher,
    StreamingInterPulseCorrelation,
    StreamingSequenceSimilarity,
)


def correlation_summary(
    x: Any,
    y: Any = None,
    *,
    template: Any = None,
    pulse_list: list[Any] | None = None,
    feature_seq_a: Any = None,
    feature_seq_b: Any = None,
    sample_rate: float | None = None,
    maxlag: int | None = None,
    scaleopt: str = "coeff",
    method: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Unified High-Performance Correlation and Similarity Analysis Pipeline.
    Executes the shared core correlation engine once and derives all correlation-based metrics
    (Autocorrelation, Cross-Correlation, Waveform Similarity, Lag Analysis) with zero duplicate transforms,
    along with template matching and sequence similarity if templates/sequences are provided.

    Outputs:
      1. Autocorrelation (r_x[h])
      2. Cross-Correlation (R_xy[m], rho_xy[m])
      3. Waveform Similarity (S_waveform in [0, 1])
      4. Lag Analysis (MATLAB finddelay convention: D = -m_p, delay in sec)
      5. Inter-Pulse Correlation (pairwise correlation matrix or candidate pair)
      6. Template Matching (MATLAB findsignal squared Euclidean distance)
      7. Sequence Similarity (MATLAB dtw dynamic time warping)

    Parameters:
        x: First complex IQ signal (or canonical dictionary).
        y: Optional second complex IQ signal.
        template: Optional reference template waveform for findsignal matching.
        pulse_list: Optional list of pulse IQ arrays/PulseCandidates for inter-pulse correlation matrix.
        feature_seq_a: Optional first multidimensional pulse feature sequence.
        feature_seq_b: Optional second multidimensional pulse feature sequence.
        sample_rate (Fs): Sampling rate in Hz.
        maxlag: Maximum lag search range.
        scaleopt: Normalization mode ('coeff', 'none', 'biased', 'unbiased').
        method: Calculation method ('auto', 'fft', 'direct').

    Returns:
        Comprehensive dictionary containing all computed metrics, results, and alignments.
    """
    fs = _extract_sample_rate(data=x, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
    zx = _as_iq_array(x)

    # 1. Autocorrelation
    auto_res = _compute_xcorr_core(
        zx,
        None,
        maxlag=maxlag,
        scaleopt=scaleopt,
        method=method,
        sample_rate=fs,
    )

    out: dict[str, Any] = {
        "sample_rate_hz": fs,
        "num_samples_x": len(zx),
        "autocorrelation": auto_res.to_dict(),
    }

    # 2. Cross-Correlation, Waveform Similarity, and Lag Analysis (computed once via common xcorr engine)
    if y is not None:
        zy = _as_iq_array(y)
        out["num_samples_y"] = len(zy)
        cross_res = _compute_xcorr_core(
            zx,
            zy,
            maxlag=maxlag,
            scaleopt=scaleopt,
            method=method,
            sample_rate=fs,
        )
        out["cross_correlation"] = cross_res.to_dict()

        # Waveform similarity reuses cross_res directly
        sim_metric = WaveformSimilarityMetric()
        out["waveform_similarity"] = sim_metric.compute(cross_res, as_dict=True)

        # Lag analysis reuses cross_res directly
        lag_metric = LagAnalysisMetric()
        out["lag_analysis"] = lag_metric.compute(cross_res, as_dict=True)

    # 3. Inter-Pulse Correlation
    if pulse_list is not None and len(pulse_list) > 0:
        inter_metric = InterPulseCorrelationMetric()
        out["inter_pulse_correlation"] = inter_metric.compute(
            pulse_list[0],
            pulse_list=pulse_list,
            sample_rate=fs,
            maxlag=maxlag,
            method=method,
            as_dict=True,
        )

    # 4. Template Matching (MATLAB findsignal)
    if template is not None:
        match_metric = TemplateMatchingMetric()
        out["template_matching"] = match_metric.compute(
            zx,
            template,
            sample_rate=fs,
            as_dict=True,
        )

    # 5. Sequence Similarity (MATLAB dtw)
    if feature_seq_a is not None and feature_seq_b is not None:
        seq_metric = SequenceSimilarityMetric()
        out["sequence_similarity"] = seq_metric.compute(
            feature_seq_a,
            feature_seq_b,
            as_dict=True,
        )

    return out


__all__ = [
    "BaseCorrelationMetric",
    "CorrelationMetricFactory",
    "CorrelationResult",
    "InterPulseCorrelationResult",
    "TemplateMatchResult",
    "SequenceSimilarityResult",
    "AutocorrelationMetric",
    "compute_autocorrelation",
    "CrossCorrelationMetric",
    "compute_cross_correlation",
    "LagAnalysisMetric",
    "compute_lag_analysis",
    "finddelay",
    "WaveformSimilarityMetric",
    "compute_waveform_similarity",
    "InterPulseCorrelationMetric",
    "compute_inter_pulse_correlation",
    "TemplateMatchingMetric",
    "compute_template_matching",
    "findsignal",
    "SequenceSimilarityMetric",
    "compute_sequence_similarity",
    "dtw",
    "StreamingCorrelation",
    "StreamingTemplateMatcher",
    "StreamingInterPulseCorrelation",
    "StreamingSequenceSimilarity",
    "correlation_summary",
]
