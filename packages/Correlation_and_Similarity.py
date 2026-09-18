from __future__ import annotations
import importlib

_pkg = importlib.import_module("packages.Correlation and Similarity")

BaseCorrelationMetric = _pkg.BaseCorrelationMetric
CorrelationMetricFactory = _pkg.CorrelationMetricFactory
CorrelationResult = _pkg.CorrelationResult
InterPulseCorrelationResult = _pkg.InterPulseCorrelationResult
TemplateMatchResult = _pkg.TemplateMatchResult
SequenceSimilarityResult = _pkg.SequenceSimilarityResult

AutocorrelationMetric = _pkg.AutocorrelationMetric
compute_autocorrelation = _pkg.compute_autocorrelation

CrossCorrelationMetric = _pkg.CrossCorrelationMetric
compute_cross_correlation = _pkg.compute_cross_correlation

LagAnalysisMetric = _pkg.LagAnalysisMetric
compute_lag_analysis = _pkg.compute_lag_analysis
finddelay = _pkg.finddelay

WaveformSimilarityMetric = _pkg.WaveformSimilarityMetric
compute_waveform_similarity = _pkg.compute_waveform_similarity

InterPulseCorrelationMetric = _pkg.InterPulseCorrelationMetric
compute_inter_pulse_correlation = _pkg.compute_inter_pulse_correlation

TemplateMatchingMetric = _pkg.TemplateMatchingMetric
compute_template_matching = _pkg.compute_template_matching
findsignal = _pkg.findsignal

SequenceSimilarityMetric = _pkg.SequenceSimilarityMetric
compute_sequence_similarity = _pkg.compute_sequence_similarity
dtw = _pkg.dtw

StreamingCorrelation = _pkg.StreamingCorrelation
StreamingTemplateMatcher = _pkg.StreamingTemplateMatcher
StreamingInterPulseCorrelation = _pkg.StreamingInterPulseCorrelation
StreamingSequenceSimilarity = _pkg.StreamingSequenceSimilarity

correlation_summary = _pkg.correlation_summary

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
