"""
Statistical Analysis Package (Factory & Abstract Base Class Architecture)
Provides an enterprise Factory Pattern with Abstract Base Classes (ABCs) for statistical
signal analysis, higher-order statistics, and distribution modeling.
"""

try:
    from .base import BaseStatisticalAnalyzer, StatisticalAnalysisFactory
    from ._common import _as_array, _as_real_array
except ImportError:
    from base import BaseStatisticalAnalyzer, StatisticalAnalysisFactory
    from _common import _as_array, _as_real_array

import importlib

# Dynamic module imports to register all analyzers into the Factory
_mean_mod = importlib.import_module(".Mean", package=__name__)
_var_mod = importlib.import_module(".Variance", package=__name__)
_std_mod = importlib.import_module(".Standard Deviation", package=__name__)
_skew_mod = importlib.import_module(".Skewness", package=__name__)
_kurt_mod = importlib.import_module(".Kurtosis", package=__name__)
_ent_mod = importlib.import_module(".Entropy", package=__name__)
_cov_mod = importlib.import_module(".Covariance", package=__name__)
_corr_mod = importlib.import_module(".Correlation", package=__name__)
_hoc_mod = importlib.import_module(".Higher-Order Moments", package=__name__)
_dist_mod = importlib.import_module(".Distribution Analysis", package=__name__)

# Exported Concrete Analyzer Classes
MeanAnalyzer = _mean_mod.MeanAnalyzer
VarianceAnalyzer = _var_mod.VarianceAnalyzer
StandardDeviationAnalyzer = _std_mod.StandardDeviationAnalyzer
SkewnessAnalyzer = _skew_mod.SkewnessAnalyzer
KurtosisAnalyzer = _kurt_mod.KurtosisAnalyzer
EntropyAnalyzer = _ent_mod.EntropyAnalyzer
SpectralEntropyAnalyzer = _ent_mod.SpectralEntropyAnalyzer
CovarianceAnalyzer = _cov_mod.CovarianceAnalyzer
CorrelationAnalyzer = _corr_mod.CorrelationAnalyzer
HigherOrderMomentsAnalyzer = _hoc_mod.HigherOrderMomentsAnalyzer
DistributionAnalyzer = _dist_mod.DistributionAnalyzer

# Exported Standalone Functional APIs
mean = _mean_mod.mean
signal_mean = _mean_mod.signal_mean

variance = _var_mod.variance
signal_variance = _var_mod.signal_variance

standard_deviation = _std_mod.standard_deviation
std = _std_mod.std
signal_std = _std_mod.signal_std

skewness = _skew_mod.skewness

kurtosis = _kurt_mod.kurtosis
excess_kurtosis = _kurt_mod.excess_kurtosis

shannon_entropy = _ent_mod.shannon_entropy
spectral_entropy = _ent_mod.spectral_entropy
entropy = _ent_mod.entropy

covariance = _cov_mod.covariance
iq_covariance_matrix = _cov_mod.iq_covariance_matrix

correlation_coefficient = _corr_mod.correlation_coefficient
cross_correlation = _corr_mod.cross_correlation
auto_correlation = _corr_mod.auto_correlation

raw_moment = _hoc_mod.raw_moment
central_moment = _hoc_mod.central_moment
higher_order_cumulants = _hoc_mod.higher_order_cumulants
cumulants = _hoc_mod.cumulants

distribution_analysis = _dist_mod.distribution_analysis
analyze_distribution = _dist_mod.analyze_distribution


def statistical_summary(data) -> dict:
    """
    Computes a comprehensive statistical summary dictionary using the Factory.
    """
    return {
        "mean": mean(data),
        "variance": variance(data),
        "standard_deviation": standard_deviation(data),
        "skewness": skewness(data),
        "excess_kurtosis": kurtosis(data),
        "shannon_entropy_bits": shannon_entropy(data),
        "spectral_entropy": spectral_entropy(data),
        "cumulants": higher_order_cumulants(data) if _as_array(data).dtype.kind == "c" else None,
    }


__all__ = [
    # Factory & Base
    "BaseStatisticalAnalyzer",
    "StatisticalAnalysisFactory",
    # Classes
    "MeanAnalyzer",
    "VarianceAnalyzer",
    "StandardDeviationAnalyzer",
    "SkewnessAnalyzer",
    "KurtosisAnalyzer",
    "EntropyAnalyzer",
    "SpectralEntropyAnalyzer",
    "CovarianceAnalyzer",
    "CorrelationAnalyzer",
    "HigherOrderMomentsAnalyzer",
    "DistributionAnalyzer",
    # Functions
    "mean",
    "signal_mean",
    "variance",
    "signal_variance",
    "standard_deviation",
    "std",
    "signal_std",
    "skewness",
    "kurtosis",
    "excess_kurtosis",
    "shannon_entropy",
    "spectral_entropy",
    "entropy",
    "covariance",
    "iq_covariance_matrix",
    "correlation_coefficient",
    "cross_correlation",
    "auto_correlation",
    "raw_moment",
    "central_moment",
    "higher_order_cumulants",
    "cumulants",
    "distribution_analysis",
    "analyze_distribution",
    "statistical_summary",
]
