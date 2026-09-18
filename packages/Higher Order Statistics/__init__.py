"""
Higher Order Statistics Package (Factory & Abstract Base Class Architecture)
Provides an enterprise Factory Pattern with Abstract Base Classes (ABCs) for computing
2nd, 4th, and 6th order complex cumulants, normalized invariant features, and higher-order moment analysis.
"""

from .base import BaseHigherOrderMetric, HigherOrderStatisticsFactory
from ._common import _as_iq_array, _prepare_zero_mean_iq

import importlib

# Dynamic module imports to register all metric classes into the Factory
_c20_mod = importlib.import_module(".C20", package=__name__)
_c21_mod = importlib.import_module(".C21", package=__name__)
_c40_mod = importlib.import_module(".C40", package=__name__)
_c41_mod = importlib.import_module(".C41", package=__name__)
_c42_mod = importlib.import_module(".C42", package=__name__)
_c4x_mod = importlib.import_module(".Fourth-Order Cumulants", package=__name__)
_c6x_mod = importlib.import_module(".Sixth-Order Cumulants", package=__name__)
_norm_mod = importlib.import_module(".Normalized Cumulants", package=__name__)
_analysis_mod = importlib.import_module(".Higher-Order Moment Analysis", package=__name__)

# Exported Concrete Metric Classes
C20Metric = _c20_mod.C20Metric
C21Metric = _c21_mod.C21Metric
C40Metric = _c40_mod.C40Metric
C41Metric = _c41_mod.C41Metric
C42Metric = _c42_mod.C42Metric
FourthOrderCumulantsMetric = _c4x_mod.FourthOrderCumulantsMetric
SixthOrderCumulantsMetric = _c6x_mod.SixthOrderCumulantsMetric
NormalizedCumulantsMetric = _norm_mod.NormalizedCumulantsMetric
HigherOrderMomentAnalysisMetric = _analysis_mod.HigherOrderMomentAnalysisMetric

# Exported Standalone Functional APIs
c20 = _c20_mod.c20
c21 = _c21_mod.c21
c40 = _c40_mod.c40
c41 = _c41_mod.c41
c42 = _c42_mod.c42
fourth_order_cumulants = _c4x_mod.fourth_order_cumulants
sixth_order_cumulants = _c6x_mod.sixth_order_cumulants
normalized_cumulants = _norm_mod.normalized_cumulants
higher_order_moment_analysis = _analysis_mod.higher_order_moment_analysis


def higher_order_statistics_summary(iq_data) -> dict:
    """
    Computes a comprehensive summary of higher-order cumulants and modulation features.
    """
    return higher_order_moment_analysis(iq_data)


__all__ = [
    # Factory & Base
    "BaseHigherOrderMetric",
    "HigherOrderStatisticsFactory",
    # Classes
    "C20Metric",
    "C21Metric",
    "C40Metric",
    "C41Metric",
    "C42Metric",
    "FourthOrderCumulantsMetric",
    "SixthOrderCumulantsMetric",
    "NormalizedCumulantsMetric",
    "HigherOrderMomentAnalysisMetric",
    # Functions
    "_as_iq_array",
    "c20",
    "c21",
    "c40",
    "c41",
    "c42",
    "fourth_order_cumulants",
    "sixth_order_cumulants",
    "normalized_cumulants",
    "higher_order_moment_analysis",
    "higher_order_statistics_summary",
]
