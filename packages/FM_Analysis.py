from __future__ import annotations
import importlib

_pkg = importlib.import_module("packages.Modulation_Analysis")

FMAnalysis = _pkg.FMAnalysis
compute_fm_analysis = _pkg.compute_fm_analysis
FMAnalysisResult = _pkg.FMAnalysisResult

__all__ = [
    "FMAnalysis",
    "compute_fm_analysis",
    "FMAnalysisResult",
]
