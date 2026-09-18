from __future__ import annotations
import importlib

_pkg = importlib.import_module("packages.Modulation_Analysis")

AMAnalysis = _pkg.AMAnalysis
compute_am_analysis = _pkg.compute_am_analysis
AMAnalysisResult = _pkg.AMAnalysisResult

__all__ = [
    "AMAnalysis",
    "compute_am_analysis",
    "AMAnalysisResult",
]
