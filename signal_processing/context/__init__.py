"""Analysis context coordinator and caching layer for RF signal events.

This package provides the `SignalContext` abstraction which wraps a single
immutable `SignalEvent` with analysis transform configurations (`AnalysisConfig`)
and lazily computes and memoizes canonical DSP representations (envelope, power,
phase, instantaneous frequency, Fourier transform, PSD, and autocorrelation).
"""

from signal_processing.context.config import AnalysisConfig
from signal_processing.context.exceptions import ContextError
from signal_processing.context.signal_context import SignalContext

__all__ = [
    "AnalysisConfig",
    "ContextError",
    "SignalContext",
]
