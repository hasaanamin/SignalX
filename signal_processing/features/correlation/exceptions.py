"""Exceptions for correlation feature extraction."""

from signal_processing.core import SignalProcessingError


class CorrelationFeatureError(SignalProcessingError):
    """Raised when correlation feature extraction fails or receives invalid input."""
