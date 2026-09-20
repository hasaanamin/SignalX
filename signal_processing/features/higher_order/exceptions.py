"""Exceptions for higher-order statistics feature extraction."""

from signal_processing.core import SignalProcessingError


class HigherOrderFeatureError(SignalProcessingError):
    """Raised when higher-order statistics feature extraction fails or receives invalid input."""
