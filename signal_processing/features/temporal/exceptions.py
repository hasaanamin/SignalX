"""Exceptions for temporal feature extraction."""

from signal_processing.core import SignalProcessingError


class TemporalFeatureError(SignalProcessingError):
    """Raised when temporal feature extraction fails or receives invalid input."""
