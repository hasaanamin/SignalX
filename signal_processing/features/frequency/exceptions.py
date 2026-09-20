"""Exceptions for frequency feature extraction."""

from signal_processing.core import SignalProcessingError


class FrequencyFeatureError(SignalProcessingError):
    """Raised when frequency feature extraction fails or receives invalid input."""
