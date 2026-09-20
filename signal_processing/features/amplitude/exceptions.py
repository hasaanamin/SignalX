"""Exceptions for amplitude feature extraction."""

from signal_processing.core import SignalProcessingError


class AmplitudeFeatureError(SignalProcessingError):
    """Raised when amplitude feature extraction fails or receives invalid input."""
