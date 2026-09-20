"""Exceptions for bandwidth feature extraction."""

from signal_processing.core import SignalProcessingError


class BandwidthFeatureError(SignalProcessingError):
    """Raised when bandwidth feature extraction fails or receives invalid input."""
