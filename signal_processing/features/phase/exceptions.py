"""Exceptions for phase feature extraction."""

from signal_processing.core import SignalProcessingError


class PhaseFeatureError(SignalProcessingError):
    """Raised when phase feature extraction fails or receives invalid input."""
