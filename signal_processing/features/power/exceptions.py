"""Exceptions for power-domain feature extraction."""

from signal_processing.core import SignalProcessingError


class PowerFeatureError(SignalProcessingError):
    """Raised when power feature extraction fails or receives invalid input."""
