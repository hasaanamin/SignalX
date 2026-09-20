"""Exceptions for spectral feature extraction."""

from signal_processing.core import SignalProcessingError


class SpectralFeatureError(SignalProcessingError):
    """Raised when spectral feature extraction fails or receives invalid input."""
