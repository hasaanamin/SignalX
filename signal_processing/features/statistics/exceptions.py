"""Exceptions for statistics feature extraction."""

from signal_processing.core import SignalProcessingError


class StatisticsFeatureError(SignalProcessingError):
    """Raised when statistics feature extraction fails or receives invalid input."""
