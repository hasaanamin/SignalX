"""Exceptions for signal detection and noise estimation."""

from signal_processing.core.exceptions import SignalProcessingError


class DetectionError(SignalProcessingError):
    """Base exception for detection failures."""


class NoiseEstimationError(DetectionError):
    """Noise estimation failed or was evaluated under invalid conditions."""


class FilterConfigurationError(DetectionError):
    """Filter specification or frequency band configuration is invalid."""


class UncalibratedNoiseError(NoiseEstimationError):
    """Raised when energy detection is attempted with an invalid, NaN, or uncalibrated noise reference."""


class PolicyCompatibilityError(DetectionError):
    """Raised when a threshold policy is incompatible with detector configuration."""
