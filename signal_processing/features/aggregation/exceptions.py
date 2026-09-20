"""Exceptions for feature aggregation and registry management."""

from signal_processing.core import SignalProcessingError


class FeatureAggregationError(SignalProcessingError):
    """Raised when feature aggregation, validation, or merge fails."""


class FeatureRegistryError(FeatureAggregationError):
    """Raised when feature extractor registry validation, registration, or lookup fails."""
