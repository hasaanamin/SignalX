"""Exceptions for signal segmentation."""

from signal_processing.core.exceptions import SignalProcessingError


class SegmentationError(SignalProcessingError):
    """Base exception for signal segmentation state machine errors."""


class InvalidTransitionError(SegmentationError):
    """An illegal state machine transition was requested."""
