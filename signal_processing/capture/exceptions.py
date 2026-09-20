"""Exceptions for event capture and assembly."""

from signal_processing.core.exceptions import SignalProcessingError


class CaptureError(SignalProcessingError):
    """Base exception for event capture and assembly failures."""


class CaptureCapacityError(CaptureError):
    """Capture buffer or ring buffer capacity is insufficient for requested policy."""


class CaptureDiscontinuityError(CaptureError):
    """A stream continuity loss interrupted active event capture."""
