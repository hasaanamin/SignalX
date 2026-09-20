"""Exceptions for circular IQ buffering."""

from signal_processing.core.exceptions import SignalProcessingError


class BufferError(SignalProcessingError):
    """Base exception for circular buffer failures."""


class BufferRangeError(BufferError):
    """Requested sample range is invalid or out of bounds."""


class BufferOverwrittenError(BufferRangeError):
    """Requested samples have already been overwritten by newer data."""


class BufferUnavailableError(BufferRangeError):
    """Requested samples have not yet arrived in the buffer."""


class BufferContinuityError(BufferError):
    """Sequential write continuity was violated without an explicit reset/generation change."""
