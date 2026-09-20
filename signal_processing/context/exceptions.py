"""Exceptions for signal_processing.context."""

from signal_processing.core import SignalProcessingError


class ContextError(SignalProcessingError):
    """Raised when context configuration or operational invariants are violated.

    Examples:
        - Invalid event provided to SignalContext (e.g. non-SignalEvent)
        - Invalid AnalysisConfig supplied
        - Malformed configuration parameters
    """
