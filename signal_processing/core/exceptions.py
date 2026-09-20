"""Exception hierarchy for signal_processing.core.

Hierarchy:
    SignalProcessingError
        ├── ContractValidationError
        └── StreamConsistencyError
"""


class SignalProcessingError(Exception):
    """Base exception for all signal processing errors.

    All custom exceptions in the signal_processing package inherit from this
    class to enable unified error handling across subsystems.
    """


class ContractValidationError(SignalProcessingError, ValueError):
    """Raised when a domain contract or data model fails validation constraints.

    Examples:
        - Non-finite numeric fields (NaN, Inf)
        - Negative counts or sequence indices
        - Incompatible array shapes or data types
        - Violated temporal / sample-index ordering
    """


class StreamConsistencyError(SignalProcessingError):
    """Raised when stream continuity or consistency checks between blocks fail.

    Examples:
        - Stream ID mismatch between consecutive blocks
        - Sample rate mismatch
        - Out-of-order or duplicate block sequence numbers
        - Unexplained sample gap or overlapping sample index range
    """
