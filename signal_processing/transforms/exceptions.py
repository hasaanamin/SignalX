"""Exceptions for signal_processing.transforms."""

from signal_processing.core import SignalProcessingError


class TransformError(SignalProcessingError):
    """Raised when a DSP transform operation fails or receives invalid input.

    Examples:
        - Malformed IQ input (non-complex, non-1D, empty, non-finite)
        - Unsupported window functions
        - Invalid FFT lengths (e.g. nfft < input_samples)
        - Insufficient samples (e.g. instantaneous frequency with < 2 samples)
        - Invalid sample rate (<= 0 or non-finite)
        - Impossible normalizations (e.g. normalizing zero-energy autocorrelation)
        - Negative linear power inputs to log-converters
    """
