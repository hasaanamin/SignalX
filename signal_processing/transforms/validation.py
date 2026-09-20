"""Validation helpers for transform operations."""

from typing import Any

import numpy as np

from signal_processing.transforms.exceptions import TransformError


def validate_iq(iq: Any, *, min_samples: int = 1) -> np.ndarray:
    """Validate that input data is a valid 1-D complex IQ numpy array.

    Requirements:
        - Safely convertible to a numpy array
        - Exactly one-dimensional
        - Non-empty with at least `min_samples` elements
        - Complex numeric type (e.g. np.complex64 or np.complex128)
        - All real and imaginary components must be finite (reject NaN, Inf)
        - Reject real-only, string, or object arrays
        - Does not mutate caller data

    Args:
        iq: The input IQ array or convertible sequence.
        min_samples: Minimum number of samples required (default: 1).

    Returns:
        Validated 1-D complex numpy array.

    Raises:
        TransformError: If any validation constraint is violated.
    """
    if min_samples < 1:
        raise TransformError(f"min_samples must be >= 1, got {min_samples}")

    try:
        arr = np.asarray(iq)
    except Exception as exc:
        raise TransformError(f"IQ input cannot be converted to a numpy array: {exc}") from exc

    # Dimension check
    if arr.ndim != 1:
        raise TransformError(
            f"IQ input must be exactly 1-dimensional, got {arr.ndim} dimensions (shape {arr.shape})"
        )

    # Length check
    if arr.size < min_samples:
        raise TransformError(
            f"IQ input must contain at least {min_samples} samples, got {arr.size}"
        )

    # Complex numeric check: reject real-only, string, object arrays
    if not (
        np.issubdtype(arr.dtype, np.complexfloating) or np.iscomplexobj(arr)
    ) or arr.dtype == object or arr.dtype.kind in ("U", "S"):
        raise TransformError(
            f"IQ input must be a complex numeric array (complex64 or complex128), got dtype {arr.dtype}"
        )

    # Finiteness check
    try:
        if not np.all(np.isfinite(arr)):
            raise TransformError("IQ input contains non-finite values (NaN or Inf)")
    except (TypeError, ValueError) as exc:
        raise TransformError(f"Failed finiteness validation on IQ input: {exc}") from exc

    return arr
