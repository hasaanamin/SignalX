"""Internal window resolver for spectral transforms."""

import numpy as np

from signal_processing.transforms.exceptions import TransformError

_SUPPORTED_WINDOWS = ("rectangular", "hann", "hamming", "blackman")


def get_window(name: str | None, length: int) -> tuple[np.ndarray, str, float]:
    """Resolve and generate a discrete window function of given length.

    Supported window types:
        - "rectangular" (None is treated as "rectangular")
        - "hann"
        - "hamming"
        - "blackman"

    Args:
        name: Name of the window function (case-insensitive) or None.
        length: Number of window coefficients (must be >= 1).

    Returns:
        Tuple of:
            - window: 1-D read-only float64 numpy array of window coefficients.
            - canonical_name: Normalized lowercase name of the window.
            - window_energy: Sum of squared window coefficients: sum(window^2).

    Raises:
        TransformError: If window name is unsupported or length < 1.
    """
    if length < 1:
        raise TransformError(f"Window length must be >= 1, got {length}")

    if name is None:
        norm_name = "rectangular"
    elif isinstance(name, str) and name.strip():
        norm_name = name.strip().lower()
    else:
        raise TransformError(f"Window name must be a string or None, got {name!r}")

    if norm_name not in _SUPPORTED_WINDOWS:
        raise TransformError(
            f"Unsupported window '{name}'. Supported windows: {', '.join(_SUPPORTED_WINDOWS)}"
        )

    if norm_name == "rectangular":
        win = np.ones(length, dtype=np.float64)
    elif norm_name == "hann":
        if length < 3:
            raise TransformError(
                f"Window 'hann' of length {length} produces zero or negligible energy; requires at least 3 samples"
            )
        win = np.hanning(length).astype(np.float64)
    elif norm_name == "hamming":
        win = np.hamming(length).astype(np.float64)
    elif norm_name == "blackman":
        if length < 3:
            raise TransformError(
                f"Window 'blackman' of length {length} produces zero or negligible energy; requires at least 3 samples"
            )
        win = np.blackman(length).astype(np.float64)
    else:
        raise TransformError(f"Unsupported window '{norm_name}'")

    energy = float(np.sum(win ** 2))
    if not np.isfinite(energy) or energy <= 0.0 or np.allclose(win, 0.0):
        raise TransformError(
            f"Window '{norm_name}' of length {length} produces zero or negligible energy ({energy}); "
            f"cannot be used for spectral transforms"
        )

    c_win = np.ascontiguousarray(win)
    frozen_win = np.frombuffer(c_win.tobytes(), dtype=c_win.dtype).reshape(c_win.shape)

    return frozen_win, norm_name, energy

