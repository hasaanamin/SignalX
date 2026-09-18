from __future__ import annotations
from typing import Any
import numpy as np


def _as_signal_array(data: Any) -> np.ndarray:
    """
    Parses input data into a 1-D complex or real NumPy array.
    Supports complex arrays, canonical records, (I, Q) tuples, and real 1-D series.
    """
    if isinstance(data, dict):
        if data.get("complex_samples") is not None:
            x = np.asarray(data["complex_samples"])
        elif data.get("i") is not None and data.get("q") is not None:
            i = np.asarray(data["i"], dtype=np.float64)
            q = np.asarray(data["q"], dtype=np.float64)
            if i.shape != q.shape:
                raise ValueError("I/Q shape mismatch")
            x = i + 1j * q
        elif data.get("samples") is not None:
            x = np.asarray(data["samples"])
        else:
            raise ValueError("Record must contain 'complex_samples', 'samples', or both 'i' and 'q'")
    elif isinstance(data, tuple) and len(data) == 2 and not np.isscalar(data[0]):
        i = np.asarray(data[0], dtype=np.float64)
        q = np.asarray(data[1], dtype=np.float64)
        if i.shape != q.shape:
            raise ValueError("I/Q shape mismatch")
        x = i + 1j * q
    else:
        x = np.asarray(data)

    x = x.reshape(-1)
    if x.size == 0:
        raise ValueError("Input signal must contain at least one sample")
    if not np.all(np.isfinite(np.real(x))) or not np.all(np.isfinite(np.imag(x))):
        raise ValueError("Input signal contains NaN or infinity")
    return x
