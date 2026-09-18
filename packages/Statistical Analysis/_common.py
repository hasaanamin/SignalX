from __future__ import annotations
import numpy as np


def _as_array(data) -> np.ndarray:
    """
    Converts any input (canonical dict, complex array, real array, (I, Q) tuple, or list)
    into a clean 1-D NumPy array.
    """
    if isinstance(data, dict):
        if data.get("complex_samples") is not None:
            arr = np.asarray(data["complex_samples"])
        elif data.get("i") is not None and data.get("q") is not None:
            i = np.asarray(data["i"], dtype=np.float64)
            q = np.asarray(data["q"], dtype=np.float64)
            if i.shape != q.shape:
                raise ValueError("I and Q shape mismatch")
            arr = i + 1j * q
        elif "samples" in data:
            arr = np.asarray(data["samples"])
        else:
            raise ValueError("Dictionary must contain 'complex_samples', ('i', 'q'), or 'samples'")
    elif isinstance(data, tuple) and len(data) == 2 and not np.isscalar(data[0]):
        i = np.asarray(data[0], dtype=np.float64)
        q = np.asarray(data[1], dtype=np.float64)
        if i.shape != q.shape:
            raise ValueError("I and Q shape mismatch")
        arr = i + 1j * q
    elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
        i = np.array([p[0] for p in data], dtype=np.float64)
        q = np.array([p[1] for p in data], dtype=np.float64)
        arr = i + 1j * q
    else:
        arr = np.asarray(data)

    arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError("Data must contain at least one sample")
    if not np.all(np.isfinite(np.real(arr))) or (np.iscomplexobj(arr) and not np.all(np.isfinite(np.imag(arr)))):
        raise ValueError("Data contains NaN or infinity")

    return arr


def _as_real_array(data, mode="auto") -> np.ndarray:
    """
    Ensures input is 1-D real array. If complex, extracts according to mode:
    mode='magnitude' -> |z|
    mode='power'     -> |z|^2
    mode='real'      -> real(z)
    mode='imag'      -> imag(z)
    mode='phase'     -> angle(z)
    """
    arr = _as_array(data)
    if np.iscomplexobj(arr):
        m = str(mode).lower()
        if m in ("magnitude", "auto", "mag", "abs"):
            return np.abs(arr)
        elif m == "power":
            return np.abs(arr) ** 2
        elif m in ("real", "i"):
            return np.real(arr)
        elif m in ("imag", "q"):
            return np.imag(arr)
        elif m == "phase":
            return np.angle(arr)
        else:
            raise ValueError(f"Unknown mode for complex conversion: {mode}")
    return arr.astype(np.float64)
