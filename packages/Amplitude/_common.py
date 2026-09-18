from __future__ import annotations
import numpy as np


def _as_iq_array(iq_data) -> np.ndarray:
    """Convert canonical record / complex array / (I,Q) into complex128 1-D IQ."""
    if isinstance(iq_data, dict):
        if iq_data.get("complex_samples") is not None:
            x = np.asarray(iq_data["complex_samples"])
        elif iq_data.get("i") is not None and iq_data.get("q") is not None:
            i = np.asarray(iq_data["i"])
            q = np.asarray(iq_data["q"])
            if i.shape != q.shape:
                raise ValueError("I/Q shape mismatch")
            x = i.astype(np.float64) + 1j * q.astype(np.float64)
        else:
            raise ValueError(
                "IQ record must contain 'complex_samples' or both 'i' and 'q'"
            )
    elif isinstance(iq_data, tuple) and len(iq_data) == 2:
        i = np.asarray(iq_data[0])
        q = np.asarray(iq_data[1])
        if i.shape != q.shape:
            raise ValueError("I/Q shape mismatch")
        x = i.astype(np.float64) + 1j * q.astype(np.float64)
    elif isinstance(iq_data, (list, tuple)) and len(iq_data) > 0 and isinstance(iq_data[0], (tuple, list)) and len(iq_data[0]) == 2:
        i = np.array([p[0] for p in iq_data], dtype=np.float64)
        q = np.array([p[1] for p in iq_data], dtype=np.float64)
        x = i + 1j * q
    else:
        x = np.asarray(iq_data)
        if not np.iscomplexobj(x):
            raise TypeError(
                "Expected complex IQ samples, a canonical IQ record, or (I, Q). "
                "Use parse_iq() for raw/interleaved data."
            )

    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if x.size == 0:
        raise ValueError("IQ data must contain at least one sample")
    if not np.all(np.isfinite(x.real)) or not np.all(np.isfinite(x.imag)):
        raise ValueError("IQ data contains NaN or infinity")
    return x
