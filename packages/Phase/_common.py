from __future__ import annotations
import numpy as np


def _as_iq_array(data) -> np.ndarray:
    """
    Converts any input (canonical dict, numpy complex array, (I, Q) tuple, or list of pairs)
    into a clean 1-D complex128 NumPy array.
    """
    if isinstance(data, dict):
        if data.get("complex_samples") is not None:
            arr = np.asarray(data["complex_samples"], dtype=np.complex128)
        elif data.get("i") is not None and data.get("q") is not None:
            i = np.asarray(data["i"], dtype=np.float64)
            q = np.asarray(data["q"], dtype=np.float64)
            if i.shape != q.shape:
                raise ValueError("I and Q shape mismatch")
            arr = (i + 1j * q).astype(np.complex128)
        elif "samples" in data:
            arr = np.asarray(data["samples"], dtype=np.complex128)
        else:
            raise ValueError("Dictionary must contain 'complex_samples' or ('i', 'q')")
    elif isinstance(data, tuple) and len(data) == 2 and not np.isscalar(data[0]):
        i = np.asarray(data[0], dtype=np.float64)
        q = np.asarray(data[1], dtype=np.float64)
        if i.shape != q.shape:
            raise ValueError("I and Q shape mismatch")
        arr = (i + 1j * q).astype(np.complex128)
    elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
        i = np.array([p[0] for p in data], dtype=np.float64)
        q = np.array([p[1] for p in data], dtype=np.float64)
        arr = (i + 1j * q).astype(np.complex128)
    else:
        raw = np.asarray(data)
        if np.iscomplexobj(raw):
            arr = raw.astype(np.complex128)
        else:
            arr = raw.astype(np.float64).astype(np.complex128)

    arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError("IQ data must contain at least one sample")
    if not np.all(np.isfinite(np.real(arr))) or not np.all(np.isfinite(np.imag(arr))):
        raise ValueError("IQ data contains NaN or infinity")

    return arr


def _extract_sample_rate(iq_data, sample_rate: float | None = None, default_fs: float = 1.0) -> float:
    """
    Extracts or validates the sampling rate (fs) in Hz from input metadata or argument.
    """
    if sample_rate is not None:
        fs = float(sample_rate)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("sample_rate must be a positive finite number")
        return fs

    if isinstance(iq_data, dict):
        if iq_data.get("sample_rate") is not None:
            return float(iq_data["sample_rate"])
        if iq_data.get("sampling_rate") is not None:
            return float(iq_data["sampling_rate"])
        if iq_data.get("fs") is not None:
            return float(iq_data["fs"])

    return float(default_fs)


def _compute_phase_arrays(z: np.ndarray, unwrap: bool = False, deg: bool = False) -> np.ndarray:
    """
    Extracts raw or unwrapped instantaneous phase array.
    """
    phase_rad = np.angle(z)
    if unwrap:
        phase_rad = np.unwrap(phase_rad)
    if deg:
        return np.degrees(phase_rad)
    return phase_rad
