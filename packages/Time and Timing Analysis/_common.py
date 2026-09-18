from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

from typing import Any, Sequence
import numpy as np

from .models import UINT64_MIN, UINT64_MAX


def validate_uint64_timestamp(ts: int | None) -> int | None:
    """Validates that a timestamp integer falls within valid unsigned 64-bit boundaries."""
    if ts is None:
        return None
    if isinstance(ts, (bool, np.bool_)):
        raise TypeError(f"hardware_timestamp_ns cannot be a boolean (got {ts})")
    if isinstance(ts, (float, np.floating)):
        raise TypeError(f"hardware_timestamp_ns must be an integer, got float {ts}")
    if isinstance(ts, str):
        raise TypeError(f"hardware_timestamp_ns must be an integer, got str '{ts}'")
    if not isinstance(ts, (int, np.integer)):
        raise TypeError(f"hardware_timestamp_ns must be an integer, got {type(ts).__name__}")
    val = int(ts)
    if val < UINT64_MIN or val > UINT64_MAX:
        raise ValueError(
            f"hardware_timestamp_ns must be a valid unsigned 64-bit integer ({UINT64_MIN}..{UINT64_MAX}), got {ts}"
        )
    return val


def validate_sample_rate(sample_rate_hz: float | None) -> float:
    """Validates that sample rate is a positive, finite float."""
    if sample_rate_hz is None:
        raise ValueError("sample_rate_hz is required and cannot be None")
    if isinstance(sample_rate_hz, bool):
        raise TypeError(f"sample_rate_hz cannot be a boolean (got {sample_rate_hz})")
    fs = float(sample_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"sample_rate_hz must be a positive finite number, got {sample_rate_hz}")
    return fs


def validate_strictly_positive_intervals(arr: np.ndarray | Sequence[float], name: str = "intervals") -> None:
    """Validates that all finite interval values in an array are strictly positive (> 0)."""
    v = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite_v = v[np.isfinite(v)]
    if len(finite_v) > 0 and np.any(finite_v <= 0):
        bad_val = float(finite_v[finite_v <= 0][0])
        raise ValueError(f"{name} must contain strictly positive values (> 0), found {bad_val}")


def validate_nonnegative_durations(arr: np.ndarray | Sequence[float], name: str = "durations") -> None:
    """Validates that all finite duration values in an array are non-negative (>= 0) and not infinite."""
    v = np.asarray(arr, dtype=np.float64).reshape(-1)
    if len(v) > 0 and np.any(np.isinf(v)):
        raise ValueError(f"{name} cannot contain infinite values (+Inf / -Inf)")
    finite_v = v[np.isfinite(v)]
    if len(finite_v) > 0 and np.any(finite_v < 0):
        bad_val = float(finite_v[finite_v < 0][0])
        raise ValueError(f"{name} must contain non-negative values (>= 0), found {bad_val}")


def ensure_1d_float_array(values: Any, name: str = "values") -> np.ndarray:
    """Converts input sequence to a 1D float64 numpy array."""
    if values is None:
        return np.array([], dtype=np.float64)
    if isinstance(values, (list, tuple)):
        for x in values:
            if isinstance(x, (bool, np.bool_)):
                raise TypeError(f"Elements of {name} cannot be boolean values (got {x})")
    elif isinstance(values, np.ndarray):
        if values.dtype == bool:
            raise TypeError(f"{name} cannot be boolean numpy array")
        for x in values.reshape(-1):
            if isinstance(x, (bool, np.bool_)):
                raise TypeError(f"Elements of {name} cannot be boolean values (got {x})")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr


def ensure_1d_int_array(values: Any, name: str = "values") -> list[int]:
    """Converts input sequence to a list of Python integers preserving full uint64 precision."""
    if values is None:
        return []
    if isinstance(values, str):
        raise TypeError(f"{name} cannot be a string value (got '{values}')")
    if isinstance(values, (list, tuple)):
        out: list[int] = []
        for x in values:
            if isinstance(x, (bool, np.bool_)):
                raise TypeError(f"Elements of {name} cannot be boolean values (got {x})")
            if isinstance(x, str):
                raise TypeError(f"Elements of {name} cannot be strings (got '{x}')")
            if isinstance(x, (float, np.floating)):
                raise TypeError(f"Elements of {name} cannot be floating-point numbers (got {x})")
            if not isinstance(x, (int, np.integer)):
                raise TypeError(f"Elements of {name} must be integer-like, got {type(x).__name__}")
            out.append(int(x))
        return out
    if isinstance(values, np.ndarray):
        if values.dtype == bool:
            raise TypeError(f"{name} cannot be boolean numpy array")
        if np.issubdtype(values.dtype, np.floating):
            raise TypeError(f"{name} cannot be floating-point numpy array")
        out_np: list[int] = []
        for x in values.reshape(-1):
            if isinstance(x, (bool, np.bool_)):
                raise TypeError(f"Elements of {name} cannot be boolean values (got {x})")
            if isinstance(x, str):
                raise TypeError(f"Elements of {name} cannot be strings (got '{x}')")
            if isinstance(x, (float, np.floating)):
                raise TypeError(f"Elements of {name} cannot be floating-point numbers (got {x})")
            if not isinstance(x, (int, np.integer)):
                raise TypeError(f"Elements of {name} must be integer-like, got {type(x).__name__}")
            out_np.append(int(x))
        return out_np
    if isinstance(values, (bool, np.bool_)):
        raise TypeError(f"{name} cannot be a boolean value")
    if isinstance(values, (float, np.floating)):
        raise TypeError(f"{name} cannot be a floating-point number (got {values})")
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    raise TypeError(f"{name} must be a sequence or integer, got {type(values).__name__}")


def validate_processing_delay_sec(delay: Any) -> float:
    """Validates processing delay parameter."""
    if delay is None:
        return 0.0
    if isinstance(delay, (bool, np.bool_)):
        raise TypeError(f"processing_delay_sec cannot be a boolean (got {delay})")
    if not isinstance(delay, (int, float, np.number)):
        raise TypeError(f"processing_delay_sec must be numeric, got {type(delay).__name__}")
    val = float(delay)
    if not np.isfinite(val):
        raise ValueError(f"processing_delay_sec must be a finite number, got {val}")
    if val < 0.0:
        raise ValueError(f"processing_delay_sec must be non-negative (>= 0), got {val}")
    return val


def calculate_hardware_toas_ns(
    hw_timestamp_ns: int,
    indices: np.ndarray | Sequence[int],
    sample_rate_hz: float,
    processing_delay_sec: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """
    Computes absolute TOAs with genuine integer nanosecond precision using Python integers
    to avoid overflow on large epoch uint64 timestamps.

    Parameters
    ----------
    hw_timestamp_ns : int
        Sample-0 hardware capture timestamp in nanoseconds (0..2^64-1).
    indices : np.ndarray | Sequence[int]
        Sample indices of pulse start / arrival events.
    sample_rate_hz : float
        Sampling frequency in Hz.
    processing_delay_sec : float
        Known group delay or processing delay to subtract in seconds.

    Returns
    -------
    tuple[np.ndarray, list[int]]
        (toas_sec, toas_ns) where toas_sec is float64 seconds and toas_ns is a list of Python ints.
    """
    fs = validate_sample_rate(sample_rate_hz)
    hw_ns = validate_uint64_timestamp(hw_timestamp_ns)
    if hw_ns is None:
        raise ValueError("hw_timestamp_ns must not be None for hardware absolute TOA calculation")

    idx_arr = np.asarray(indices, dtype=np.float64).reshape(-1)
    delay_ns = int(round(float(processing_delay_sec) * 1e9))

    toas_ns: list[int] = []
    for idx in idx_arr:
        delta_ns = int(round((float(idx) / fs) * 1e9)) - delay_ns
        toas_ns.append(hw_ns + delta_ns)

    # Convert to seconds representation for downstream floating algorithms
    toas_sec = np.array([t / 1e9 for t in toas_ns], dtype=np.float64)
    return toas_sec, toas_ns


def calculate_relative_toas_sec(
    indices: np.ndarray | Sequence[int],
    sample_rate_hz: float,
    processing_delay_sec: float = 0.0,
) -> np.ndarray:
    """
    Computes relative TOAs in seconds from sample 0 of the capture.
    """
    fs = validate_sample_rate(sample_rate_hz)
    idx = np.asarray(indices, dtype=np.float64).reshape(-1)
    toas_sec = (idx / fs) - float(processing_delay_sec)
    return toas_sec


def compute_gap_aware_diff(toas: np.ndarray | Sequence[float]) -> np.ndarray:
    """
    Computes consecutive interval differences (toas[i+1] - toas[i]),
    preserving gaps so that missing/NaN TOAs break adjacency rather than bridging.

    For an input array of length N, returns an array of length N-1.
    If either toas[i] or toas[i+1] is NaN/non-finite, diff[i] is NaN.
    """
    arr = ensure_1d_float_array(toas, name="toas")
    n = len(arr)
    if n <= 1:
        return np.array([], dtype=np.float64)

    t0 = arr[:-1]
    t1 = arr[1:]

    valid_mask = np.isfinite(t0) & np.isfinite(t1)
    diffs = np.full(n - 1, np.nan, dtype=np.float64)
    diffs[valid_mask] = t1[valid_mask] - t0[valid_mask]
    return diffs


def compute_gap_aware_diff_ns(toas_ns: Sequence[int] | np.ndarray) -> np.ndarray:
    """
    Computes consecutive interval differences in seconds directly from integer nanosecond TOAs
    by differencing in integer nanoseconds first (toa_ns[i] - toa_ns[i-1]),
    then dividing by 1e9.
    This preserves sub-microsecond and nanosecond precision even at large epoch timestamps (~1.7e18 ns).
    """
    arr_ns = ensure_1d_int_array(toas_ns, name="toas_ns")
    n = len(arr_ns)
    if n <= 1:
        return np.array([], dtype=np.float64)
    diffs = np.array([(arr_ns[i] - arr_ns[i - 1]) / 1e9 for i in range(1, n)], dtype=np.float64)
    return diffs


def robust_nan_statistics(arr: np.ndarray | Sequence[float]) -> dict[str, Any]:
    """
    Calculates robust summary statistics on an array, filtering out non-finite values.
    """
    v = ensure_1d_float_array(arr)
    valid = v[np.isfinite(v)]
    valid_count = len(valid)

    if valid_count == 0:
        return {
            "count": len(v),
            "valid_count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "rms": None,
            "peak_to_peak": None,
        }

    mean_val = float(np.mean(valid))
    med_val = float(np.median(valid))
    min_val = float(np.min(valid))
    max_val = float(np.max(valid))
    std_val = float(np.std(valid, ddof=1)) if valid_count > 1 else 0.0
    rms_val = float(np.sqrt(np.mean(valid ** 2)))
    p2p_val = float(max_val - min_val)

    return {
        "count": len(v),
        "valid_count": valid_count,
        "mean": mean_val,
        "median": med_val,
        "min": min_val,
        "max": max_val,
        "std": std_val,
        "rms": rms_val,
        "peak_to_peak": p2p_val,
    }


__all__ = [
    "validate_uint64_timestamp",
    "validate_sample_rate",
    "validate_strictly_positive_intervals",
    "validate_nonnegative_durations",
    "validate_processing_delay_sec",
    "ensure_1d_float_array",
    "ensure_1d_int_array",
    "calculate_hardware_toas_ns",
    "calculate_relative_toas_sec",
    "compute_gap_aware_diff",
    "compute_gap_aware_diff_ns",
    "robust_nan_statistics",
]


def main() -> None:
    print("=" * 60)
    print("  Time and Timing Analysis: _common Utilities")
    print("=" * 60)
    hw_ts = 1718000000000000000
    indices = [0, 1000, 2000]
    toas_sec, toas_ns = calculate_hardware_toas_ns(hw_ts, indices, 1e6)
    print(f"TOAs (ns): {toas_ns}")
    print(f"TOAs (s): {toas_sec}")
    print("=" * 60)


if __name__ == "__main__":
    main()
