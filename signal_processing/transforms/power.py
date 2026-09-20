"""Instantaneous and mean power calculation DSP primitives."""

from typing import Any

import numpy as np

from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.validation import validate_iq


def compute_power(iq: np.ndarray) -> np.ndarray:
    """Compute instantaneous linear power of complex IQ samples.

    Mathematical definition:
        power[n] = |iq[n]|^2 = I[n]^2 + Q[n]^2

    Args:
        iq: One-dimensional complex IQ array.

    Returns:
        One-dimensional real-valued, non-negative, finite numpy array representing
        instantaneous linear power. The array is read-only.

    Raises:
        TransformError: If IQ input is invalid, non-complex, empty, or non-finite.
    """
    val_iq = validate_iq(iq, min_samples=1)
    # Promote real and imag components to float64 to prevent float32 squaring overflow (e.g. for |1e20|^2)
    i_f64 = val_iq.real.astype(np.float64, copy=False)
    q_f64 = val_iq.imag.astype(np.float64, copy=False)
    p = i_f64 ** 2 + q_f64 ** 2
    if not np.all(np.isfinite(p)):
        raise TransformError("Power calculation produced non-finite values (overflow)")
    c_p = np.ascontiguousarray(p)
    return np.frombuffer(c_p.tobytes(), dtype=c_p.dtype).reshape(c_p.shape)


def compute_mean_power_linear(iq: np.ndarray) -> float:
    """Compute average (mean) linear power of complex IQ samples.

    Mathematical definition:
        mean_power = (1 / N) * sum(|iq[n]|^2)

    Args:
        iq: One-dimensional complex IQ array.

    Returns:
        Scalar finite float representing average linear power.

    Raises:
        TransformError: If IQ input is invalid, non-complex, empty, or non-finite.
    """
    p = compute_power(iq)
    return float(np.mean(p))


def linear_power_to_dbfs(
    power: Any,
    *,
    floor_dbfs: float = -300.0,
) -> Any:
    """Convert non-negative linear power to decibels relative to full scale (dBFS).

    Mathematical definition:
        P_dBFS = 10 * log10(max(P_linear, 10^(floor_dbfs / 10)))

    Reference:
        0 dBFS power reference = linear power 1.0.

    Args:
        power: Scalar float/int or numeric numpy array of non-negative linear power.
        floor_dbfs: Minimum dBFS floor to prevent log10(0) resulting in -inf
            (default: -300.0 dBFS).

    Returns:
        Converted power in dBFS. Returns float for scalar input, or a read-only
        numpy array for array input.

    Raises:
        TransformError: If power contains negative, non-numeric, complex, or non-finite values.
    """
    if isinstance(floor_dbfs, bool) or not isinstance(floor_dbfs, (int, float, np.floating)):
        raise TransformError(f"floor_dbfs must be a numeric float, got {floor_dbfs!r}")
    floor_val = float(floor_dbfs)
    if not np.isfinite(floor_val):
        raise TransformError(f"floor_dbfs must be finite, got {floor_dbfs}")

    # Scalar branch
    if isinstance(power, (int, float, np.floating, np.integer)) and not isinstance(power, bool):
        val = float(power)
        if not np.isfinite(val):
            raise TransformError(f"Linear power must be finite, got {power}")
        if val < 0.0:
            raise TransformError(f"Linear power must be non-negative, got {power}")
        if val == 0.0:
            return floor_val
        with np.errstate(divide="ignore", invalid="ignore"):
            db_val = float(10.0 * np.log10(val))
        return max(db_val, floor_val)

    # Array branch
    try:
        arr = np.asarray(power)
    except Exception as exc:
        raise TransformError(f"Cannot convert power input to array: {exc}") from exc

    if not (
        np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
    ) or np.iscomplexobj(arr) or arr.dtype == object or arr.dtype.kind in ("U", "S"):
        raise TransformError(
            f"Linear power array must be a real-valued numeric type, got dtype {arr.dtype}"
        )

    try:
        if not np.all(np.isfinite(arr)):
            raise TransformError("Linear power contains non-finite values (NaN or Inf)")
    except (TypeError, ValueError) as exc:
        raise TransformError(f"Finiteness check failed on linear power: {exc}") from exc

    if np.any(arr < 0.0):
        raise TransformError("Linear power array contains negative values")

    # Vectorized safe log10 with floor to avoid underflow on extreme negative floors
    with np.errstate(divide="ignore", invalid="ignore"):
        dbfs = 10.0 * np.log10(arr)
        dbfs = np.maximum(dbfs, floor_val)
        dbfs = np.where(np.isnan(dbfs), floor_val, dbfs)

    c_out = np.ascontiguousarray(dbfs)
    return np.frombuffer(c_out.tobytes(), dtype=c_out.dtype).reshape(c_out.shape)



def compute_mean_power_dbfs(
    iq: np.ndarray | None = None,
    *,
    floor_dbfs: float = -300.0,
    mean_power_linear: float | None = None,
) -> float:
    """Compute average (mean) power of complex IQ samples in dBFS.

    Mathematical definition:
        mean_power_dBFS = 10 * log10(mean(|iq|^2))

    Args:
        iq: One-dimensional complex IQ array (optional if mean_power_linear is provided).
        floor_dbfs: Minimum dBFS floor for zero or extremely low power (default: -300.0 dBFS).
        mean_power_linear: Precomputed mean linear power to avoid redundant computation.

    Returns:
        Scalar finite float representing average power in dBFS.

    Raises:
        TransformError: If IQ input is invalid, non-complex, empty, or non-finite.
    """
    if mean_power_linear is not None:
        mean_p = float(mean_power_linear)
    else:
        if iq is None:
            raise TransformError("Either iq or mean_power_linear must be provided")
        mean_p = compute_mean_power_linear(iq)
    return float(linear_power_to_dbfs(mean_p, floor_dbfs=floor_dbfs))
