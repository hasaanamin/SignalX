from __future__ import annotations
from typing import Any, Optional, Tuple, Union
import numpy as np


def validate_sequence_inputs(
    values: Any,
    timestamps_sec: Any | None = None,
    sort_by_time: bool = False,
    min_samples: int = 2,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validates, masks missing values, and optionally sorts 1D sequence inputs.

    Parameters
    ----------
    values : array-like or list
        The sequence values to analyze.
    timestamps_sec : array-like or list, optional
        Timestamps corresponding to each value in values.
    sort_by_time : bool, default=False
        Whether to sort the output sequence in chronological timestamp order.
    min_samples : int, default=2
        Minimum number of valid samples required.
    allow_empty : bool, default=False
        If True, empty inputs return empty arrays without raising an error.

    Returns
    -------
    clean_values : np.ndarray
        1D float64 array of sequence values.
    clean_timestamps : np.ndarray
        1D float64 array of timestamps (or empty/synthetic if not provided).
    valid_mask : np.ndarray
        Boolean mask indicating finite, valid entries.
    original_indices : np.ndarray
        Integer array mapping current order back to original input indices.
    """
    if values is None:
        if allow_empty:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=bool), np.empty(0, dtype=int)
        raise ValueError("Input sequence values cannot be None.")

    try:
        val_arr = np.asarray(values, dtype=np.float64).ravel()
    except (ValueError, TypeError):
        val_arr = np.array(values, dtype=object).ravel()

    n = len(val_arr)
    if n == 0:
        if allow_empty:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=bool), np.empty(0, dtype=int)
        raise ValueError("Input sequence is empty.")

    orig_indices = np.arange(n, dtype=int)

    # Validate timestamps if supplied
    if timestamps_sec is not None:
        ts_arr = np.asarray(timestamps_sec, dtype=np.float64).ravel()
        if len(ts_arr) != n:
            raise ValueError(f"Length mismatch: len(values)={n} != len(timestamps_sec)={len(ts_arr)}.")
        
        if sort_by_time and n > 1:
            sort_order = np.argsort(ts_arr)
            val_arr = val_arr[sort_order]
            ts_arr = ts_arr[sort_order]
            orig_indices = orig_indices[sort_order]
    else:
        ts_arr = np.empty(0, dtype=np.float64)

    # Compute valid mask (finite values)
    if val_arr.dtype == object:
        valid_mask = np.array([v is not None and str(v).lower() not in ("none", "nan", "unknown", "") for v in val_arr], dtype=bool)
    else:
        valid_mask = np.isfinite(val_arr)

    usable_count = int(np.sum(valid_mask))
    if not allow_empty and usable_count < min_samples:
        raise ValueError(f"Insufficient usable observations: found {usable_count} valid samples, minimum required is {min_samples}.")

    return val_arr, ts_arr, valid_mask, orig_indices


def validate_matrix_input(
    matrix: Any,
    min_size: int = 2,
    min_dim: int | None = None,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validates a 2D square matrix (e.g. correlation matrix or similarity matrix).

    Parameters
    ----------
    matrix : array-like
        2D square matrix.
    min_size : int, default=2
        Minimum matrix dimension (N x N).
    min_dim : int, optional
        Alias for min_size.
    allow_empty : bool, default=False
        If True, empty inputs return empty arrays.

    Returns
    -------
    clean_mat : np.ndarray
        2D float64 matrix.
    valid_mask : np.ndarray
        Boolean mask of finite matrix elements.
    """
    eff_min = min_dim if min_dim is not None else min_size

    if matrix is None:
        if allow_empty:
            return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=bool)
        raise ValueError("Input matrix cannot be None.")

    mat_arr = np.asarray(matrix, dtype=np.float64)
    if mat_arr.ndim != 2 or mat_arr.shape[0] != mat_arr.shape[1]:
        raise ValueError(f"Expected a 2D square matrix, got shape {mat_arr.shape}.")

    n = mat_arr.shape[0]
    if n == 0:
        if allow_empty:
            return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=bool)
        raise ValueError("Input matrix is empty.")

    if not allow_empty and n < eff_min:
        raise ValueError(f"Matrix dimension ({n}x{n}) is less than minimum required ({eff_min}x{eff_min}).")

    valid_mask = np.isfinite(mat_arr)
    return mat_arr, valid_mask
