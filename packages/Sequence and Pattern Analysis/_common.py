from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple, Union
import numpy as np


def wrap_circular_difference(diff_rad: np.ndarray | float) -> np.ndarray | float:
    """
    Wraps phase difference into standard [-pi, pi) circular interval:
        delta_phi = ((diff + pi) % (2*pi)) - pi
    """
    if isinstance(diff_rad, (int, float, np.floating)):
        val = (float(diff_rad) + math.pi) % (2.0 * math.pi) - math.pi
        return val
    arr = np.asarray(diff_rad, dtype=np.float64)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def estimate_discrete_states(
    values: np.ndarray,
    tolerance: float | None = None,
    max_states: int = 16,
    relative_tol: bool = False,
) -> tuple[list[float], np.ndarray]:
    """
    Quantizes a 1D continuous sequence into distinct discrete state levels.

    Parameters
    ----------
    values : np.ndarray
        1D array of valid numerical values.
    tolerance : float, optional
        Absolute or relative clustering threshold. If None, automatically derived from variance.
    max_states : int, default=16
        Maximum allowable discrete states.
    relative_tol : bool, default=False
        Whether tolerance is relative to state magnitude.

    Returns
    -------
    state_levels : list[float]
        Sorted list of discovered discrete state center values.
    state_sequence : np.ndarray
        Integer array of state assignments (0 .. K-1) for each element in values.
    """
    n = len(values)
    if n == 0:
        return [], np.empty(0, dtype=int)
    if n == 1:
        return [float(values[0])], np.zeros(1, dtype=int)

    valid_vals = values[np.isfinite(values)]
    if len(valid_vals) == 0:
        return [], np.full(n, -1, dtype=int)

    val_min = float(np.min(valid_vals))
    val_max = float(np.max(valid_vals))
    val_range = val_max - val_min
    val_std = float(np.std(valid_vals))

    if val_range < 1e-9 or val_std < 1e-9:
        mean_v = float(np.mean(valid_vals))
        state_seq = np.where(np.isfinite(values), 0, -1)
        return [mean_v], state_seq

    if tolerance is None:
        if relative_tol:
            tol = 0.02
        else:
            tol = max(val_range * 0.05, val_std * 0.15, 1e-6)
    else:
        tol = float(tolerance)

    # 1D single-linkage agglomerative merge on sorted unique values
    sorted_idx = np.argsort(valid_vals)
    sorted_vals = valid_vals[sorted_idx]

    clusters: list[list[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        curr_center = np.mean(clusters[-1])
        diff = abs(v - curr_center)
        thresh = (max(abs(curr_center) * tol, 1e-12)) if relative_tol else tol
        if diff <= thresh:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    if len(clusters) > max_states:
        # Merge closest adjacent clusters until <= max_states
        while len(clusters) > max_states:
            centers = [np.mean(c) for c in clusters]
            diffs = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
            min_idx = int(np.argmin(diffs))
            clusters[min_idx].extend(clusters[min_idx + 1])
            clusters.pop(min_idx + 1)

    state_levels = sorted([float(np.mean(c)) for c in clusters])

    # Assign each original value to nearest state level
    state_levels_arr = np.array(state_levels)
    state_seq = np.full(n, -1, dtype=int)
    for i in range(n):
        if np.isfinite(values[i]):
            state_seq[i] = int(np.argmin(np.abs(values[i] - state_levels_arr)))

    return state_levels, state_seq


def build_transition_matrix(
    state_sequence: np.ndarray,
    num_states: int,
    missing_value: int = -1,
) -> np.ndarray:
    """
    Builds a normalized K x K Markov state transition probability matrix.
    Gaps (missing_value) break adjacency so non-adjacent observations are not falsely linked.

    Parameters
    ----------
    state_sequence : np.ndarray
        Array of integer state assignments (0 .. num_states-1, with missing_value for invalid).
    num_states : int
        Total number of discrete states K.
    missing_value : int, default=-1
        Sentinel value representing missing or invalid observations.

    Returns
    -------
    transition_matrix : np.ndarray
        K x K transition probability matrix where row i sums to 1.0 (or 0 if unvisited).
    """
    if num_states <= 0:
        return np.empty((0, 0), dtype=np.float64)

    counts = np.zeros((num_states, num_states), dtype=np.float64)
    for i in range(len(state_sequence) - 1):
        s_from = state_sequence[i]
        s_to = state_sequence[i + 1]
        if (
            s_from != missing_value
            and s_to != missing_value
            and s_from >= 0
            and s_to >= 0
            and s_from < num_states
            and s_to < num_states
        ):
            counts[s_from, s_to] += 1.0

    row_sums = np.sum(counts, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_mat = np.where(row_sums > 0, counts / row_sums, 0.0)

    return trans_mat


def compute_dwell_lengths(
    state_sequence: np.ndarray,
    missing_value: int = -1,
) -> list[int]:
    """
    Computes consecutive dwell run-lengths for each active state segment.
    Missing/invalid observations (missing_value) break contiguous run-lengths.

    Parameters
    ----------
    state_sequence : np.ndarray
        Array of integer state assignments.
    missing_value : int, default=-1
        Sentinel value representing missing or invalid observations.

    Returns
    -------
    dwell_lengths : list[int]
        List of integer run-lengths for each contiguous state block.
    """
    dwells: list[int] = []
    curr_state = None
    curr_len = 0
    for s in state_sequence:
        if s != missing_value and s >= 0:
            if s == curr_state:
                curr_len += 1
            else:
                if curr_len > 0:
                    dwells.append(curr_len)
                curr_state = s
                curr_len = 1
        else:
            if curr_len > 0:
                dwells.append(curr_len)
            curr_state = None
            curr_len = 0
    if curr_len > 0:
        dwells.append(curr_len)
    return dwells


def valid_adjacent_pairs(
    values: np.ndarray | list[Any],
    valid_mask: np.ndarray | None = None,
    missing_value: Any = -1,
) -> list[tuple[Any, Any, int, int]]:
    """
    Extracts all strictly adjacent pairs (i, i+1) where both elements are valid.
    Missing observations explicitly break adjacency and never form cross-gap pairs.

    Parameters
    ----------
    values : array-like
        Input sequence.
    valid_mask : np.ndarray, optional
        Boolean mask of valid elements. If None, derived from np.isfinite and missing_value.
    missing_value : Any, default=-1
        Sentinel value representing missing discrete states.

    Returns
    -------
    pairs : list[tuple[Any, Any, int, int]]
        List of (val_i, val_{i+1}, index_i, index_{i+1}) for valid adjacent pairs.
    """
    n = len(values)
    if n < 2:
        return []

    if valid_mask is None:
        mask = np.ones(n, dtype=bool)
        for idx, v in enumerate(values):
            if v is None:
                mask[idx] = False
            elif isinstance(v, (int, float, np.floating, np.integer)):
                if not np.isfinite(v) or (missing_value is not None and v == missing_value):
                    mask[idx] = False
            elif isinstance(v, str):
                if v.strip().lower() in ("none", "nan", "unknown", "unresolved", "", str(missing_value)):
                    mask[idx] = False
    else:
        mask = np.asarray(valid_mask, dtype=bool)

    pairs: list[tuple[Any, Any, int, int]] = []
    for i in range(n - 1):
        if mask[i] and mask[i + 1]:
            pairs.append((values[i], values[i + 1], i, i + 1))
    return pairs


def adjacent_difference_with_gaps(
    values: np.ndarray,
    valid_mask: np.ndarray | None = None,
    is_circular: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes consecutive differences strictly across adjacent valid pairs (i, i+1).
    Missing values (NaN, None, inf) break adjacency so no difference is taken across a gap.

    Parameters
    ----------
    values : np.ndarray
        1D array of numerical observations.
    valid_mask : np.ndarray, optional
        Boolean validity mask. If None, computed via np.isfinite(values).
    is_circular : bool, default=False
        Whether to wrap differences into [-pi, pi).

    Returns
    -------
    deltas : np.ndarray
        1D array of valid adjacent differences.
    target_indices : np.ndarray
        1D integer array of step target indices (i+1) corresponding to each delta.
    signed_deltas : np.ndarray
        Signed difference values (same as deltas).
    """
    val_arr = np.asarray(values, dtype=np.float64)
    n = len(val_arr)
    if n < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=int), np.empty(0, dtype=np.float64)

    if valid_mask is None:
        mask = np.isfinite(val_arr)
    else:
        mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(val_arr)

    deltas_list: list[float] = []
    target_idx_list: list[int] = []

    for i in range(n - 1):
        if mask[i] and mask[i + 1]:
            d = float(val_arr[i + 1] - val_arr[i])
            if is_circular:
                d = float(wrap_circular_difference(d))
            deltas_list.append(d)
            target_idx_list.append(i + 1)

    deltas = np.array(deltas_list, dtype=np.float64)
    target_indices = np.array(target_idx_list, dtype=int)
    return deltas, target_indices, deltas


def _is_missing_element(val: Any, missing_value: Any = -1) -> bool:
    """Checks whether a sequence item represents a missing/invalid observation."""
    if val is None:
        return True
    if isinstance(val, (int, float, np.floating, np.integer)):
        if not np.isfinite(val):
            return True
        if missing_value is not None and val == missing_value:
            return True
    elif isinstance(val, str):
        if val.strip().lower() in ("none", "nan", "unknown", "unresolved", "", str(missing_value)):
            return True
    return False


def _are_elements_equal(v1: Any, v2: Any, tolerance: float = 0.0, missing_value: Any = -1) -> bool:
    """Shared comparator for numeric tolerance or symbolic equality with missing data rejection."""
    if _is_missing_element(v1, missing_value) or _is_missing_element(v2, missing_value):
        return False
    if isinstance(v1, (int, float, np.floating, np.integer)) and isinstance(v2, (int, float, np.floating, np.integer)):
        return bool(abs(float(v1) - float(v2)) <= max(tolerance, 1e-9))
    return bool(v1 == v2)


def detect_symbolic_periodicity(
    sequence: Any,
    max_period: int | None = None,
    tolerance: float = 0.0,
    missing_value: Any = -1,
) -> tuple[bool, Optional[int], list[Any], float, list[int]]:
    """
    Detects repeating pattern motifs and periodicity in a symbolic or numerical sequence.
    Missing data positions (-1, NaN, None) are preserved in their original time positions
    and break cycle continuity without compressing the sequence.

    Parameters
    ----------
    sequence : array-like or list
        Input sequence.
    max_period : int, optional
        Maximum period length to search. Defaults to len(sequence) // 2.
    tolerance : float, default=0.0
        Tolerable matching error for numeric sequences.
    missing_value : Any, default=-1
        Sentinel value representing missing entries.

    Returns
    -------
    is_periodic : bool
        Whether a recurring periodic motif was detected.
    period_length : int or None
        Length of the fundamental repeating cycle.
    repeating_motif : list
        The fundamental repeating unit.
    periodic_fidelity : float
        Score in [0, 1] indicating consistency of repetition.
    harmonic_periods : list[int]
        Other candidate recurring periods.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if max_period is not None and max_period <= 0:
        raise ValueError(f"max_period must be a positive integer, got {max_period}")

    seq = list(sequence)
    n = len(seq)
    if n < 4:
        return False, None, [], 0.0, []

    # Count valid elements
    valid_elements = [x for x in seq if not _is_missing_element(x, missing_value)]
    if len(valid_elements) < 4:
        return False, None, [], 0.0, []

    limit = min(n // 2, max_period if max_period is not None else n // 2)
    if limit < 2:
        # Check for constant period = 1
        comparisons = n - 1
        valid_comps = 0
        matches = 0
        for i in range(comparisons):
            if not _is_missing_element(seq[i], missing_value) and not _is_missing_element(seq[i + 1], missing_value):
                valid_comps += 1
                if _are_elements_equal(seq[i], seq[i + 1], tolerance, missing_value):
                    matches += 1
        if valid_comps >= 3 and (matches / valid_comps) >= 0.85:
            first_valid = next(x for x in seq if not _is_missing_element(x, missing_value))
            return True, 1, [first_valid], float(matches / valid_comps), []
        return False, None, [], 0.0, []

    best_p = None
    best_fidelity = 0.0
    harmonics = []

    # Test candidate period lengths p in 1..limit
    for p in range(1, limit + 1):
        matches = 0
        valid_comparisons = 0
        comparisons = n - p
        for i in range(comparisons):
            if not _is_missing_element(seq[i], missing_value) and not _is_missing_element(seq[i + p], missing_value):
                valid_comparisons += 1
                if _are_elements_equal(seq[i], seq[i + p], tolerance, missing_value):
                    matches += 1

        if valid_comparisons >= max(3, p * 2):
            fidelity = matches / valid_comparisons
            if fidelity >= 0.85:
                harmonics.append(p)
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_p = p

    if best_p is not None and best_fidelity >= 0.85:
        # Reduce to fundamental period using tolerance-aware comparison
        fund_p = best_p
        for h in sorted(harmonics):
            if fund_p % h == 0 and h < fund_p:
                comp_h = n - h
                m_h = 0
                v_h = 0
                for i in range(comp_h):
                    if not _is_missing_element(seq[i], missing_value) and not _is_missing_element(seq[i + h], missing_value):
                        v_h += 1
                        if _are_elements_equal(seq[i], seq[i + h], tolerance, missing_value):
                            m_h += 1
                if v_h >= max(3, h * 2) and (m_h / v_h) >= 0.85:
                    fund_p = h
                    break

        # Extract repeating motif unit (preserving positions)
        motif = seq[:fund_p]
        return True, fund_p, motif, float(best_fidelity), [h for h in harmonics if h != fund_p]

    return False, None, [], float(best_fidelity), []


def estimate_linear_trend(values: np.ndarray, timestamps_sec: np.ndarray | None = None) -> float:
    """
    Computes linear trend slope per pulse (or per second if timestamps provided).

    Parameters
    ----------
    values : np.ndarray
        1D array of values.
    timestamps_sec : np.ndarray, optional
        1D array of timestamps.

    Returns
    -------
    slope : float
        Linear regression rate of change.
    """
    valid = np.isfinite(values)
    if np.sum(valid) < 2:
        return 0.0

    y = values[valid]
    if timestamps_sec is not None and len(timestamps_sec) == len(values):
        t_valid = timestamps_sec[valid]
        t_span = np.max(t_valid) - np.min(t_valid)
        if t_span > 1e-12:
            x = t_valid - np.min(t_valid)
        else:
            x = np.arange(len(y), dtype=np.float64)
    else:
        x = np.arange(len(y), dtype=np.float64)

    # Linear least squares slope
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 1e-15:
        return 0.0

    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    return slope
