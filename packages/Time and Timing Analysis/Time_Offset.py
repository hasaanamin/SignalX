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

from .models import TimeOffsetResult
from ._common import (
    ensure_1d_float_array,
    ensure_1d_int_array,
    robust_nan_statistics,
)


class TimeOffset:
    """
    Time Offset (delta-t / TDOA) analysis calculator.

    Computes time differences between matched timing observations from multiple
    channels, receivers, or reference clocks.
    Supports both absolute integer-nanosecond events (offset_ns = b_ns - a_ns)
    and relative floating-point seconds without float precision loss.
    """

    def analyze(
        self,
        events_a: np.ndarray | Sequence[float] | None = None,
        events_b: np.ndarray | Sequence[float] | None = None,
        events_a_ns: np.ndarray | Sequence[int] | None = None,
        events_b_ns: np.ndarray | Sequence[int] | None = None,
        matched_pairs: Sequence[tuple[float, float] | list[float]] | tuple[np.ndarray, np.ndarray] | list[Any] | None = None,
        matched_pairs_ns: Sequence[tuple[int, int] | list[int]] | list[Any] | None = None,
        timing_reference: str | None = None,
    ) -> TimeOffsetResult:
        """
        Executes time offset calculation.
        Offset is defined as: delta_t = events_b - events_a.
        """
        # Mode A: Integer nanosecond events provided directly
        if events_a_ns is not None and events_b_ns is not None:
            a_ns = ensure_1d_int_array(events_a_ns, name="events_a_ns")
            b_ns = ensure_1d_int_array(events_b_ns, name="events_b_ns")
            if len(a_ns) != len(b_ns):
                raise ValueError(f"events_a_ns length ({len(a_ns)}) != events_b_ns length ({len(b_ns)})")

            offsets_ns: list[int] | np.ndarray = [int(b - a) for a, b in zip(a_ns, b_ns)]
            offsets_sec = np.array([float(d) / 1e9 for d in offsets_ns], dtype=np.float64)
            ref = timing_reference or "externally_absolute"

        # Mode B: Matched integer nanosecond pairs provided
        elif matched_pairs_ns is not None:
            pairs_ns = list(matched_pairs_ns)
            offsets_ns = []
            for p in pairs_ns:
                if not isinstance(p, (tuple, list, np.ndarray)):
                    raise ValueError(f"Each element of matched_pairs_ns must be a pair (got {type(p).__name__})")
                if len(p) != 2:
                    raise ValueError(f"Each matched pair in matched_pairs_ns must have length 2 (got {len(p)})")
                p0, p1 = p[0], p[1]
                if isinstance(p0, (bool, np.bool_)) or isinstance(p1, (bool, np.bool_)):
                    raise TypeError("Elements of matched_pairs_ns cannot be boolean values")
                if isinstance(p0, str) or isinstance(p1, str):
                    raise TypeError("Elements of matched_pairs_ns cannot be strings")
                if isinstance(p0, (float, np.floating)):
                    raise TypeError(f"Elements of matched_pairs_ns must be integers, got float {p0}")
                elif not isinstance(p0, (int, np.integer)):
                    raise TypeError(f"Elements of matched_pairs_ns must be integer-like, got {type(p0).__name__}")
                if isinstance(p1, (float, np.floating)):
                    raise TypeError(f"Elements of matched_pairs_ns must be integers, got float {p1}")
                elif not isinstance(p1, (int, np.integer)):
                    raise TypeError(f"Elements of matched_pairs_ns must be integer-like, got {type(p1).__name__}")
                offsets_ns.append(int(p1) - int(p0))

            offsets_sec = np.array([float(d) / 1e9 for d in offsets_ns], dtype=np.float64)
            ref = timing_reference or "externally_absolute"

        # Mode C: Matched pairs sequence provided (handles list of pairs, tuple of arrays, array of pairs)
        elif matched_pairs is not None:
            # Handle tuple of 2 arrays: (array_a, array_b)
            if isinstance(matched_pairs, tuple) and len(matched_pairs) == 2 and isinstance(matched_pairs[0], (np.ndarray, list, tuple)) and not isinstance(matched_pairs[0][0] if len(matched_pairs[0]) > 0 else 0, (list, tuple)):
                a = ensure_1d_float_array(matched_pairs[0])
                b = ensure_1d_float_array(matched_pairs[1])
            elif isinstance(matched_pairs, np.ndarray) and matched_pairs.ndim == 2 and matched_pairs.shape[1] == 2:
                a = matched_pairs[:, 0].astype(np.float64)
                b = matched_pairs[:, 1].astype(np.float64)
            else:
                pairs = list(matched_pairs)
                if len(pairs) == 0:
                    a = np.array([], dtype=np.float64)
                    b = np.array([], dtype=np.float64)
                else:
                    a = np.array([float(p[0]) for p in pairs], dtype=np.float64)
                    b = np.array([float(p[1]) for p in pairs], dtype=np.float64)

            if len(a) != len(b):
                raise ValueError(f"Matched pair lengths mismatch: len(a)={len(a)}, len(b)={len(b)}")

            offsets_sec = b - a
            offsets_ns = None
            ref = timing_reference or "capture_relative"

        # Mode D: Floating seconds arrays provided
        elif events_a is not None and events_b is not None:
            a = ensure_1d_float_array(events_a)
            b = ensure_1d_float_array(events_b)
            if len(a) != len(b):
                raise ValueError(f"events_a length ({len(a)}) != events_b length ({len(b)})")
            offsets_sec = b - a
            offsets_ns = None
            ref = timing_reference or "capture_relative"

        else:
            raise ValueError("Must provide either (events_a, events_b), (events_a_ns, events_b_ns), matched_pairs, or matched_pairs_ns")

        stats = robust_nan_statistics(offsets_sec)

        return TimeOffsetResult(
            offsets_sec=offsets_sec,
            offsets_ns=offsets_ns,
            pair_count=stats["valid_count"],
            mean_offset_sec=stats["mean"],
            median_offset_sec=stats["median"],
            std_offset_sec=stats["std"],
            min_offset_sec=stats["min"],
            max_offset_sec=stats["max"],
            timing_reference=ref,
        )


def compute_time_offset(
    events_a: np.ndarray | Sequence[float] | None = None,
    events_b: np.ndarray | Sequence[float] | None = None,
    events_a_ns: np.ndarray | Sequence[int] | None = None,
    events_b_ns: np.ndarray | Sequence[int] | None = None,
    matched_pairs: Sequence[tuple[float, float] | list[float]] | tuple[np.ndarray, np.ndarray] | list[Any] | None = None,
    matched_pairs_ns: Sequence[tuple[int, int] | list[int]] | list[Any] | None = None,
    timing_reference: str | None = None,
) -> TimeOffsetResult:
    """
    Functional API to compute Time Offset between matched timing observations.

    Parameters
    ----------
    events_a : Sequence[float] | np.ndarray | None
        Event times from reference channel / station A in seconds.
    events_b : Sequence[float] | np.ndarray | None
        Event times from target channel / station B in seconds.
    events_a_ns : Sequence[int] | np.ndarray | None
        Event times in integer nanoseconds from reference channel A.
    events_b_ns : Sequence[int] | np.ndarray | None
        Event times in integer nanoseconds from target channel B.
    matched_pairs : Sequence[tuple[float, float]] | tuple[np.ndarray, np.ndarray] | None
        Sequence of (time_a, time_b) matched pairs in seconds.
    matched_pairs_ns : Sequence[tuple[int, int]] | None
        Sequence of (time_a_ns, time_b_ns) matched pairs in integer nanoseconds.
    timing_reference : str | None
        Explicit timing reference type ('hardware_absolute', 'capture_relative', 'coherent_relative').

    Returns
    -------
    TimeOffsetResult
        Calculated time offsets and statistics.
    """
    calc = TimeOffset()
    return calc.analyze(
        events_a=events_a,
        events_b=events_b,
        events_a_ns=events_a_ns,
        events_b_ns=events_b_ns,
        matched_pairs=matched_pairs,
        matched_pairs_ns=matched_pairs_ns,
        timing_reference=timing_reference,
    )



__all__ = ["TimeOffset", "compute_time_offset"]


def main() -> None:
    print("=" * 60)
    print("  Time Offset Demonstration")
    print("=" * 60)
    a_ns = [1700000000000000000, 1700000000001000000]
    b_ns = [1700000000000000050, 1700000000001000060]
    res = compute_time_offset(events_a_ns=a_ns, events_b_ns=b_ns)
    print(f"Offsets (ns): {res.offsets_ns}, Mean (s): {res.mean_offset_sec}")
    print("=" * 60)


if __name__ == "__main__":
    main()
