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

from .models import InterPulseIntervalResult
from ._common import (
    ensure_1d_float_array,
    ensure_1d_int_array,
    compute_gap_aware_diff,
    validate_strictly_positive_intervals,
    robust_nan_statistics,
)


class InterPulseInterval:
    """
    Inter-Pulse Interval (IPI) calculator.

    Computes time differences between strictly consecutive valid pulse TOAs.
    Preserves missing-pulse gaps: non-finite/NaN TOAs break adjacency and produce NaN intervals.
    Supports integer nanosecond TOAs to prevent precision loss on large epoch timestamps.
    Enforces physical validity: finite intervals must be strictly positive (> 0).
    """

    def analyze(
        self,
        toas: np.ndarray | Sequence[float] | None = None,
        toas_ns: np.ndarray | Sequence[int] | None = None,
        previous_final_toa_sec: float | None = None,
        previous_final_toa_ns: int | None = None,
        timing_reference: str | None = None,
    ) -> InterPulseIntervalResult:
        """
        Executes IPI calculation from ordered TOAs with optional streaming cross-frame continuity.
        """
        cross_frame_conn = False

        if toas_ns is not None:
            arr_ns = ensure_1d_int_array(toas_ns)
            if previous_final_toa_ns is not None and len(arr_ns) > 0:
                # Prepend previous frame's final TOA
                full_ns = [int(previous_final_toa_ns)] + arr_ns
                cross_frame_conn = True
            else:
                full_ns = arr_ns

            n = len(full_ns)
            if n <= 1:
                return InterPulseIntervalResult(
                    intervals_sec=np.array([], dtype=np.float64),
                    valid_intervals_count=0,
                    gap_count=0,
                    total_pairs_count=0,
                )

            diffs_ns = [int(full_ns[i + 1] - full_ns[i]) for i in range(n - 1)]
            intervals = np.array([float(d) / 1e9 for d in diffs_ns], dtype=np.float64)
            res_intervals_ns = [int(d) for d in diffs_ns]
            timing_ref = timing_reference or "externally_absolute"

        elif toas is not None:
            arr = ensure_1d_float_array(toas, name="toas")
            if previous_final_toa_sec is not None and len(arr) > 0 and np.isfinite(previous_final_toa_sec):
                full_arr = np.concatenate([[float(previous_final_toa_sec)], arr])
                cross_frame_conn = True
            else:
                full_arr = arr

            n = len(full_arr)
            if n <= 1:
                return InterPulseIntervalResult(
                    intervals_sec=np.array([], dtype=np.float64),
                    valid_intervals_count=0,
                    gap_count=0,
                    total_pairs_count=0,
                )
            intervals = compute_gap_aware_diff(full_arr)
            res_intervals_ns = None
            timing_ref = timing_reference or "capture_relative"
        else:
            raise ValueError("Either toas or toas_ns must be provided")

        # Physical validation: intervals must be strictly positive
        validate_strictly_positive_intervals(intervals, name="Inter-pulse intervals")

        stats = robust_nan_statistics(intervals)
        total_pairs = len(intervals)
        valid_cnt = stats["valid_count"]
        gap_cnt = total_pairs - valid_cnt

        return InterPulseIntervalResult(
            intervals_sec=intervals,
            intervals_ns=res_intervals_ns,
            timing_reference=timing_ref,
            valid_intervals_count=valid_cnt,
            gap_count=gap_cnt,
            total_pairs_count=total_pairs,
            mean_interval_sec=stats["mean"],
            median_interval_sec=stats["median"],
            min_interval_sec=stats["min"],
            max_interval_sec=stats["max"],
            std_interval_sec=stats["std"],
            cross_frame_connected=cross_frame_conn,
        )



def compute_inter_pulse_interval(
    toas: np.ndarray | Sequence[float] | None = None,
    toas_ns: np.ndarray | Sequence[int] | None = None,
    previous_final_toa_sec: float | None = None,
    previous_final_toa_ns: int | None = None,
    timing_reference: str | None = None,
) -> InterPulseIntervalResult:
    """
    Functional API to compute Inter-Pulse Intervals (IPI).

    Parameters
    ----------
    toas : Sequence[float] | np.ndarray | None
        Ordered sequence of Time-of-Arrivals in seconds.
    toas_ns : Sequence[int] | np.ndarray | None
        Ordered sequence of Time-of-Arrivals in integer nanoseconds.
    previous_final_toa_sec : float | None
        Optional final TOA from previous frame for streaming continuity in seconds.
    previous_final_toa_ns : int | None
        Optional final TOA from previous frame in integer nanoseconds.
    timing_reference : str | None
        Explicit timing reference type ('hardware_absolute', 'capture_relative', 'externally_absolute').

    Returns
    -------
    InterPulseIntervalResult
        Interval array (with NaNs at gaps) and statistical metrics.
    """
    calc = InterPulseInterval()
    return calc.analyze(
        toas=toas,
        toas_ns=toas_ns,
        previous_final_toa_sec=previous_final_toa_sec,
        previous_final_toa_ns=previous_final_toa_ns,
        timing_reference=timing_reference,
    )


__all__ = ["InterPulseInterval", "compute_inter_pulse_interval"]


def main() -> None:
    print("=" * 60)
    print("  Inter-Pulse Interval Demonstration")
    print("=" * 60)
    toas = [0.0, 0.001, np.nan, 0.003, 0.004]
    res = compute_inter_pulse_interval(toas)
    print(f"TOAs: {toas}")
    print(f"IPI Intervals (s): {res.intervals_sec}")
    print(f"Valid count: {res.valid_intervals_count}, Gaps: {res.gap_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
