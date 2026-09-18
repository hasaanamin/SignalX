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

from .models import PRIResult
from ._common import (
    ensure_1d_float_array,
    compute_gap_aware_diff,
    compute_gap_aware_diff_ns,
    validate_strictly_positive_intervals,
    robust_nan_statistics,
)


class PulseRepetitionInterval:
    """
    Pulse Repetition Interval (PRI) analysis calculator.

    Computes nominal, mean, median, min, max, and standard deviation
    from a pre-computed interval sequence (IPI) or directly from pulse TOAs.
    Focuses strictly on interval measurement and physical validation without
    duplicating downstream Sequence and Pattern Analysis algorithms.
    """

    def analyze(
        self,
        intervals: np.ndarray | Sequence[float] | None = None,
        toas: np.ndarray | Sequence[float] | None = None,
        toas_ns: np.ndarray | Sequence[int] | None = None,
    ) -> PRIResult:
        """
        Executes PRI analysis.
        """
        if intervals is not None:
            pri_seq = ensure_1d_float_array(intervals, name="intervals")
        elif toas_ns is not None:
            pri_seq = compute_gap_aware_diff_ns(toas_ns)
        elif toas is not None:
            pri_seq = compute_gap_aware_diff(toas)
        else:
            raise ValueError("Either intervals, toas_ns, or toas must be provided")

        # Physical validation: PRI values must be strictly positive
        validate_strictly_positive_intervals(pri_seq, name="Pulse Repetition Intervals")

        stats = robust_nan_statistics(pri_seq)
        nominal = stats["median"]  # Median is a robust estimator for nominal PRI

        return PRIResult(
            pri_sequence_sec=pri_seq,
            pri_count=stats["valid_count"],
            nominal_pri_sec=nominal,
            mean_pri_sec=stats["mean"],
            median_pri_sec=stats["median"],
            min_pri_sec=stats["min"],
            max_pri_sec=stats["max"],
            std_pri_sec=stats["std"],
        )


def compute_pri(
    intervals: np.ndarray | Sequence[float] | None = None,
    toas: np.ndarray | Sequence[float] | None = None,
    toas_ns: np.ndarray | Sequence[int] | None = None,
) -> PRIResult:
    """
    Functional API to compute Pulse Repetition Interval (PRI) metrics.

    Parameters
    ----------
    intervals : Sequence[float] | np.ndarray | None
        Pre-computed interval sequence in seconds (e.g. from IPI module).
    toas : Sequence[float] | np.ndarray | None
        Pulse TOAs in seconds (if called independently without prior IPI calculation).
    toas_ns : Sequence[int] | np.ndarray | None
        Pulse TOAs in integer nanoseconds (preserves precision on large epoch timestamps).

    Returns
    -------
    PRIResult
        PRI sequence and statistical descriptors.
    """
    calc = PulseRepetitionInterval()
    return calc.analyze(intervals=intervals, toas=toas, toas_ns=toas_ns)


__all__ = ["PulseRepetitionInterval", "compute_pri"]


def main() -> None:
    print("=" * 60)
    print("  PRI Module Demonstration")
    print("=" * 60)
    intervals = [0.001, 0.001, 0.00101, 0.00099, 0.001]
    res = compute_pri(intervals=intervals)
    print(f"Nominal PRI: {res.nominal_pri_sec * 1e6:.2f} us, Mean: {res.mean_pri_sec * 1e6:.2f} us")
    print("=" * 60)


if __name__ == "__main__":
    main()
