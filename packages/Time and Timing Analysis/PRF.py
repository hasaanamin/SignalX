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

from .models import PRFResult
from ._common import (
    ensure_1d_float_array,
    validate_strictly_positive_intervals,
    robust_nan_statistics,
)


class PulseRepetitionFrequency:
    """
    Pulse Repetition Frequency (PRF) analysis calculator.

    Computes PRF in Hz as the inverse of PRI (1 / PRI) for each interval
    and computes nominal, mean, median, min, max PRF metrics.
    Enforces physical validity: PRI values must be strictly positive (> 0).
    Does NOT call PRI.py.
    """

    def analyze(
        self,
        pri_sequence: np.ndarray | Sequence[float] | float,
    ) -> PRFResult:
        """
        Executes PRF calculation from a PRI sequence or scalar PRI.
        """
        if isinstance(pri_sequence, (int, float, np.number)):
            pri_val = float(pri_sequence)
            if not np.isfinite(pri_val) or pri_val <= 0:
                raise ValueError(f"Scalar PRI must be a positive finite number (> 0), got {pri_val}")
            prf_seq = np.array([1.0 / pri_val], dtype=np.float64)
        else:
            pri_arr = ensure_1d_float_array(pri_sequence, name="pri_sequence")
            if len(pri_arr) == 0:
                return PRFResult()

            # Physical validation: PRI must be strictly positive
            validate_strictly_positive_intervals(pri_arr, name="PRI values for PRF calculation")

            # Element-wise reciprocal with divide-by-zero protection
            with np.errstate(divide="ignore", invalid="ignore"):
                prf_seq = np.where((pri_arr > 0) & np.isfinite(pri_arr), 1.0 / pri_arr, np.nan)

        stats = robust_nan_statistics(prf_seq)
        nominal_prf = stats["median"]

        return PRFResult(
            prf_sequence_hz=prf_seq,
            prf_count=stats["valid_count"],
            nominal_prf_hz=nominal_prf,
            mean_prf_hz=stats["mean"],
            median_prf_hz=stats["median"],
            min_prf_hz=stats["min"],
            max_prf_hz=stats["max"],
        )


def compute_prf(
    pri_sequence: np.ndarray | Sequence[float] | float,
) -> PRFResult:
    """
    Functional API to compute Pulse Repetition Frequency (PRF).

    Parameters
    ----------
    pri_sequence : Sequence[float] | np.ndarray | float
        Pulse Repetition Interval(s) in seconds.

    Returns
    -------
    PRFResult
        PRF sequence in Hz and aggregate metrics.
    """
    calc = PulseRepetitionFrequency()
    return calc.analyze(pri_sequence=pri_sequence)


__all__ = ["PulseRepetitionFrequency", "compute_prf"]


def main() -> None:
    print("=" * 60)
    print("  PRF Module Demonstration")
    print("=" * 60)
    pris = [0.001, 0.001, 0.0005, 0.002]
    res = compute_prf(pris)
    print(f"PRFs (Hz): {res.prf_sequence_hz}, Mean: {res.mean_prf_hz:.1f} Hz")
    print("=" * 60)


if __name__ == "__main__":
    main()
