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

from .models import SignalDurationResult
from ._common import (
    validate_sample_rate,
    validate_nonnegative_durations,
    ensure_1d_float_array,
    ensure_1d_int_array,
    robust_nan_statistics,
)


class SignalDuration:
    """
    Signal and pulse duration analysis calculator.

    Computes pulse durations given pulse boundary indices and sampling rate,
    or calculates statistical metrics over supplied pulse duration sequences.
    Enforces strict physical non-negativity: duration >= 0, end >= start.
    Pulse start and end indices use inclusive boundaries by default
    (pulse_width_samples = stop_idx - start_idx + 1), aligning with Pulse Detector conventions.
    Does NOT perform pulse boundary detection.
    """

    def analyze(
        self,
        pulse_start_indices: np.ndarray | Sequence[int] | None = None,
        pulse_end_indices: np.ndarray | Sequence[int] | None = None,
        sample_rate_hz: float | None = None,
        pulse_durations: np.ndarray | Sequence[float] | None = None,
        inclusive_bounds: bool = True,
    ) -> SignalDurationResult:
        """
        Executes signal duration calculation.
        """
        # Mode A: Direct durations provided
        if pulse_durations is not None:
            durations = ensure_1d_float_array(pulse_durations, name="pulse_durations")
            validate_nonnegative_durations(durations, name="pulse_durations")
        elif pulse_start_indices is not None and pulse_end_indices is not None:
            fs = validate_sample_rate(sample_rate_hz)
            starts_list = ensure_1d_int_array(pulse_start_indices, name="pulse_start_indices")
            ends_list = ensure_1d_int_array(pulse_end_indices, name="pulse_end_indices")
            starts = np.array(starts_list, dtype=np.float64)
            ends = np.array(ends_list, dtype=np.float64)
            if len(starts) != len(ends):
                raise ValueError(f"pulse_start_indices length ({len(starts)}) != pulse_end_indices length ({len(ends)})")

            if len(starts) > 0:
                if np.any(ends < starts):
                    bad_idx = np.where(ends < starts)[0][0]
                    raise ValueError(
                        f"Pulse end index ({ends[bad_idx]}) cannot precede start index ({starts[bad_idx]}) at index {bad_idx}"
                    )

            sample_spans = (ends - starts + 1.0) if inclusive_bounds else (ends - starts)
            durations = sample_spans / fs
            validate_nonnegative_durations(durations, name="calculated pulse durations")
        else:
            raise ValueError("Either pulse_durations or both pulse_start_indices and pulse_end_indices must be provided")

        stats = robust_nan_statistics(durations)

        return SignalDurationResult(
            durations_sec=durations,
            pulse_count=stats["valid_count"],
            mean_duration_sec=stats["mean"],
            median_duration_sec=stats["median"],
            min_duration_sec=stats["min"],
            max_duration_sec=stats["max"],
            std_duration_sec=stats["std"],
        )


def compute_signal_duration(
    pulse_start_indices: np.ndarray | Sequence[int] | None = None,
    pulse_end_indices: np.ndarray | Sequence[int] | None = None,
    sample_rate_hz: float | None = None,
    pulse_durations: np.ndarray | Sequence[float] | None = None,
    inclusive_bounds: bool = True,
) -> SignalDurationResult:
    """
    Functional API to compute signal duration statistics.

    Parameters
    ----------
    pulse_start_indices : Sequence[int] | np.ndarray | None
        Sample indices of pulse start / leading edges (inclusive).
    pulse_end_indices : Sequence[int] | np.ndarray | None
        Sample indices of pulse end / trailing edges (inclusive).
    sample_rate_hz : float | None
        Sampling rate in Hz.
    pulse_durations : Sequence[float] | np.ndarray | None
        Direct sequence of pulse duration measurements in seconds.
    inclusive_bounds : bool
        Whether pulse start and end indices are inclusive boundaries (default: True, stop - start + 1).

    Returns
    -------
    SignalDurationResult
        Summary statistics and duration array.
    """
    calc = SignalDuration()
    return calc.analyze(
        pulse_start_indices=pulse_start_indices,
        pulse_end_indices=pulse_end_indices,
        sample_rate_hz=sample_rate_hz,
        pulse_durations=pulse_durations,
        inclusive_bounds=inclusive_bounds,
    )


__all__ = ["SignalDuration", "compute_signal_duration"]


def main() -> None:
    print("=" * 60)
    print("  Signal Duration Demonstration")
    print("=" * 60)
    starts = [100, 500, 900]
    ends = [150, 560, 950]
    fs = 1e6
    res = compute_signal_duration(pulse_start_indices=starts, pulse_end_indices=ends, sample_rate_hz=fs)
    print(f"Durations (s): {res.durations_sec}, Mean: {res.mean_duration_sec * 1e6:.2f} us")
    print("=" * 60)


if __name__ == "__main__":
    main()
