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

from .models import JitterResult
from ._common import (
    ensure_1d_float_array,
    validate_strictly_positive_intervals,
    robust_nan_statistics,
)


class TimingJitter:
    """
    Timing Jitter analysis calculator.

    Quantifies pulse-to-pulse timing jitter, deviation from nominal PRI,
    standard deviation, RMS jitter, peak-to-peak jitter, and relative jitter percentage.
    Enforces physical validity: nominal_pri must be finite and strictly positive (> 0).
    Does NOT call PRI.py or IPI.py.
    """

    def analyze(
        self,
        intervals: np.ndarray | Sequence[float],
        nominal_pri_sec: float | None = None,
    ) -> JitterResult:
        """
        Executes jitter quantification on a pulse interval sequence.
        """
        arr = ensure_1d_float_array(intervals, name="intervals")
        valid = arr[np.isfinite(arr)]
        n_valid = len(valid)

        if n_valid == 0:
            return JitterResult()

        # Validate intervals strictly positive
        validate_strictly_positive_intervals(valid, name="Intervals for jitter calculation")

        if nominal_pri_sec is not None:
            nom_pri = float(nominal_pri_sec)
            if not np.isfinite(nom_pri) or nom_pri <= 0:
                raise ValueError(f"nominal_pri_sec must be a positive finite number (> 0), got {nominal_pri_sec}")
        else:
            nom_pri = float(np.median(valid))
            if nom_pri <= 0:
                raise ValueError(f"Inferred nominal PRI must be strictly positive, got {nom_pri}")

        # Deviation of each interval from nominal
        deviations = arr - nom_pri
        valid_dev = valid - nom_pri

        std_j = float(np.std(valid_dev, ddof=1)) if n_valid > 1 else 0.0
        rms_j = float(np.sqrt(np.mean(valid_dev ** 2)))
        p2p_j = float(np.ptp(valid_dev)) if n_valid > 0 else 0.0

        rel_pct = None
        if nom_pri > 0:
            rel_pct = float((std_j / nom_pri) * 100.0)

        # Infer basic jitter distribution
        dist = "unknown"
        if n_valid >= 10 and std_j > 0:
            mean_d = np.mean(valid_dev)
            m4 = np.mean((valid_dev - mean_d) ** 4)
            kurt = (m4 / (std_j ** 4)) if std_j > 0 else 3.0
            if abs(kurt - 3.0) < 0.6:
                dist = "gaussian"
            elif kurt < 2.0:
                dist = "uniform"
            elif kurt > 4.5:
                dist = "heavy_tailed"
            else:
                dist = "generalized_random"

        return JitterResult(
            jitter_deviations_sec=deviations,
            sample_count=n_valid,
            std_jitter_sec=std_j,
            rms_jitter_sec=rms_j,
            peak_to_peak_jitter_sec=p2p_j,
            relative_jitter_percent=rel_pct,
            nominal_pri_sec=nom_pri,
            jitter_distribution=dist,
        )


def compute_jitter(
    intervals: np.ndarray | Sequence[float],
    nominal_pri_sec: float | None = None,
) -> JitterResult:
    """
    Functional API to compute timing jitter.

    Parameters
    ----------
    intervals : Sequence[float] | np.ndarray
        Pulse interval sequence (e.g. from IPI or PRI) in seconds.
    nominal_pri_sec : float | None
        Optional reference nominal PRI in seconds. If None, median interval is used.

    Returns
    -------
    JitterResult
        Jitter metrics, deviations array, and relative jitter percentage.
    """
    calc = TimingJitter()
    return calc.analyze(intervals=intervals, nominal_pri_sec=nominal_pri_sec)


__all__ = ["TimingJitter", "compute_jitter"]


def main() -> None:
    print("=" * 60)
    print("  Jitter Module Demonstration")
    print("=" * 60)
    np.random.seed(42)
    nominal = 1e-3
    intervals = nominal + np.random.normal(0, 1e-6, 100)
    res = compute_jitter(intervals, nominal_pri_sec=nominal)
    print(f"Jitter Std: {res.std_jitter_sec * 1e6:.3f} us, Relative: {res.relative_jitter_percent:.3f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
