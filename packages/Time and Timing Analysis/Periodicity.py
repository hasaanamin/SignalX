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

from .models import TimingPeriodicityResult
from ._common import ensure_1d_float_array


def _find_fundamental_lag(
    lag_scores: dict[int, float],
    n_samples: int,
    confidence_threshold: float = 0.65,
    min_cycles: float = 2.5,
) -> tuple[int | None, float, list[int]]:
    """
    Selects the true fundamental period lag, resolving harmonics to their base fundamental.

    If lag 3 and lag 6 both have high correlation, lag 3 is preferred if its correlation
    is statistically significant (e.g. >= 0.70 * score(6) or >= threshold).
    """
    candidates = [
        lag for lag, sc in lag_scores.items()
        if sc >= confidence_threshold and (n_samples / lag) >= min_cycles
    ]

    if not candidates:
        return None, 0.0, []

    candidates.sort()

    # Find the candidate that represents the smallest valid fundamental explaining the sequence
    max_lag = max(candidates, key=lambda l: lag_scores[l])
    max_score = lag_scores[max_lag]

    best_fund: int | None = None
    best_score: float = -1.0

    # Prefer the smallest divisor of the highest peaks that is itself a strong candidate
    for c_lag in candidates:
        c_score = lag_scores[c_lag]
        # If candidate is a divisor of max_lag and has sufficient correlation
        if max_lag % c_lag == 0 and c_score >= (max_score * 0.70):
            best_fund = c_lag
            best_score = c_score
            break
        elif c_score >= (max_score * 0.85) and (best_fund is None or c_lag < best_fund):
            best_fund = c_lag
            best_score = c_score
            break

    if best_fund is None:
        best_fund = max_lag
        best_score = max_score

    # Identify harmonics (integer multiples of best_fund)
    harmonics: list[int] = [
        lag for lag in candidates
        if lag != best_fund and (lag % best_fund == 0)
    ]

    return best_fund, best_score, harmonics


class TimingPeriodicity:
    """
    Timing Periodicity analysis calculator.

    Evaluates whether a pre-computed timing sequence (e.g. PRI, IPI, or BRI)
    exhibits a repeating discrete pattern (e.g. staggered PRI, periodic jitter).
    Implements harmonic reduction to reliably identify the fundamental period.
    Suppresses false positives on short or random noise sequences.
    Missing values (NaNs) are handled explicitly without false compression.
    """

    def analyze(
        self,
        timing_sequence: np.ndarray | Sequence[float],
        max_period_samples: int = 32,
        min_cycles: float = 2.5,
        confidence_threshold: float = 0.70,
    ) -> TimingPeriodicityResult:
        """
        Executes periodicity detection via gap-aware normalized autocorrelation and template fitting.
        """
        arr = ensure_1d_float_array(timing_sequence, name="timing_sequence")
        n = len(arr)

        if n < 6:
            return TimingPeriodicityResult(
                is_periodic=False,
                periodicity_confidence=0.0,
                metrics={"reason": "insufficient_samples", "sample_count": n},
            )

        valid_mask = np.isfinite(arr)
        n_valid = int(np.sum(valid_mask))
        if n_valid < 6:
            return TimingPeriodicityResult(
                is_periodic=False,
                periodicity_confidence=0.0,
                metrics={"reason": "insufficient_valid_samples", "valid_count": n_valid},
            )

        # Center finite values around mean
        mean_val = float(np.mean(arr[valid_mask]))
        centered = np.where(valid_mask, arr - mean_val, 0.0)

        # Check if sequence is virtually constant (period 1)
        std_val = float(np.std(arr[valid_mask]))
        if std_val < 1e-12 or (mean_val > 0 and (std_val / mean_val) < 1e-4):
            return TimingPeriodicityResult(
                is_periodic=True,
                fundamental_period_sec=mean_val,
                fundamental_period_samples=1,
                periodicity_confidence=1.0,
                periodic_pattern=[mean_val],
                harmonic_lags=[],
                harmonic_periods_samples=[],
                harmonic_periods_sec=[],
                metrics={"type": "constant", "cv": (std_val / mean_val) if mean_val > 0 else 0.0},
            )

        # Gap-aware normalized autocorrelation
        # For short sequences (n < 20), require at least 3.0 cycles to avoid small-sample spurious matches
        effective_min_cycles = max(min_cycles, 3.0 if n <= 20 else 2.5)
        effective_threshold = max(confidence_threshold, 0.78 if n <= 20 else 0.68)

        max_lag = min(int(max_period_samples), int(n / effective_min_cycles))
        if max_lag < 1:
            return TimingPeriodicityResult(
                is_periodic=False,
                periodicity_confidence=0.0,
                metrics={"reason": "sequence_too_short_for_min_cycles", "n": n, "min_cycles": effective_min_cycles},
            )

        lag_scores: dict[int, float] = {}
        for lag in range(1, max_lag + 1):
            x0 = centered[:-lag]
            x1 = centered[lag:]
            m0 = valid_mask[:-lag]
            m1 = valid_mask[lag:]
            pair_mask = m0 & m1
            k_pairs = int(np.sum(pair_mask))

            # Need sufficient valid overlap pairs
            if k_pairs < (4 if n <= 15 else 5):
                continue

            denom = np.sqrt(np.sum(x0[pair_mask] ** 2) * np.sum(x1[pair_mask] ** 2))
            if denom > 1e-15:
                num = np.sum(x0[pair_mask] * x1[pair_mask])
                corr = float(num / denom)
                lag_scores[lag] = float(np.clip(corr, -1.0, 1.0))

        if not lag_scores:
            return TimingPeriodicityResult(
                is_periodic=False,
                periodicity_confidence=0.0,
                metrics={"reason": "no_valid_lags_evaluated"},
            )

        # Harmonic reduction & fundamental period selection
        fund_lag, fund_score, harmonic_lags = _find_fundamental_lag(
            lag_scores=lag_scores,
            n_samples=n,
            confidence_threshold=effective_threshold,
            min_cycles=effective_min_cycles,
        )

        is_periodic = False
        pattern: list[float] | None = None
        fund_sec: float | None = None
        harmonic_periods_sec: list[float] = []
        r2_adj = 0.0

        if fund_lag is not None and fund_score >= effective_threshold:
            # Fit periodic template and evaluate explained variance (adjusted R^2)
            pattern_accum: list[list[float]] = [[] for _ in range(fund_lag)]
            for i in range(n):
                if valid_mask[i]:
                    pattern_accum[i % fund_lag].append(float(arr[i]))

            pattern = [float(np.mean(p_vals)) if p_vals else mean_val for p_vals in pattern_accum]
            fund_sec = float(np.sum(pattern))

            # Compute template reconstruction residual
            ss_tot = float(np.sum((arr[valid_mask] - mean_val) ** 2))
            pred = np.array([pattern[i % fund_lag] for i in range(n)])
            ss_res = float(np.sum((arr[valid_mask] - pred[valid_mask]) ** 2))

            df_tot = max(1, n_valid - 1)
            df_res = max(1, n_valid - fund_lag)

            if ss_tot > 1e-15 and df_res > 0:
                r2_adj = 1.0 - ((ss_res / df_res) / (ss_tot / df_tot))
            else:
                r2_adj = 1.0

            # Require strong adjusted R^2 (explained variance) to eliminate random noise false positives
            min_r2_adj = 0.75 if n <= 20 else 0.55
            if r2_adj >= min_r2_adj and (n / fund_lag) >= effective_min_cycles:
                is_periodic = True
                for h_lag in harmonic_lags:
                    harmonic_periods_sec.append(float(fund_sec * (h_lag / fund_lag)))
            else:
                is_periodic = False
                pattern = None
                fund_sec = None
                harmonic_lags = []



        return TimingPeriodicityResult(
            is_periodic=bool(is_periodic),
            fundamental_period_sec=fund_sec,
            fundamental_period_samples=fund_lag if is_periodic else None,
            periodicity_confidence=float(max(0.0, fund_score)),
            periodic_pattern=pattern,
            harmonic_lags=harmonic_lags if is_periodic else [],
            harmonic_periods_samples=harmonic_lags if is_periodic else [],
            harmonic_periods_sec=harmonic_periods_sec,
            metrics={
                "fundamental_lag": fund_lag,
                "fundamental_score": fund_score,
                "lag_autocorrelations": lag_scores,
            },
        )


def compute_periodicity(
    timing_sequence: np.ndarray | Sequence[float],
    max_period_samples: int = 32,
    min_cycles: float = 2.5,
    confidence_threshold: float = 0.70,
) -> TimingPeriodicityResult:
    """
    Functional API to detect and quantify timing sequence periodicity.

    Parameters
    ----------
    timing_sequence : Sequence[float] | np.ndarray
        Sequence of timing measurements (e.g. PRI, IPI, BRI) in seconds.
    max_period_samples : int
        Maximum discrete lag / pattern length to test.
    min_cycles : float
        Minimum number of repetitions required in the sequence.
    confidence_threshold : float
        Normalized correlation threshold to declare periodicity.

    Returns
    -------
    TimingPeriodicityResult
        Periodicity status, fundamental period (in samples and seconds), and confidence.
    """
    calc = TimingPeriodicity()
    return calc.analyze(
        timing_sequence=timing_sequence,
        max_period_samples=max_period_samples,
        min_cycles=min_cycles,
        confidence_threshold=confidence_threshold,
    )


__all__ = ["TimingPeriodicity", "compute_periodicity"]


def main() -> None:
    print("=" * 60)
    print("  Periodicity Module Demonstration")
    print("=" * 60)
    # Stagger pattern with true fundamental period 3
    pattern = [100e-6, 120e-6, 150e-6]
    stagger = pattern * 10
    res = compute_periodicity(stagger)
    print(f"Is periodic: {res.is_periodic}, Fundamental lag: {res.fundamental_period_samples}")
    print(f"Harmonics: {res.harmonic_lags}, Pattern: {res.periodic_pattern}")
    print("=" * 60)


if __name__ == "__main__":
    main()
