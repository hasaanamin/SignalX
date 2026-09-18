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

import math
from typing import Any, List, Optional, Union
import numpy as np

from .models import SequencePeriodicityResult
from .base import BaseSequencePatternMetric, SequencePatternMetricFactory
from ._common import detect_symbolic_periodicity, estimate_discrete_states


class SequencePeriodicity(BaseSequencePatternMetric):
    """
    Evaluates sequence periodicity, fundamental cycle length, repeating motif extraction,
    and harmonic candidates across any symbolic, categorical, or numerical sequence.
    """

    metric_name = "periodicity"

    def compute(
        self,
        sequence: Any,
        max_period: Optional[int] = None,
        tolerance: float = 0.0,
        quantize_continuous: bool = False,
        **kwargs: Any,
    ) -> SequencePeriodicityResult:
        """
        Analyzes periodicity and extracts repeating motifs from input sequence.

        Parameters
        ----------
        sequence : array-like or list
            Input discrete states, categorical strings, or continuous numerical values.
        max_period : int, optional
            Maximum period length to search.
        tolerance : float, default=0.0
            Tolerance for continuous numerical comparison.
        quantize_continuous : bool, default=False
            Whether to quantize continuous numerical sequence into discrete states before period analysis.

        Returns
        -------
        SequencePeriodicityResult
        """
        if tolerance is not None:
            tol_val = float(tolerance)
            if tol_val < 0.0 or not math.isfinite(tol_val):
                raise ValueError(f"tolerance must be non-negative and finite, got {tolerance}")
        if max_period is not None:
            mp = int(max_period)
            if mp < 1:
                raise ValueError(f"max_period must be >= 1, got {max_period}")

        if sequence is None:
            return SequencePeriodicityResult(
                is_periodic=False,
                period_length=None,
                repeating_motif=[],
                num_cycles=0.0,
                periodic_fidelity=0.0,
                harmonic_periods=[],
                confidence=0.0,
                valid=False,
                diagnostics={"reason": "null_sequence"},
            )

        seq_list = list(sequence)
        n = len(seq_list)

        if n < 4:
            return SequencePeriodicityResult(
                is_periodic=False,
                period_length=None,
                repeating_motif=[],
                num_cycles=0.0,
                periodic_fidelity=0.0,
                harmonic_periods=[],
                confidence=0.0,
                valid=False,
                diagnostics={"reason": "insufficient_length", "length": n},
            )

        # Optional continuous quantization
        if quantize_continuous and all(isinstance(x, (int, float, np.floating, np.integer)) for x in seq_list):
            arr = np.array(seq_list, dtype=np.float64)
            _, state_seq = estimate_discrete_states(arr)
            search_seq = list(state_seq)
        else:
            search_seq = seq_list

        is_p, p_len, motif_indices, fidelity, harmonics = detect_symbolic_periodicity(
            sequence=search_seq,
            max_period=max_period,
            tolerance=tolerance,
        )

        if is_p and p_len is not None and p_len > 0:
            motif = seq_list[:p_len]
            num_cycles = float(n) / float(p_len)
            confidence = min(1.0, max(0.2, (num_cycles * fidelity) / 3.0))
        else:
            motif = []
            num_cycles = 0.0
            confidence = 0.1

        diagnostics = {
            "sequence_length": n,
            "fidelity": fidelity,
            "harmonics": harmonics,
        }

        return SequencePeriodicityResult(
            is_periodic=is_p,
            period_length=p_len,
            repeating_motif=motif,
            num_cycles=num_cycles,
            periodic_fidelity=fidelity,
            harmonic_periods=harmonics,
            confidence=confidence,
            valid=True,
            diagnostics=diagnostics,
        )


SequencePatternMetricFactory.register("periodicity", SequencePeriodicity)


def analyze_sequence_periodicity(
    sequence: Any,
    max_period: Optional[int] = None,
    tolerance: float = 0.0,
    quantize_continuous: bool = False,
    **kwargs: Any,
) -> SequencePeriodicityResult:
    """
    Functional wrapper to analyze sequence periodicity and repeating motifs.
    """
    metric = SequencePeriodicity()
    return metric.compute(
        sequence=sequence,
        max_period=max_period,
        tolerance=tolerance,
        quantize_continuous=quantize_continuous,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  SequencePeriodicity: Demonstration")
    print("=" * 60)
    seq = np.array([10.0, 10.2, 9.9, 10.1, 10.0, 10.05, 9.95])
    metric = SequencePeriodicity()
    result = metric.compute(seq)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
