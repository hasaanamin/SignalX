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
from typing import Any, List, Optional
import numpy as np

from .base import BaseSequencePatternMetric, SequencePatternMetricFactory
from .models import ModulationPatternResult
from ._common import (
    build_transition_matrix,
    compute_dwell_lengths,
    detect_symbolic_periodicity,
)


@SequencePatternMetricFactory.register([
    "modulation_patterns",
    "modulation_pattern",
    "modulation_pattern_analysis",
    "modulationpatterns",
])
class ModulationPatterns(BaseSequencePatternMetric):
    """
    Modulation Classification Sequence Pattern Analyzer.

    Analyzes pulse-to-pulse modulation type sequences, mode switching, dwell lengths, transition
    probabilities, and adaptive rate scheduling.

    This processor does NOT perform modulation recognition or classification. The caller must
    provide pre-classified modulation labels from the Modulation_Analysis package.
    """

    @property
    def name(self) -> str:
        return "modulation_patterns"

    def compute(
        self,
        modulation_labels: list[str] | np.ndarray,
        frequency_states_hz: Optional[List[List[float]]] = None,
        confidences: list[float] | np.ndarray | None = None,
        timestamps_sec: list[float] | np.ndarray | None = None,
        sort_by_time: bool = False,
        **kwargs: Any,
    ) -> ModulationPatternResult:
        """
        Analyzes pattern structure across an already-classified modulation label sequence.

        Parameters
        ----------
        modulation_labels : list of str or array-like
            Sequence of classification labels (e.g., ['QPSK', 'QPSK', '8PSK', ...]).
        frequency_states_hz : list of list of float, optional
            Precomputed intra-pulse frequency states per pulse.
        confidences : list of float, optional
            Corresponding classification confidence values in [0, 1].
        timestamps_sec : list of float, optional
            Corresponding pulse arrival timestamps.
        sort_by_time : bool, default=False
            Whether to sort labels chronologically by timestamps.

        Returns
        -------
        ModulationPatternResult
        """
        if modulation_labels is None:
            raise ValueError("Input modulation_labels cannot be None.")

        labels = [str(x).strip() for x in modulation_labels]
        n = len(labels)

        if confidences is not None:
            if len(confidences) != n:
                raise ValueError(f"confidences length ({len(confidences)}) must match modulation_labels length ({n}).")
            for c in confidences:
                c_val = float(c)
                if c_val < 0.0 or c_val > 1.0 or not math.isfinite(c_val):
                    raise ValueError(f"confidences elements must be finite numbers in [0, 1], got {c}")

        if n == 0:
            return ModulationPatternResult(
                pattern_type="unresolved",
                num_pulses=0,
                unique_modulations=[],
                modulation_sequence=[],
                dwell_lengths=[],
                transition_matrix=np.empty((0, 0), dtype=np.float64),
                repetition_length=None,
                consistency_score=0.0,
                confidence=0.0,
                valid=False,
                diagnostics={"reason": "Input modulation label sequence is empty."},
            )

        if timestamps_sec is not None and sort_by_time and len(timestamps_sec) == n:
            order = np.argsort(np.asarray(timestamps_sec, dtype=np.float64))
            labels = [labels[i] for i in order]
            if confidences is not None and len(confidences) == n:
                confidences = [confidences[i] for i in order]

        # Identify valid vs unknown modulation entries
        valid_mask = [x.lower() not in ("none", "nan", "unknown", "unresolved", "") for x in labels]
        valid_labels = [labels[i] for i in range(n) if valid_mask[i]]

        if len(valid_labels) < 2:
            return ModulationPatternResult(
                pattern_type="unresolved",
                num_pulses=n,
                unique_modulations=sorted(list(set(valid_labels))),
                modulation_sequence=labels,
                dwell_lengths=[],
                transition_matrix=np.empty((0, 0), dtype=np.float64),
                repetition_length=None,
                consistency_score=0.0,
                confidence=0.0,
                valid=False,
                diagnostics={"reason": "Insufficient valid modulation label entries (< 2)."},
            )

        unique_mods = sorted(list(set(valid_labels)))
        num_mods = len(unique_mods)
        mod_to_id = {mod: idx for idx, mod in enumerate(unique_mods)}

        # Encode state sequence with -1 for invalid/missing labels
        state_seq = np.array([mod_to_id[lbl] if valid_mask[i] else -1 for i, lbl in enumerate(labels)], dtype=int)

        # Transition matrix and dwells with -1 sentinels breaking adjacency
        trans_mat = build_transition_matrix(state_seq, num_mods, missing_value=-1)
        dwells = compute_dwell_lengths(state_seq, missing_value=-1)

        # Periodicity / schedule detection with -1 breaking periodicity
        is_periodic, rep_len, motif, fidelity, harmonics = detect_symbolic_periodicity(state_seq, missing_value=-1)

        # Upstream confidence incorporation
        if confidences is not None and len(confidences) == n:
            valid_confs = [float(c) for i, c in enumerate(confidences) if valid_mask[i]]
            upstream_mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
            conf_supplied = True
        else:
            upstream_mean_conf = None
            conf_supplied = False

        # Pattern classification
        pattern_type = "unresolved"
        mean_dwell = float(np.mean(dwells)) if dwells else 1.0

        if num_mods == 1:
            pattern_type = "constant_mode"
            consistency_score = 1.0
            pattern_conf = 0.95
        elif is_periodic and rep_len is not None and rep_len >= 2:
            pattern_type = "repeating_code"
            consistency_score = float(fidelity)
            pattern_conf = float(np.clip(fidelity * 0.95, 0.6, 1.0))
        elif num_mods >= 2 and len(dwells) >= 2:
            pattern_type = "switching"
            consistency_score = 0.80
            pattern_conf = 0.85
        else:
            pattern_type = "adaptive_rate"
            consistency_score = 0.50
            pattern_conf = 0.70

        # Overall combined confidence
        if upstream_mean_conf is not None:
            confidence = float(np.clip(pattern_conf * upstream_mean_conf, 0.0, 1.0))
        else:
            confidence = float(np.clip(pattern_conf, 0.0, 1.0))

        return ModulationPatternResult(
            pattern_type=pattern_type,
            num_pulses=n,
            unique_modulations=unique_mods,
            modulation_sequence=labels,
            dwell_lengths=dwells,
            transition_matrix=trans_mat,
            repetition_length=rep_len if is_periodic else None,
            consistency_score=consistency_score,
            confidence=confidence,
            valid=True,
            diagnostics={
                "is_periodic": is_periodic,
                "periodic_fidelity": fidelity,
                "repeating_motif": [unique_mods[s] for s in motif] if motif else [],
                "valid_count": len(valid_labels),
                "frequency_states_hz": frequency_states_hz,
                "confidences_supplied": conf_supplied,
                "upstream_mean_confidence": upstream_mean_conf,
                "pattern_confidence": pattern_conf,
            },
        )


def analyze_modulation_pattern(
    modulation_labels: list[str] | np.ndarray,
    frequency_states_hz: Optional[List[List[float]]] = None,
    confidences: list[float] | np.ndarray | None = None,
    timestamps_sec: list[float] | np.ndarray | None = None,
    sort_by_time: bool = False,
    **kwargs: Any,
) -> ModulationPatternResult:
    """
    Convenience function to analyze modulation patterns.

    This function does not perform modulation classification.
    The caller must provide precomputed modulation labels from the Modulation_Analysis package.
    """
    processor = ModulationPatterns(**kwargs)
    return processor.compute(
        modulation_labels=modulation_labels,
        frequency_states_hz=frequency_states_hz,
        confidences=confidences,
        timestamps_sec=timestamps_sec,
        sort_by_time=sort_by_time,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  ModulationPatterns: Demonstration")
    print("=" * 60)
    seq = np.array([10.0, 10.2, 9.9, 10.1, 10.0, 10.05, 9.95])
    metric = ModulationPatterns()
    result = metric.compute(seq)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
