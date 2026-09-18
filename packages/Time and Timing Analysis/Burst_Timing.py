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

from .models import BurstTimingResult
from ._common import (
    ensure_1d_float_array,
    ensure_1d_int_array,
    robust_nan_statistics,
)

UNASSIGNED_BURST_LABELS = {-1, "-1", "unassigned", "noise", "none", None}


def _is_unassigned(label: Any) -> bool:
    """Checks if a burst label represents an unassigned/noise pulse."""
    if label in UNASSIGNED_BURST_LABELS:
        return True
    if isinstance(label, (int, np.integer)) and int(label) < 0:
        return True
    if str(label).strip().lower() in {"-1", "unassigned", "noise", "none"}:
        return True
    return False


class BurstTiming:
    """
    Burst Timing analysis calculator.

    Aggregates pulse TOAs by their upstream burst labels/IDs to calculate
    burst start times, end times, burst durations, inter-burst intervals (IBI),
    and burst repetition intervals (BRI).
    Excludes unassigned pulses (label -1, 'unassigned', 'noise') and reports their count.
    Handles NaN/missing durations gracefully without poisoning the entire burst.
    Supports integer nanosecond arithmetic for absolute timing.
    Does NOT perform burst clustering.
    """

    def analyze(
        self,
        toas: np.ndarray | Sequence[float] | None = None,
        burst_labels: np.ndarray | Sequence[Any] | None = None,
        pulse_durations: np.ndarray | Sequence[float] | None = None,
        toas_ns: np.ndarray | Sequence[int] | None = None,
        timing_reference: str | None = None,
    ) -> BurstTimingResult:
        """
        Executes burst timing aggregation on labeled pulse sequences.
        """
        if burst_labels is None:
            raise ValueError("burst_labels must be provided for burst timing analysis")

        labels = np.asarray(burst_labels)
        n_pulses = len(labels)

        if toas_ns is not None:
            arr_ns = ensure_1d_int_array(toas_ns)
            if len(arr_ns) != n_pulses:
                raise ValueError(f"toas_ns length ({len(arr_ns)}) != burst_labels length ({n_pulses})")
            has_ns = True
            arr_sec = None
        elif toas is not None:
            arr_sec = ensure_1d_float_array(toas, name="toas")
            if len(arr_sec) != n_pulses:
                raise ValueError(f"toas length ({len(arr_sec)}) != burst_labels length ({n_pulses})")
            arr_ns = None
            has_ns = False
        else:
            raise ValueError("Either toas or toas_ns must be provided")

        if n_pulses == 0:
            return BurstTimingResult()

        if pulse_durations is not None:
            dur_arr = ensure_1d_float_array(pulse_durations, name="pulse_durations")
            if len(dur_arr) != n_pulses:
                raise ValueError(f"pulse_durations length ({len(dur_arr)}) != burst_labels length ({n_pulses})")
            # Reject +/- Inf and negative durations
            if len(dur_arr) > 0:
                if np.any(np.isinf(dur_arr)):
                    raise ValueError("pulse_durations cannot contain infinite values (+Inf / -Inf)")
                finite_d = dur_arr[np.isfinite(dur_arr)]
                if len(finite_d) > 0 and np.any(finite_d < 0):
                    bad_d = float(finite_d[finite_d < 0][0])
                    raise ValueError(f"pulse_durations cannot contain negative values (found {bad_d})")
        else:
            dur_arr = np.zeros(n_pulses, dtype=np.float64)

        # Count unassigned pulses and exclude them
        unassigned_count = 0
        assigned_indices: list[int] = []

        for idx, lbl in enumerate(labels):
            if _is_unassigned(lbl):
                unassigned_count += 1
            else:
                assigned_indices.append(idx)

        if not assigned_indices:
            return BurstTimingResult(unassigned_pulse_count=unassigned_count)

        # Identify unique ordered burst groups
        seen_labels: set[Any] = set()
        ordered_unique_labels: list[Any] = []
        for idx in assigned_indices:
            lbl = labels[idx]
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                ordered_unique_labels.append(lbl)

        unique_bursts: list[Any] = []
        burst_starts: list[float] = []
        burst_ends: list[float] = []
        burst_durations: list[float] = []
        burst_counts: list[int] = []
        burst_starts_ns: list[int] = []
        burst_ends_ns: list[int] = []
        burst_durations_ns: list[int] = []

        for b_lbl in ordered_unique_labels:
            mask = [(labels[i] == b_lbl) for i in range(n_pulses)]
            indices_in_burst = [i for i in range(n_pulses) if mask[i]]
            if not indices_in_burst:
                continue

            if has_ns and arr_ns is not None:
                b_ns_vals = [arr_ns[i] for i in indices_in_burst]
                b_dur_vals = [dur_arr[i] for i in indices_in_burst]

                # Start of burst is min TOA in integer ns
                b_s_ns = int(min(b_ns_vals))

                # End of burst is max(TOA_i + pulse_width_i) across all pulses in burst
                b_end_candidates_ns: list[int] = []
                for t_ns, d_s in zip(b_ns_vals, b_dur_vals):
                    dur_ns = int(round(float(d_s) * 1e9)) if (np.isfinite(d_s) and d_s >= 0) else 0
                    b_end_candidates_ns.append(t_ns + dur_ns)

                b_e_ns = int(max(b_end_candidates_ns))
                b_dur_ns = b_e_ns - b_s_ns

                unique_bursts.append(b_lbl)
                burst_starts_ns.append(b_s_ns)
                burst_ends_ns.append(b_e_ns)
                burst_durations_ns.append(b_dur_ns)
                burst_starts.append(float(b_s_ns) / 1e9)
                burst_ends.append(float(b_e_ns) / 1e9)
                burst_durations.append(float(b_dur_ns) / 1e9)
                burst_counts.append(len(b_ns_vals))

            elif arr_sec is not None:
                b_toas = arr_sec[mask]
                b_durs = dur_arr[mask]

                valid_mask = np.isfinite(b_toas)
                if not np.any(valid_mask):
                    continue

                v_toas = b_toas[valid_mask]
                v_durs = b_durs[valid_mask]

                # Start of burst is min valid TOA
                b_start = float(np.min(v_toas))

                # End of burst is max(TOA_i + pulse_width_i) across all pulses in burst
                b_end_candidates: list[float] = []
                for t, d in zip(v_toas, v_durs):
                    dur_s = float(d) if (np.isfinite(d) and d >= 0) else 0.0
                    b_end_candidates.append(t + dur_s)

                b_end = float(np.max(b_end_candidates))
                b_dur = b_end - b_start

                unique_bursts.append(b_lbl)
                burst_starts.append(b_start)
                burst_ends.append(b_end)
                burst_durations.append(b_dur)
                burst_counts.append(int(len(v_toas)))

        n_bursts = len(unique_bursts)
        if n_bursts == 0:
            return BurstTimingResult(unassigned_pulse_count=unassigned_count)

        if has_ns:
            # Sort bursts using integer nanosecond start timestamps
            sort_order = sorted(range(n_bursts), key=lambda i: burst_starts_ns[i])
            burst_starts_ns = [burst_starts_ns[i] for i in sort_order]
            burst_ends_ns = [burst_ends_ns[i] for i in sort_order]
            burst_durations_ns = [burst_durations_ns[i] for i in sort_order]
            unique_bursts = [unique_bursts[i] for i in sort_order]
            counts_arr = np.array([burst_counts[i] for i in sort_order], dtype=np.int64)

            # Integer nanosecond interval differences
            if n_bursts > 1:
                bri_ns = [burst_starts_ns[i + 1] - burst_starts_ns[i] for i in range(n_bursts - 1)]
                ibi_ns = [burst_starts_ns[i + 1] - burst_ends_ns[i] for i in range(n_bursts - 1)]
                bri_arr = np.array([float(d) / 1e9 for d in bri_ns], dtype=np.float64)
                ibi_arr = np.array([float(d) / 1e9 for d in ibi_ns], dtype=np.float64)
            else:
                bri_arr = np.array([], dtype=np.float64)
                ibi_arr = np.array([], dtype=np.float64)

            durations_arr = np.array([float(d) / 1e9 for d in burst_durations_ns], dtype=np.float64)
            starts_arr = np.array([float(t) / 1e9 for t in burst_starts_ns], dtype=np.float64)
            ends_arr = np.array([float(t) / 1e9 for t in burst_ends_ns], dtype=np.float64)

        else:
            starts_arr = np.array(burst_starts, dtype=np.float64)
            ends_arr = np.array(burst_ends, dtype=np.float64)
            durations_arr = np.array(burst_durations, dtype=np.float64)
            counts_arr = np.array(burst_counts, dtype=np.int64)

            sort_order_np = np.argsort(starts_arr)
            starts_arr = starts_arr[sort_order_np]
            ends_arr = ends_arr[sort_order_np]
            durations_arr = durations_arr[sort_order_np]
            counts_arr = counts_arr[sort_order_np]
            unique_bursts = [unique_bursts[i] for i in sort_order_np]

            if n_bursts > 1:
                ibi_arr = starts_arr[1:] - ends_arr[:-1]
                bri_arr = starts_arr[1:] - starts_arr[:-1]
            else:
                ibi_arr = np.array([], dtype=np.float64)
                bri_arr = np.array([], dtype=np.float64)

        dur_stats = robust_nan_statistics(durations_arr)
        bri_stats = robust_nan_statistics(bri_arr)
        ibi_stats = robust_nan_statistics(ibi_arr)

        ref = timing_reference or ("externally_absolute" if has_ns else "capture_relative")

        return BurstTimingResult(
            num_bursts=n_bursts,
            burst_ids=unique_bursts,
            burst_start_toas=starts_arr,
            burst_end_toas=ends_arr,
            burst_start_toas_ns=burst_starts_ns if has_ns else None,
            burst_end_toas_ns=burst_ends_ns if has_ns else None,
            timing_reference=ref,
            burst_durations_sec=durations_arr,
            inter_burst_intervals_sec=ibi_arr,
            burst_repetition_intervals_sec=bri_arr,
            burst_pulse_counts=counts_arr,
            unassigned_pulse_count=unassigned_count,
            mean_burst_duration_sec=dur_stats["mean"],
            mean_bri_sec=bri_stats["mean"],
            mean_ibi_sec=ibi_stats["mean"],
            mean_pulses_per_burst=float(np.mean(counts_arr)) if len(counts_arr) > 0 else None,
        )


def compute_burst_timing(
    toas: np.ndarray | Sequence[float] | None = None,
    burst_labels: np.ndarray | Sequence[Any] | None = None,
    pulse_durations: np.ndarray | Sequence[float] | None = None,
    toas_ns: np.ndarray | Sequence[int] | None = None,
    timing_reference: str | None = None,
) -> BurstTimingResult:
    """
    Functional API to compute Burst Timing characteristics.

    Parameters
    ----------
    toas : Sequence[float] | np.ndarray | None
        Pulse TOAs in seconds.
    burst_labels : Sequence[Any] | np.ndarray | None
        Upstream burst assignment labels or IDs corresponding to each pulse.
    pulse_durations : Sequence[float] | np.ndarray | None
        Optional pulse durations in seconds.
    toas_ns : Sequence[int] | np.ndarray | None
        Optional pulse TOAs in integer nanoseconds.
    timing_reference : str | None
        Explicit timing reference type ('hardware_absolute', 'capture_relative', 'externally_absolute').

    Returns
    -------
    BurstTimingResult
        Burst starts, ends, durations, IBI, BRI, and counts.
    """
    calc = BurstTiming()
    return calc.analyze(
        toas=toas,
        burst_labels=burst_labels,
        pulse_durations=pulse_durations,
        toas_ns=toas_ns,
        timing_reference=timing_reference,
    )


__all__ = ["BurstTiming", "compute_burst_timing"]


def main() -> None:
    print("=" * 60)
    print("  Burst Timing Demonstration")
    print("=" * 60)
    toas = [0.001, 0.002, 0.003, 0.050, 0.051, 0.052, 0.099]
    labels = ["B1", "B1", "B1", "B2", "B2", "B2", -1]
    res = compute_burst_timing(toas=toas, burst_labels=labels)
    print(f"Num bursts: {res.num_bursts}, Unassigned: {res.unassigned_pulse_count}")
    print(f"BRI: {res.burst_repetition_intervals_sec}")
    print("=" * 60)


if __name__ == "__main__":
    main()
