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

from typing import Any
import numpy as np

from .base import BaseClusteringMetric, ClusteringMetricFactory, IQEvent, PulseBurst, BurstGroupResult
from ._common import (
    FEATURE_NAMES,
    extract_events_from_input,
)


@ClusteringMetricFactory.register([
    "burst_grouping",
    "burst_group",
    "burst_segmentation",
    "burstgrouping",
    "dwell_grouping",
    "packet_grouping",
])
class BurstGrouping(BaseClusteringMetric):
    """
    Pulse Burst & Dwell Temporal Grouping Processor.

    Segments continuous or interleaved pulse streams into discrete transmission bursts,
    radar dwells, frequency hops, or packet transmissions based on inter-pulse arrival gaps (TOA).
    Computes burst duration, intra-burst PRI, Burst Repetition Interval (BRI), and burst duty cycle.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        burst_gap_sec: float | None = None,
        burst_gap_multiplier: float = 3.0,
        min_pulses_per_burst: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.burst_gap_sec = float(burst_gap_sec) if burst_gap_sec is not None else None
        if self.burst_gap_sec is not None and self.burst_gap_sec <= 0:
            raise ValueError(f"burst_gap_sec must be > 0 when specified, got {burst_gap_sec}")

        self.burst_gap_multiplier = float(burst_gap_multiplier)
        if self.burst_gap_multiplier <= 0 or not np.isfinite(self.burst_gap_multiplier):
            raise ValueError(f"burst_gap_multiplier must be a positive finite float, got {burst_gap_multiplier}")

        self.min_pulses_per_burst = int(min_pulses_per_burst)
        if self.min_pulses_per_burst < 1:
            raise ValueError(f"min_pulses_per_burst must be >= 1, got {min_pulses_per_burst}")

    @property
    def name(self) -> str:
        return "burst_grouping"

    def compute(self, data: Any, **kwargs: Any) -> BurstGroupResult:
        """
        Delineates pulse bursts from input pulse stream.

        Parameters
        ----------
        data : complex ndarray, list of PulseCandidate, list of IQEvent, dict, or 2D feature matrix.

        Returns
        -------
        BurstGroupResult
        """
        sample_rate = float(kwargs.get("sample_rate", self.sample_rate))
        gap_sec = kwargs.get("burst_gap_sec", self.burst_gap_sec)
        gap_mult = float(kwargs.get("burst_gap_multiplier", self.burst_gap_multiplier))
        min_pulses = int(kwargs.get("min_pulses_per_burst", self.min_pulses_per_burst))

        events, _, _ = extract_events_from_input(
            data=data,
            sample_rate=sample_rate,
            feature_names=FEATURE_NAMES,
            **kwargs,
        )

        n_events = len(events)
        if n_events == 0:
            return BurstGroupResult(
                events=[],
                burst_labels=np.empty(0, dtype=int),
                num_bursts=0,
                bursts=[],
                inter_burst_gap_threshold_sec=0.0,
                sample_rate=sample_rate,
            )

        # Check for real TOA values in temporal burst grouping
        for ev in events:
            if getattr(ev, "is_synthetic_time", False):
                raise ValueError("BurstGrouping requires real TOA values.")

        # Sort pulses by Time of Arrival (TOA) while tracking original indices
        sorted_indices = sorted(range(n_events), key=lambda i: events[i].start_time_sec)
        sorted_toas = np.array([events[i].start_time_sec for i in sorted_indices], dtype=np.float64)

        # Calculate inter-pulse arrival times (delta TOA)
        if n_events > 1:
            delta_toas = np.diff(sorted_toas)
            pos_deltas = delta_toas[delta_toas > 0]
            if gap_sec is None or gap_sec <= 0:
                if len(pos_deltas) > 0:
                    med_gap = float(np.median(pos_deltas))
                    effective_gap_threshold = max(gap_mult * med_gap, 10.0 / sample_rate)
                else:
                    effective_gap_threshold = 1.0 / sample_rate
            else:
                effective_gap_threshold = float(gap_sec)
        else:
            effective_gap_threshold = float(gap_sec) if gap_sec is not None else 1.0 / sample_rate

        # Group pulses into bursts
        raw_bursts: list[list[int]] = []
        current_burst: list[int] = [sorted_indices[0]]

        for k in range(1, n_events):
            idx = sorted_indices[k]
            prev_idx = sorted_indices[k - 1]
            dt = events[idx].start_time_sec - events[prev_idx].start_time_sec

            if dt > effective_gap_threshold:
                raw_bursts.append(current_burst)
                current_burst = [idx]
            else:
                current_burst.append(idx)

        if len(current_burst) > 0:
            raw_bursts.append(current_burst)

        # Filter and construct PulseBurst objects
        bursts: list[PulseBurst] = []
        burst_labels = np.full(n_events, -1, dtype=int)
        burst_id_counter = 0

        for b_indices in raw_bursts:
            if len(b_indices) < min_pulses:
                continue

            b_events = [events[i] for i in b_indices]
            b_toas = [ev.start_time_sec for ev in b_events]
            b_durations = [ev.duration_sec for ev in b_events]
            b_freqs = [(ev.carrier_freq_hz if ev.carrier_freq_hz is not None else ev.spectral_centroid_hz) for ev in b_events]
            b_powers = [ev.mean_power_db for ev in b_events]

            t_start = float(min(b_toas))
            t_end = float(max(t + d for t, d in zip(b_toas, b_durations)))
            dur = max(t_end - t_start, float(max(b_durations)))

            if len(b_toas) > 1:
                b_deltas = np.diff(np.sort(b_toas))
                pri_mean = float(np.mean(b_deltas))
                pri_std = float(np.std(b_deltas))
            else:
                pri_mean = 0.0
                pri_std = 0.0

            total_on_time = float(sum(b_durations))
            duty_cycle = float(np.clip(total_on_time / max(dur, 1e-15), 0.0, 1.0))

            burst_obj = PulseBurst(
                burst_id=burst_id_counter,
                pulse_indices=b_indices,
                pulse_count=len(b_indices),
                start_time_sec=t_start,
                end_time_sec=t_end,
                duration_sec=dur,
                bri_sec=0.0,  # updated in next pass
                intra_burst_pri_mean_sec=pri_mean,
                intra_burst_pri_std_sec=pri_std,
                duty_cycle=duty_cycle,
                mean_frequency_hz=float(np.mean(b_freqs)),
                mean_power_db=float(np.mean(b_powers)),
            )

            for idx in b_indices:
                burst_labels[idx] = burst_id_counter

            bursts.append(burst_obj)
            burst_id_counter += 1

        # Calculate BRI (Burst Repetition Interval)
        for b_i in range(len(bursts) - 1):
            bursts[b_i].bri_sec = float(bursts[b_i + 1].start_time_sec - bursts[b_i].start_time_sec)

        return BurstGroupResult(
            events=events,
            burst_labels=burst_labels,
            num_bursts=len(bursts),
            bursts=bursts,
            inter_burst_gap_threshold_sec=effective_gap_threshold,
            sample_rate=sample_rate,
        )


def compute_burst_grouping(
    data: Any,
    sample_rate: float = 1.0,
    burst_gap_sec: float | None = None,
    burst_gap_multiplier: float = 3.0,
    min_pulses_per_burst: int = 1,
    **kwargs: Any,
) -> BurstGroupResult:
    """Convenience functional API for pulse burst grouping."""
    processor = BurstGrouping(
        sample_rate=sample_rate,
        burst_gap_sec=burst_gap_sec,
        burst_gap_multiplier=burst_gap_multiplier,
        min_pulses_per_burst=min_pulses_per_burst,
        **kwargs,
    )
    return processor.compute(data)


def main() -> None:
    print("=" * 60)
    print("  BurstGrouping: Demonstration")
    print("=" * 60)
    pulses = [
        {"toa": 0.001, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.0011, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.002, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
        {"toa": 0.0022, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
    ]
    metric = BurstGrouping()
    result = metric.compute(pulses)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
