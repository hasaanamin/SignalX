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
from .base import BasePulseDetectionMetric, PulseDetectionFactory
from ._common import _as_signal_array


@PulseDetectionFactory.register(["pulse_boundaries", "boundaries", "refine_boundaries"])
class PulseBoundariesMetric(BasePulseDetectionMetric):
    """
    Stage 4: Coarse Boundary Determination & Complete Pulse Window Span.
    Calculates coarse pulse boundaries encompassing the full region where the signal is elevated
    above the noise threshold, with gap tolerance for intrapulse phase-code dropouts.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "pulse_boundaries"

    def compute(
        self,
        detection_statistic_or_signal: Any,
        candidates: list[dict[str, Any]] | None = None,
        low_threshold: float | np.ndarray | None = None,
        integration_window_samples: int = 4,
        min_duration_samples: int = 6,
        max_duration_samples: int | None = None,
        min_gap_samples: int = 2,
        search_margin_samples: int = 4,
        **kwargs: Any
    ) -> list[dict[str, int | float]]:
        """
        Determines coarse pulse boundaries in the detection statistic domain.

        Parameters:
            detection_statistic_or_signal: 1-D detection statistic array.
            candidates: Candidate regions from Stage 3 PulseDetectionMetric.
            low_threshold: Low continuation threshold array or scalar.
            integration_window_samples: Causal integration delay M (default 4).
            min_duration_samples: Discard pulses shorter than this limit (default 6).
            max_duration_samples: Discard pulses longer than this limit.
            min_gap_samples: Merge candidate pulses separated by gaps smaller than this threshold (default 2).
            search_margin_samples: Extra margin around coarse boundaries (default 4).
        """
        # Parse detection statistic input
        if isinstance(detection_statistic_or_signal, dict) and "detection_statistic" in detection_statistic_or_signal:
            det_stat = np.asarray(detection_statistic_or_signal["detection_statistic"], dtype=np.float64)
            M = int(detection_statistic_or_signal.get("integration_window_samples", integration_window_samples))
        else:
            arr = _as_signal_array(detection_statistic_or_signal)
            det_stat = np.abs(arr).astype(np.float64) if np.iscomplexobj(arr) else arr.astype(np.float64)
            M = int(integration_window_samples)

        n = det_stat.size
        if n == 0:
            return []

        # Auto-detect candidates if none provided
        if candidates is None:
            from . import PulseDetectionMetric, ThresholdDetectionMetric
            th_res = ThresholdDetectionMetric().compute(det_stat, **kwargs)
            lo = th_res["low_threshold"]
            hi = th_res["high_threshold"]
            candidates = PulseDetectionMetric().compute(det_stat, high_threshold=hi, low_threshold=lo, **kwargs)
            if low_threshold is None:
                low_threshold = lo

        if not candidates:
            return []

        if low_threshold is None:
            from . import ThresholdDetectionMetric
            low_threshold = ThresholdDetectionMetric().compute(det_stat, **kwargs)["low_threshold"]

        th_lo = np.broadcast_to(np.asarray(low_threshold, dtype=np.float64), (n,))

        # 1. Walk backward/forward to local low-threshold crossings
        raw_bounds: list[dict[str, int | float]] = []
        for cand in candidates:
            s = int(cand["start"])
            e = int(cand["end"])

            while s > 0 and det_stat[s - 1] >= th_lo[s - 1]:
                s -= 1
            while e < n - 1 and det_stat[e + 1] >= th_lo[e + 1]:
                e += 1

            raw_bounds.append({
                "start_idx": max(0, s),
                "end_idx": min(n - 1, e),
            })

        if not raw_bounds:
            return []

        # 2. Merge neighboring pulses separated by short gaps (<= min_gap_samples)
        merged_bounds: list[dict[str, int | float]] = []
        curr_start = int(raw_bounds[0]["start_idx"])
        curr_end = int(raw_bounds[0]["end_idx"])

        for i in range(1, len(raw_bounds)):
            nxt_start = int(raw_bounds[i]["start_idx"])
            nxt_end = int(raw_bounds[i]["end_idx"])

            gap = nxt_start - curr_end
            if gap <= min_gap_samples:
                curr_end = max(curr_end, nxt_end)
            else:
                merged_bounds.append({"start_idx": curr_start, "end_idx": curr_end})
                curr_start = nxt_start
                curr_end = nxt_end

        merged_bounds.append({"start_idx": curr_start, "end_idx": curr_end})

        # 3. Apply duration constraints & calculate search windows
        final_bounds: list[dict[str, int | float]] = []
        for b in merged_bounds:
            s = int(b["start_idx"])
            e = int(b["end_idx"])
            dur = e - s + 1

            if dur < min_duration_samples:
                continue
            if max_duration_samples is not None and dur > max_duration_samples:
                continue

            peak_offset = int(np.argmax(det_stat[s:e + 1]))
            peak_idx = s + peak_offset
            peak_val = float(det_stat[peak_idx])

            search_start = max(0, s - M - search_margin_samples)
            search_end = min(n - 1, e + search_margin_samples)

            final_bounds.append({
                "start_idx": s,
                "end_idx": e,
                "search_start": search_start,
                "search_end": search_end,
                "duration_samples": dur,
                "peak_idx": peak_idx,
                "peak_val": peak_val,
                "integration_delay_samples": M,
            })

        return final_bounds


_instance = PulseBoundariesMetric()
pulse_boundaries = _instance.compute
refine_boundaries = _instance.compute

__all__ = ["PulseBoundariesMetric", "pulse_boundaries", "refine_boundaries"]


def main() -> None:
    print("=" * 60)
    print("  PulseBoundariesMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PulseBoundariesMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
