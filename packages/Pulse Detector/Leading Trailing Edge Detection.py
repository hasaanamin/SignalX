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
from .base import BasePulseDetectorStage, PulseDetectorFactory
from ._common import _extract_sample_rate


@PulseDetectorFactory.register([
    "leading_trailing_edge_detection",
    "leading_trailing_edges",
    "edge_detection",
    "edges",
    "refine_edges",
    "edge_refinement"
])
class LeadingTrailingEdgeDetectionStage(BasePulseDetectorStage):
    """
    Stage 5: Leading / Trailing Edge Detection.
    Refines preliminary boundary crossings into true physical pulse edges using the raw power envelope derivative:
      dP/dt = F_s * (P_raw[n] - P_raw[n-1])
    Searches in a local temporal neighborhood around each preliminary boundary:
      - Leading edge: n_lead = argmax_{n in W_s} dP/dt   (Strongest positive transition / steepest rise)
      - Trailing edge: n_trail = (argmin_{n in W_e} dP/dt) - 1 (Last sample before negative fall transition)
    MATLAB Reference:
      dP = [0; diff(Praw(:)) * Fs];
      n_lead = argmax in Ws(dP);
      n_fall_transition = argmin in We(dP);
      n_trail = n_fall_transition - 1;
    """

    def __init__(self, sample_rate: float | None = None, search_window: int = 8, **kwargs: Any) -> None:
        cfg = dict(kwargs)
        if sample_rate is not None:
            cfg["sample_rate"] = sample_rate
        cfg["search_window"] = search_window
        super().__init__(**cfg)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "leading_trailing_edge_detection"

    def compute(self, data: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Refines leading and trailing edges for each preliminary boundary region.

        Parameters:
            data: 1D envelope or raw power array P_raw.
            boundaries: List of (start_idx, stop_idx) tuples from Stage 4.
            raw_power: Optional unfiltered raw power array |x|^2 (preferred for edge refinement).
            sample_rate (Fs): Sampling frequency in Hz (must be explicitly provided and > 0).
            search_window: Half-width search window around boundaries (default: 8 samples).

        Returns:
            List of dictionaries per detected region with:
              - 'n_lead': Refined leading edge sample index
              - 'n_trail': Refined trailing edge sample index (last pulse sample)
              - 'rise_slope': Maximum positive derivative dP/dt (W/s) at leading edge
              - 'fall_slope': Minimum negative derivative dP/dt (W/s) at trailing transition
              - 'rise_delta_per_sample': Delta P per sample at leading edge
              - 'fall_delta_per_sample': Delta P per sample at trailing transition
              - 'edge_refinement_valid': True if edge ordering is preserved
              - 'prelim_start': Original start index
              - 'prelim_stop': Original stop index
        """
        params = {**self._config, **kwargs}
        fs = _extract_sample_rate(params)
        boundaries = params.get("boundaries", [])

        # Prefer raw unfiltered power for high-precision edge derivative
        raw_power_in = params.get("raw_power")
        if raw_power_in is not None:
            power_arr = np.asarray(raw_power_in, dtype=np.float64).ravel()
        else:
            power_arr = np.asarray(data, dtype=np.float64).ravel()

        if power_arr.size == 0 or len(boundaries) == 0:
            return []

        search_w = int(params.get("search_window", 8))
        if search_w < 1:
            raise ValueError(f"edge_search_window must be >= 1, got {search_w}")

        n_samples = power_arr.size

        # Physical time derivative dP/dt = Fs * Delta P
        dp_raw = np.zeros(n_samples, dtype=np.float64)
        if n_samples > 1:
            dp_raw[1:] = np.diff(power_arr)
        dp_dt = dp_raw * fs

        refined_edges: list[dict[str, Any]] = []

        for b_item in boundaries:
            if not isinstance(b_item, (tuple, list)) or len(b_item) < 2:
                continue
            start_raw, stop_raw = b_item[0], b_item[1]
            if start_raw is None:
                start_idx = 0
            else:
                start_idx = int(start_raw)
            stop_idx = int(stop_raw)

            start_idx = int(max(0, min(start_idx, n_samples - 1)))
            stop_idx = int(max(0, min(stop_idx, n_samples - 1)))

            # Search window around preliminary start for steepest positive rise
            w_lead_min = max(0, start_idx - search_w)
            w_lead_max = min(n_samples, start_idx + search_w + 1)
            dp_lead_segment = dp_dt[w_lead_min:w_lead_max]

            if dp_lead_segment.size > 0:
                rel_max_lead = int(np.argmax(dp_lead_segment))
                n_lead = w_lead_min + rel_max_lead
                rise_slope = float(dp_lead_segment[rel_max_lead])
                rise_delta = float(dp_raw[n_lead])
            else:
                n_lead = start_idx
                rise_slope = float(dp_dt[n_lead]) if n_lead < n_samples else 0.0
                rise_delta = float(dp_raw[n_lead]) if n_lead < n_samples else 0.0

            # Search window around preliminary stop for steepest negative fall
            w_trail_min = max(0, stop_idx - search_w)
            w_trail_max = min(n_samples, stop_idx + search_w + 1)
            dp_trail_segment = dp_dt[w_trail_min:w_trail_max]

            if dp_trail_segment.size > 0:
                rel_min_trail = int(np.argmin(dp_trail_segment))
                fall_transition_idx = w_trail_min + rel_min_trail
                # BUG 1 Fix: Final pulse sample is one sample before negative transition
                n_trail = max(0, fall_transition_idx - 1)
                # Keep slope measurement at the transition
                fall_slope = float(dp_trail_segment[rel_min_trail])
                fall_delta = float(dp_raw[fall_transition_idx])
            else:
                n_trail = stop_idx
                fall_transition_idx = min(n_samples - 1, stop_idx + 1)
                fall_slope = float(dp_dt[fall_transition_idx]) if fall_transition_idx < n_samples else 0.0
                fall_delta = float(dp_raw[fall_transition_idx]) if fall_transition_idx < n_samples else 0.0

            # Fallback if refinement reverses edges
            if n_trail < n_lead:
                n_lead = start_idx
                n_trail = stop_idx
                rise_slope = float(dp_dt[n_lead]) if n_lead < n_samples else 0.0
                fall_idx = min(n_samples - 1, stop_idx + 1)
                fall_slope = float(dp_dt[fall_idx]) if fall_idx < n_samples else 0.0
                rise_delta = float(dp_raw[n_lead]) if n_lead < n_samples else 0.0
                fall_delta = float(dp_raw[fall_idx]) if fall_idx < n_samples else 0.0
                edge_refinement_valid = False
            else:
                edge_refinement_valid = True

            refined_edges.append({
                "n_lead": int(n_lead),
                "n_trail": int(n_trail),
                "rise_slope": rise_slope,
                "fall_slope": fall_slope,
                "rise_delta_per_sample": rise_delta,
                "fall_delta_per_sample": fall_delta,
                "edge_refinement_valid": edge_refinement_valid,
                "prelim_start": start_idx,
                "prelim_stop": stop_idx,
            })

        return refined_edges


def refine_edges(data: Any, boundaries: list[tuple[int, int]], sample_rate: float, search_window: int = 8, **kwargs: Any) -> list[dict[str, Any]]:
    """Functional wrapper for LeadingTrailingEdgeDetectionStage."""
    stage = LeadingTrailingEdgeDetectionStage(sample_rate=sample_rate, search_window=search_window, **kwargs)
    return stage.compute(data, boundaries=boundaries)


def main() -> None:
    print("=" * 60)
    print("  LeadingTrailingEdgeDetectionStage: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = LeadingTrailingEdgeDetectionStage()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
