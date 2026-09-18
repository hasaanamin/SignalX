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

from .models import PulseDensityResult
from ._common import (
    ensure_1d_float_array,
    validate_nonnegative_durations,
)


class PulseDensity:
    """
    Pulse Density and activity rate analysis calculator.

    Computes overall pulse density (pulses per second / PPS), observation duration,
    effective duty cycle, and binned instantaneous density over an observation window.
    Requires an explicit observation duration or observation window boundaries.
    Filters out-of-window pulses and reports excluded_event_count.
    """

    def analyze(
        self,
        toas: np.ndarray | Sequence[float] | None = None,
        pulse_count: int | None = None,
        observation_duration_sec: float | None = None,
        observation_start: float | None = None,
        observation_end: float | None = None,
        observation_start_sec: float | None = None,
        observation_end_sec: float | None = None,
        pulse_durations: np.ndarray | Sequence[float] | None = None,
        num_time_bins: int = 10,
    ) -> PulseDensityResult:
        """
        Executes pulse density calculation with explicit window bounding.
        """
        obs_s = observation_start if observation_start is not None else observation_start_sec
        obs_e = observation_end if observation_end is not None else observation_end_sec

        # 1. Determine genuine observation duration
        dur: float | None = None
        w_start: float | None = None
        w_end: float | None = None

        if observation_duration_sec is not None:
            d = float(observation_duration_sec)
            if not np.isfinite(d) or d <= 0:
                raise ValueError(f"observation_duration_sec must be a positive finite number (> 0), got {observation_duration_sec}")
            dur = d
            if obs_s is not None:
                w_start = float(obs_s)
                w_end = w_start + dur

        elif obs_s is not None and obs_e is not None:
            s = float(obs_s)
            e = float(obs_e)
            if not np.isfinite(s) or not np.isfinite(e):
                raise ValueError("observation_start and observation_end must be finite numbers")
            if e <= s:
                raise ValueError(f"observation_end ({e}) must be strictly greater than observation_start ({s})")
            dur = e - s
            w_start = s
            w_end = e
        else:
            raise ValueError(
                "Pulse density requires an explicit observation_duration_sec (> 0) or observation_start and observation_end"
            )


        # 2. Filter pulses and count in-window vs excluded
        toa_arr: np.ndarray | None = None
        in_window_count = 0
        excluded_count = 0
        in_window_durs: list[float] = []

        if toas is not None:
            toa_arr = ensure_1d_float_array(toas, name="toas")
            valid_mask = np.isfinite(toa_arr)
            valid_toas = toa_arr[valid_mask]

            dur_arr = ensure_1d_float_array(pulse_durations) if pulse_durations is not None else None
            if dur_arr is not None and len(dur_arr) != len(toa_arr):
                raise ValueError(f"pulse_durations length ({len(dur_arr)}) != toas length ({len(toa_arr)})")
            if dur_arr is not None:
                validate_nonnegative_durations(dur_arr, name="pulse_durations")

            if w_start is not None and w_end is not None:
                # Explicit window bounds: filter events
                in_mask = (valid_toas >= w_start) & (valid_toas <= w_end)
                in_window_toas = valid_toas[in_mask]
                in_window_count = len(in_window_toas)
                excluded_count = len(valid_toas) - in_window_count

                if dur_arr is not None:
                    v_durs = dur_arr[valid_mask]
                    in_window_durs = [float(d) for d in v_durs[in_mask] if np.isfinite(d)]
            else:
                in_window_toas = valid_toas
                in_window_count = len(valid_toas)
                excluded_count = 0
                if dur_arr is not None:
                    v_durs = dur_arr[valid_mask]
                    in_window_durs = [float(d) for d in v_durs if np.isfinite(d)]

        elif pulse_count is not None:
            cnt = int(pulse_count)
            if cnt < 0:
                raise ValueError(f"pulse_count must be non-negative (>= 0), got {cnt}")
            in_window_count = cnt
            in_window_toas = None
            excluded_count = 0
            if pulse_durations is not None:
                d_arr = ensure_1d_float_array(pulse_durations, name="pulse_durations")
                if len(d_arr) != cnt:
                    raise ValueError(f"pulse_durations length ({len(d_arr)}) != pulse_count ({cnt})")
                validate_nonnegative_durations(d_arr, name="pulse_durations")
                in_window_durs = [float(d) for d in d_arr if np.isfinite(d)]
        else:
            raise ValueError("Either toas or pulse_count must be provided")

        density_pps = float(in_window_count / dur)

        # Duty cycle calculation
        duty_cycle: float | None = None
        if in_window_durs and dur > 0:
            total_on_time = float(sum(in_window_durs))
            duty_cycle = float(np.clip(total_on_time / dur, 0.0, 1.0))

        # Instantaneous binned density
        inst_density: np.ndarray | None = None
        time_bins: np.ndarray | None = None

        if in_window_toas is not None and len(in_window_toas) > 0 and num_time_bins > 1:
            t_min = w_start if w_start is not None else float(np.min(in_window_toas))
            t_max = w_end if w_end is not None else float(np.max(in_window_toas))
            if t_max > t_min:
                bin_edges = np.linspace(t_min, t_max, num_time_bins + 1)
                bin_width = (t_max - t_min) / num_time_bins
                counts, _ = np.histogram(in_window_toas, bins=bin_edges)
                inst_density = counts.astype(np.float64) / bin_width
                time_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return PulseDensityResult(
            pulse_density_pps=density_pps,
            pulse_count=in_window_count,
            in_window_pulse_count=in_window_count,
            excluded_event_count=excluded_count,
            observation_duration_sec=dur,
            duty_cycle=duty_cycle,
            instantaneous_density_pps=inst_density,
            time_bins_sec=time_bins,
        )



def compute_pulse_density(
    toas: np.ndarray | Sequence[float] | None = None,
    pulse_count: int | None = None,
    observation_duration_sec: float | None = None,
    observation_start: float | None = None,
    observation_end: float | None = None,
    observation_start_sec: float | None = None,
    observation_end_sec: float | None = None,
    pulse_durations: np.ndarray | Sequence[float] | None = None,
    num_time_bins: int = 10,
) -> PulseDensityResult:
    """
    Functional API to compute pulse density and activity rates.

    Parameters
    ----------
    toas : Sequence[float] | np.ndarray | None
        Pulse TOAs in seconds.
    pulse_count : int | None
        Total number of detected pulses.
    observation_duration_sec : float | None
        Total observation duration in seconds (> 0).
    observation_start : float | None
        Observation window start in seconds.
    observation_end : float | None
        Observation window end in seconds (> observation_start).
    observation_start_sec : float | None
        Alias for observation_start.
    observation_end_sec : float | None
        Alias for observation_end.
    pulse_durations : Sequence[float] | np.ndarray | None
        Optional pulse durations in seconds for duty cycle estimation.
    num_time_bins : int
        Number of bins for instantaneous density histogram.

    Returns
    -------
    PulseDensityResult
        Pulse density in PPS, in-window count, excluded count, duration, and duty cycle.
    """
    calc = PulseDensity()
    return calc.analyze(
        toas=toas,
        pulse_count=pulse_count,
        observation_duration_sec=observation_duration_sec,
        observation_start=observation_start,
        observation_end=observation_end,
        observation_start_sec=observation_start_sec,
        observation_end_sec=observation_end_sec,
        pulse_durations=pulse_durations,
        num_time_bins=num_time_bins,
    )



__all__ = ["PulseDensity", "compute_pulse_density"]


def main() -> None:
    print("=" * 60)
    print("  Pulse Density Module Demonstration")
    print("=" * 60)
    toas = [0.01, 0.02, 0.03, 0.15]  # 0.15 is outside [0.0, 0.1]
    res = compute_pulse_density(toas=toas, observation_start=0.0, observation_end=0.1)
    print(f"Density: {res.pulse_density_pps:.1f} PPS, In-window: {res.in_window_pulse_count}, Excluded: {res.excluded_event_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
