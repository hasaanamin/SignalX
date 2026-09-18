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
from enum import IntEnum
import numpy as np
from .base import BasePulseDetectionMetric, PulseDetectionFactory
from ._common import _as_signal_array


class PulseFSMState(IntEnum):
    IDLE = 0
    RISING_CANDIDATE = 1
    IN_PULSE = 2
    FALLING_CANDIDATE = 3
    REFRACTORY = 4


@PulseDetectionFactory.register(["pulse_detection", "fsm_detection", "detect_pulse_regions"])
class PulseDetectionMetric(BasePulseDetectionMetric):
    """
    Stage 3: 5-State Finite-State Machine (FSM) Pulse Detector.
    Implements N-of-M temporal confirmation, hysteresis thresholds, noise spike rejection,
    and captures the complete pulse interval from noise-floor emergence to termination.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reset()

    def __del__(self) -> None:
        super().__del__()

    def reset(self) -> None:
        """Resets FSM state machine for a new independent stream."""
        self._state = PulseFSMState.IDLE
        self._sample_offset = 0
        self._candidate_start_idx = 0
        self._rising_window_samples = 0
        self._rising_high_hits = 0
        self._falling_count = 0
        self._refractory_count = 0
        self._current_peak_val = -np.inf
        self._current_peak_idx = 0

    @property
    def name(self) -> str:
        return "pulse_detection"

    def compute(
        self,
        detection_statistic_or_envelope: Any,
        high_threshold: float | np.ndarray | None = None,
        low_threshold: float | np.ndarray | None = None,
        confirmation_window_samples: int = 4,
        required_high_hits: int = 2,
        min_rising_samples: int | None = None,
        termination_samples: int = 3,
        refractory_samples: int = 2,
        maintain_streaming_state: bool = False,
        **kwargs: Any
    ) -> list[dict[str, int | float]]:
        """
        Executes causal FSM pulse detection across the detection statistic.

        Parameters:
            detection_statistic_or_envelope: 1-D detection statistic array.
            high_threshold: Upper trigger threshold (1-D array or scalar).
            low_threshold: Lower continuation threshold (1-D array or scalar).
            confirmation_window_samples: Window length M for confirmation.
            required_high_hits: Number of High hits required.
            min_rising_samples: Backward compatibility alias for required_high_hits.
            termination_samples: Consecutive samples below Low threshold required to confirm pulse end.
            refractory_samples: Dead-time samples after pulse termination to prevent ringing.
            maintain_streaming_state: If False (default for frame testing), resets FSM state per call.
        """
        if not maintain_streaming_state:
            self.reset()

        # Parse input
        if isinstance(detection_statistic_or_envelope, dict) and "detection_statistic" in detection_statistic_or_envelope:
            det_stat = np.asarray(detection_statistic_or_envelope["detection_statistic"], dtype=np.float64)
        else:
            arr = _as_signal_array(detection_statistic_or_envelope)
            det_stat = np.abs(arr).astype(np.float64) if np.iscomplexobj(arr) else arr.astype(np.float64)

        n = det_stat.size
        if n == 0:
            return []

        # Auto-compute thresholds if not provided
        if high_threshold is None or low_threshold is None:
            from . import ThresholdDetectionMetric
            th_res = ThresholdDetectionMetric().compute(det_stat, **kwargs)
            if high_threshold is None:
                high_threshold = th_res["high_threshold"]
            if low_threshold is None:
                low_threshold = th_res["low_threshold"]

        th_hi = np.broadcast_to(np.asarray(high_threshold, dtype=np.float64), (n,))
        th_lo = np.broadcast_to(np.asarray(low_threshold, dtype=np.float64), (n,))

        candidates: list[dict[str, int | float]] = []

        N_hits = max(1, int(min_rising_samples if min_rising_samples is not None else required_high_hits))
        M_confirm = max(N_hits, int(confirmation_window_samples))
        N_term = max(1, int(termination_samples))
        N_refr = max(0, int(refractory_samples))

        for i in range(n):
            abs_i = self._sample_offset + i
            val = float(det_stat[i])
            hi = float(th_hi[i])
            lo = float(th_lo[i])

            if self._state == PulseFSMState.IDLE:
                if val >= hi:
                    # Walk backward to find the exact point where signal rose above low threshold
                    s_coarse = i
                    while s_coarse > 0 and det_stat[s_coarse - 1] >= th_lo[s_coarse - 1]:
                        s_coarse -= 1

                    self._candidate_start_idx = self._sample_offset + s_coarse
                    self._rising_window_samples = 1
                    self._rising_high_hits = 1
                    self._current_peak_val = val
                    self._current_peak_idx = abs_i
                    if self._rising_high_hits >= N_hits:
                        self._state = PulseFSMState.IN_PULSE
                        self._falling_count = 0
                    else:
                        self._state = PulseFSMState.RISING_CANDIDATE
                continue

            elif self._state == PulseFSMState.RISING_CANDIDATE:
                self._rising_window_samples += 1
                if val >= hi:
                    self._rising_high_hits += 1
                if val > self._current_peak_val:
                    self._current_peak_val = val
                    self._current_peak_idx = abs_i

                if self._rising_high_hits >= N_hits:
                    self._state = PulseFSMState.IN_PULSE
                    self._falling_count = 0
                elif self._rising_window_samples >= M_confirm:
                    if self._rising_high_hits < N_hits:
                        self._state = PulseFSMState.IDLE
                    else:
                        self._state = PulseFSMState.IN_PULSE
                        self._falling_count = 0
                continue

            elif self._state == PulseFSMState.IN_PULSE:
                if val > self._current_peak_val:
                    self._current_peak_val = val
                    self._current_peak_idx = abs_i

                if val >= lo:
                    self._falling_count = 0
                else:
                    self._falling_count = 1
                    self._state = PulseFSMState.FALLING_CANDIDATE
                continue

            elif self._state == PulseFSMState.FALLING_CANDIDATE:
                if val < lo:
                    self._falling_count += 1
                    if self._falling_count >= N_term:
                        pulse_end = max(self._candidate_start_idx + 1, abs_i - self._falling_count + 1)
                        candidates.append({
                            "start": self._candidate_start_idx,
                            "end": pulse_end,
                            "peak_idx": self._current_peak_idx,
                            "peak_val": self._current_peak_val,
                        })
                        if N_refr > 0:
                            self._state = PulseFSMState.REFRACTORY
                            self._refractory_count = 0
                        else:
                            self._state = PulseFSMState.IDLE
                else:
                    self._state = PulseFSMState.IN_PULSE
                    self._falling_count = 0
                    if val > self._current_peak_val:
                        self._current_peak_val = val
                        self._current_peak_idx = abs_i
                continue

            elif self._state == PulseFSMState.REFRACTORY:
                self._refractory_count += 1
                if self._refractory_count >= N_refr:
                    self._state = PulseFSMState.IDLE
                continue

        # Handle unclosed candidate at end of frame
        if not maintain_streaming_state:
            if self._state in (PulseFSMState.IN_PULSE, PulseFSMState.FALLING_CANDIDATE):
                candidates.append({
                    "start": self._candidate_start_idx,
                    "end": self._sample_offset + n - 1,
                    "peak_idx": self._current_peak_idx,
                    "peak_val": self._current_peak_val,
                })

        self._sample_offset += n
        return candidates


_instance = PulseDetectionMetric()
pulse_detection = _instance.compute
detect_pulse_regions = _instance.compute

__all__ = ["PulseFSMState", "PulseDetectionMetric", "pulse_detection", "detect_pulse_regions"]


def main() -> None:
    print("=" * 60)
    print("  PulseDetectionMetric: Demonstration")
    print("=" * 60)
    fs = 1e6
    t = np.arange(1024) / fs
    sample_iq = np.zeros(1024, dtype=np.complex128)
    sample_iq[200:400] = np.exp(1j * 2 * np.pi * 50e3 * t[200:400])
    metric = PulseDetectionMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
