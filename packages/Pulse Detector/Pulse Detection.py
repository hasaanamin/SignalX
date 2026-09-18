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


@PulseDetectorFactory.register(["pulse_detection", "pulse_state", "state_machine", "detect_pulse_state", "pulse_mask"])
class PulseDetectionStage(BasePulseDetectorStage):
    """
    Stage 3: Pulse Detection (Hysteresis State Machine with Persistence/Debouncing).
    Tracks pulse activity using dual hysteresis thresholds with multi-sample confirmation:
      IDLE -> PULSE: Requires envelope[n] > T_H[n] for start_confirm_samples (default: 2) consecutive samples.
      PULSE -> IDLE: Requires envelope[n] < T_L[n] for end_confirm_samples (default: 5) consecutive samples.

    Tolerates brief power dips inside a real pulse so that a single physical pulse is not fragmented into small candidates,
    and prevents isolated 1-sample noise spikes from triggering false pulse entries.

    Maintains persistent hysteresis state (in_pulse, start_counter, end_counter) across streaming chunks.
    """

    def __init__(self, start_confirm_samples: int = 2, end_confirm_samples: int = 5, **kwargs: Any) -> None:
        cfg = dict(kwargs)
        cfg["start_confirm_samples"] = start_confirm_samples
        cfg["end_confirm_samples"] = end_confirm_samples
        super().__init__(**cfg)
        self._in_pulse: bool = False
        self._start_counter: int = 0
        self._end_counter: int = 0

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "pulse_detection"

    @property
    def in_pulse_state(self) -> bool:
        """Returns the current ongoing pulse state."""
        return self._in_pulse

    @property
    def start_counter(self) -> int:
        """Returns the current consecutive above-TH start confirmation counter."""
        return self._start_counter

    @property
    def end_counter(self) -> int:
        """Returns the current consecutive below-TL end confirmation counter."""
        return self._end_counter

    def reset(self) -> None:
        """Resets ongoing pulse state and confirmation counters to IDLE."""
        self._in_pulse = False
        self._start_counter = 0
        self._end_counter = 0

    def compute(self, data: Any, **kwargs: Any) -> np.ndarray | dict[str, Any]:
        """
        Executes debounced hysteresis state machine to generate binary pulse mask.

        Parameters:
            data: 1D envelope array P_env.
            th_high (TH): High threshold array or scalar.
            th_low (TL): Low threshold array or scalar.
            start_confirm_samples: Number of consecutive samples > TH required to enter pulse (default: 2).
            end_confirm_samples: Number of consecutive samples < TL required to exit pulse (default: 5).
            initial_state: Optional override for entry state (default: self._in_pulse).
            as_dict: If True, returns dictionary with statistics.

        Returns:
            1D boolean array pulse_mask (or dict if as_dict=True).
        """
        params = {**self._config, **kwargs}
        envelope = np.asarray(data, dtype=np.float64).ravel()

        start_confirm = int(params.get("start_confirm_samples", 2))
        if start_confirm < 1:
            raise ValueError(f"start_confirm_samples must be >= 1, got {start_confirm}")

        end_confirm = int(params.get("end_confirm_samples", 5))
        if end_confirm < 1:
            raise ValueError(f"end_confirm_samples must be >= 1, got {end_confirm}")

        if envelope.size == 0:
            empty = np.array([], dtype=bool)
            return {
                "pulse_mask": empty,
                "active_samples": 0,
                "end_in_pulse": self._in_pulse,
                "end_start_counter": self._start_counter,
                "end_end_counter": self._end_counter,
            } if params.get("as_dict") else empty

        th_high = params.get("th_high", params.get("TH"))
        th_low = params.get("th_low", params.get("TL"))

        if th_high is None or th_low is None:
            med = float(np.median(envelope))
            mad = float(np.median(np.abs(envelope - med)))
            sigma = 1.4826 * max(1e-12, mad)
            th_high = envelope * 0.0 + (med + 3.5 * sigma) if th_high is None else th_high
            th_low = envelope * 0.0 + (med + 1.8 * sigma) if th_low is None else th_low

        th_high_arr = np.broadcast_to(np.asarray(th_high, dtype=np.float64), envelope.shape)
        th_low_arr = np.broadcast_to(np.asarray(th_low, dtype=np.float64), envelope.shape)

        init_state = params.get("initial_state")
        in_pulse = self._in_pulse if init_state is None else bool(init_state)

        start_counter = int(params.get("initial_start_counter", self._start_counter if init_state is None else 0))
        end_counter = int(params.get("initial_end_counter", self._end_counter if init_state is None else 0))

        # Hysteresis State Machine loop with Debouncing
        n_samples = envelope.size
        pulse_mask = np.zeros(n_samples, dtype=bool)

        for n in range(n_samples):
            env_val = envelope[n]
            th_h = th_high_arr[n]
            th_l = th_low_arr[n]

            if not in_pulse:
                if env_val > th_h:
                    start_counter += 1
                    if start_counter >= start_confirm:
                        in_pulse = True
                        start_counter = 0
                        end_counter = 0
                else:
                    start_counter = 0
            else:
                if env_val < th_l:
                    end_counter += 1
                    if end_counter >= end_confirm:
                        in_pulse = False
                        end_counter = 0
                        start_counter = 0
                else:
                    end_counter = 0

            pulse_mask[n] = in_pulse

        self._in_pulse = in_pulse
        self._start_counter = start_counter
        self._end_counter = end_counter

        if params.get("as_dict", False):
            active_count = int(np.sum(pulse_mask))
            return {
                "pulse_mask": pulse_mask,
                "active_samples": active_count,
                "duty_cycle": float(active_count / n_samples) if n_samples > 0 else 0.0,
                "total_samples": n_samples,
                "end_in_pulse": in_pulse,
                "end_start_counter": start_counter,
                "end_end_counter": end_counter,
            }

        return pulse_mask


def detect_pulse_state(
    data: Any,
    th_high: Any = None,
    th_low: Any = None,
    start_confirm_samples: int = 2,
    end_confirm_samples: int = 5,
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:
    """Functional wrapper for PulseDetectionStage."""
    stage = PulseDetectionStage(
        start_confirm_samples=start_confirm_samples,
        end_confirm_samples=end_confirm_samples,
        as_dict=as_dict,
        **kwargs
    )
    return stage.compute(data, th_high=th_high, th_low=th_low)


def main() -> None:
    print("=" * 60)
    print("  PulseDetectionStage: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PulseDetectionStage()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
