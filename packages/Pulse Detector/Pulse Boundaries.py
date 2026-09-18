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


@PulseDetectorFactory.register(["pulse_boundaries", "boundaries", "preliminary_boundaries", "boundary_extraction"])
class PulseBoundariesStage(BasePulseDetectorStage):
    """
    Stage 4: Pulse Boundaries.
    Converts a binary pulse mask into preliminary contiguous active region boundaries [start_idx, stop_idx]:
      d = diff([False; pulse_mask; False])
      start_indices = find(d == 1)
      stop_indices = find(d == -1) - 1
    In streaming mode (final_chunk=False), open pulses crossing chunk boundaries are tracked without premature termination.
    If initial_in_pulse is True and mask[0] is False, records that the previous open pulse closed before sample 0.
    MATLAB Reference:
      d = diff([false; pulseMask(:); false]);
      startIdx = find(d == 1);
      stopIdx = find(d == -1) - 1;
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._open_pulse_start: int | None = None

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "pulse_boundaries"

    def reset(self) -> None:
        """Clears open pulse tracker."""
        self._open_pulse_start = None

    def compute(self, data: Any, **kwargs: Any) -> list[tuple[int | None, int]] | dict[str, Any]:
        """
        Extracts preliminary pulse start and stop index pairs from binary mask.

        Parameters:
            data: 1D boolean array pulse_mask.
            final_chunk: True if this is the final/offline chunk. If False, trailing open pulses remain open.
            initial_in_pulse: True if the previous chunk ended inside a pulse.
            as_dict: If True, returns dictionary with start/stop arrays and streaming state.

        Returns:
            List of (start_idx, stop_idx) index tuples (or dict if as_dict=True).
            For straddled pulses closing in this chunk, start_idx is None.
        """
        params = {**self._config, **kwargs}
        mask = np.asarray(data, dtype=bool).ravel()
        final_chunk = bool(params.get("final_chunk", True))
        initial_in_pulse = bool(params.get("initial_in_pulse", False))

        if mask.size == 0:
            if initial_in_pulse:
                boundaries = [(None, -1)] if final_chunk else []
                has_open = not final_chunk
            else:
                boundaries = []
                has_open = False
            return {
                "boundaries": boundaries,
                "has_open_pulse": has_open,
                "open_pulse_start": None,
            } if params.get("as_dict") else boundaries

        completed_boundaries: list[tuple[int | None, int]] = []
        n_samples = mask.size
        previous_state = initial_in_pulse
        current_start: int | None = None

        # BUG 3 Fix: Handle pulse ending immediately at chunk boundary (sample 0 already below TL)
        if initial_in_pulse and not mask[0]:
            completed_boundaries.append((None, -1))
            previous_state = False

        for n in range(n_samples):
            curr = bool(mask[n])
            if not previous_state and curr:
                # 0 -> 1 transition (new pulse start)
                current_start = n
                previous_state = True
            elif previous_state and not curr:
                # 1 -> 0 transition (pulse stopped at sample n - 1)
                stop_idx = n - 1
                completed_boundaries.append((current_start, stop_idx))
                previous_state = False
                current_start = None

        # Chunk boundary termination or continuation
        has_open_pulse = False
        open_pulse_start: int | None = None

        if previous_state:
            if final_chunk:
                stop_idx = n_samples - 1
                completed_boundaries.append((current_start, stop_idx))
            else:
                has_open_pulse = True
                # BUG 2 Fix: If this is an ongoing pulse from previous chunk, open_pulse_start is None (continuation)
                open_pulse_start = current_start

        if params.get("as_dict", False):
            return {
                "boundaries": completed_boundaries,
                "num_preliminary_pulses": len(completed_boundaries),
                "has_open_pulse": has_open_pulse,
                "open_pulse_start": open_pulse_start,
            }

        return completed_boundaries


def find_pulse_boundaries(data: Any, final_chunk: bool = True, initial_in_pulse: bool = False, as_dict: bool = False, **kwargs: Any) -> list[tuple[int | None, int]] | dict[str, Any]:
    """Functional wrapper for PulseBoundariesStage."""
    stage = PulseBoundariesStage(as_dict=as_dict, **kwargs)
    return stage.compute(data, final_chunk=final_chunk, initial_in_pulse=initial_in_pulse)


def main() -> None:
    print("=" * 60)
    print("  PulseBoundariesStage: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PulseBoundariesStage()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
