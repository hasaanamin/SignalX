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

from .base import BaseModulationMetric, ModulationMetricFactory
from ._results import FrequencyStatesResult
from ._validated_helpers import (
    _instantaneous_frequency,
    _center_frequency_deviation,
    _estimate_frequency_states,
)


@ModulationMetricFactory.register(["frequency_states", "fsk_frequency_states", "fsk_states", "freq_states"])
class FrequencyStates(BaseModulationMetric):
    """
    Frequency States Module.
    Independently estimates discrete FSK frequency states, state occupancies,
    and state transition probabilities directly from raw complex IQ.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        k_max: int = 8,
        smoothing: str = "moving_average",
        smoothing_window: int = 5,
        frequency_reference: str = "absolute",
        threshold_db: float = -60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.k_max = int(k_max)
        self.smoothing = str(smoothing)
        self.smoothing_window = int(smoothing_window)
        self.frequency_reference = str(frequency_reference).lower().strip()
        self.threshold_db = float(threshold_db)

    @property
    def name(self) -> str:
        return "frequency_states"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> FrequencyStatesResult:
        """Executes Frequency States estimation directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        fm_info = _instantaneous_frequency(
            iq_data,
            sample_rate=fs,
            smoothing=self.smoothing,
            smoothing_window=self.smoothing_window,
            threshold_db=self.threshold_db,
        )

        f_inst = fm_info["instantaneous_frequency_hz"]
        valid_mask = fm_info["valid_mask"]

        if self.frequency_reference == "centered":
            _, f_target = _center_frequency_deviation(f_inst, valid_mask=valid_mask)
        else:
            f_target = f_inst

        state_info = _estimate_frequency_states(
            f_target,
            valid_mask=valid_mask,
            k_max=self.k_max,
            sample_rate=fs,
        )

        return FrequencyStatesResult(
            num_frequency_states=state_info["num_frequency_states"],
            state_frequencies_hz=state_info["state_frequencies_hz"],
            state_std_hz=state_info["state_std_hz"],
            occupancies=state_info["occupancies"],
            labels=state_info["labels"],
            transition_matrix=state_info["transition_matrix"],
            minimum_state_separation=state_info["minimum_state_separation"],
            confidence=state_info["confidence"],
            valid=state_info["valid"],
            reason=state_info["reason"],
        )


def compute_frequency_states(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> FrequencyStatesResult:
    """Functional convenience wrapper for FrequencyStates."""
    return FrequencyStates(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  FrequencyStates: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FrequencyStates()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
