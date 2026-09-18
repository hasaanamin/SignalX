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
from ._results import PhaseStatesResult
from ._constellation_helpers import (
    _remove_dc,
    _normalize_complex_rms,
    _estimate_coarse_frequency_offset,
    _apply_frequency_correction,
    _recover_symbol_samples,
)
from ._validated_helpers import _estimate_phase_states


@ModulationMetricFactory.register(["phase_states", "psk_phase_states", "constellation_phase_states", "phase_clusters"])
class PhaseStates(BaseModulationMetric):
    """
    Phase States Module.
    Independently estimates discrete constellation phase states (e.g. BPSK, QPSK, 8PSK)
    directly from raw complex IQ.
    Performs internal synchronization and circular phase space embedding.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        symbol_rate_baud: float | None = None,
        samples_per_symbol: float | None = None,
        frequency_correction: str = "auto",
        k_candidates: tuple[int, ...] = (1, 2, 4, 8, 16),
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.symbol_rate_baud = symbol_rate_baud
        self.samples_per_symbol = samples_per_symbol
        self.frequency_correction = str(frequency_correction)
        self.k_candidates = tuple(k_candidates)

    @property
    def name(self) -> str:
        return "phase_states"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> PhaseStatesResult:
        """Executes Phase States estimation directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        # 1. DC removal & RMS normalization
        z_dc, _ = _remove_dc(np.asarray(iq_data))
        z_norm, _ = _normalize_complex_rms(z_dc)

        # 2. Coarse frequency offset correction
        if self.frequency_correction != "none":
            cfo_hz, _ = _estimate_coarse_frequency_offset(z_norm, sample_rate=fs, method=self.frequency_correction)
            z_corr = _apply_frequency_correction(z_norm, sample_rate=fs, freq_offset_hz=cfo_hz)
        else:
            z_corr = z_norm

        # 3. Fractional symbol recovery
        timing_info = _recover_symbol_samples(
            z_corr,
            sample_rate=fs,
            symbol_rate_baud=self.symbol_rate_baud,
            samples_per_symbol=self.samples_per_symbol,
        )

        symbols = timing_info["symbol_points"]

        # 4. Phase state estimation
        phase_info = _estimate_phase_states(
            symbols,
            k_candidates=self.k_candidates,
        )

        return PhaseStatesResult(
            num_phase_states=phase_info["num_phase_states"],
            state_angles_rad=phase_info["state_angles_rad"],
            state_angles_deg=phase_info["state_angles_deg"],
            occupancies=phase_info["occupancies"],
            concentrations=phase_info["concentrations"],
            labels=phase_info["labels"],
            symbol_points=phase_info["symbol_points"],
            model_scores=phase_info["model_scores"],
            confidence=phase_info["confidence"],
            valid=phase_info["valid"],
            reason=phase_info["reason"],
        )


def compute_phase_states(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> PhaseStatesResult:
    """Functional convenience wrapper for PhaseStates."""
    return PhaseStates(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  PhaseStates: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PhaseStates()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
