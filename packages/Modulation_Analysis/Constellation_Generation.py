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
from ._results import ConstellationResult
from ._constellation_helpers import (
    _remove_dc,
    _normalize_complex_rms,
    _estimate_coarse_frequency_offset,
    _apply_frequency_correction,
    _recover_symbol_samples,
    _estimate_common_phase_offset,
)


@ModulationMetricFactory.register(["constellation_generation", "constellation", "symbol_constellation", "iq_constellation"])
class ConstellationGeneration(BaseModulationMetric):
    """
    Constellation Generation Module.
    Independently recovers synchronized complex baseband symbol points directly from raw complex IQ.
    Executes DC cancellation, RMS power scaling, carrier frequency offset correction,
    fractional symbol timing interpolation, and common constellation phase alignment.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        mode: str = "raw",
        symbol_rate_baud: float | None = None,
        samples_per_symbol: float | None = None,
        frequency_correction: str = "none",
        phase_correction: bool = False,
        remove_dc_offset: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.mode = str(mode).lower().strip()
        self.symbol_rate_baud = symbol_rate_baud
        self.samples_per_symbol = samples_per_symbol
        self.frequency_correction = str(frequency_correction)
        self.phase_correction = bool(phase_correction)
        self.remove_dc_offset = bool(remove_dc_offset)

    @property
    def name(self) -> str:
        return "constellation_generation"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> ConstellationResult:
        """Executes Constellation Generation directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate
        raw_z = np.asarray(iq_data)
        n = len(raw_z)

        # 1. DC Removal
        if self.remove_dc_offset and n > 0:
            z_dc, dc_val = _remove_dc(raw_z)
        else:
            z_dc, dc_val = raw_z.copy(), 0.0 + 0.0j

        # 2. RMS Normalization
        if n > 0:
            z_norm, scale_val = _normalize_complex_rms(z_dc)
        else:
            z_norm, scale_val = np.array([], dtype=np.complex128), 1.0

        target_mode = self.mode
        # If user explicitly passed symbol_rate_baud, allow symbol mode
        if target_mode != "raw" or self.symbol_rate_baud is not None:
            # 3. Frequency Offset Correction
            if self.frequency_correction != "none" and n > 0:
                cfo_hz, freq_method = _estimate_coarse_frequency_offset(z_norm, sample_rate=fs, method=self.frequency_correction)
                z_corr = _apply_frequency_correction(z_norm, sample_rate=fs, freq_offset_hz=cfo_hz)
            else:
                cfo_hz, freq_method = 0.0, "none"
                z_corr = z_norm

            # 4. Fractional Symbol Timing Recovery
            timing_info = _recover_symbol_samples(
                z_corr,
                sample_rate=fs,
                symbol_rate_baud=self.symbol_rate_baud,
                samples_per_symbol=self.samples_per_symbol,
            )
            symbols = timing_info["symbol_points"]

            # 5. Phase Offset Alignment
            if self.phase_correction and len(symbols) >= 8:
                symbols_derot, phase_offset = _estimate_common_phase_offset(symbols, psk_order=4)
            else:
                symbols_derot, phase_offset = symbols, 0.0

            return ConstellationResult(
                raw_points=raw_z,
                normalized_points=z_norm,
                symbol_points=symbols_derot,
                symbol_times_sec=timing_info["symbol_times_sec"],
                estimated_symbol_rate_baud=timing_info["symbol_rate_baud"],
                samples_per_symbol=timing_info["samples_per_symbol"],
                timing_offset_samples=timing_info["timing_offset_samples"],
                estimated_frequency_offset_hz=cfo_hz,
                estimated_phase_offset_rad=phase_offset,
                dc_removed=dc_val,
                normalization_scale=scale_val,
                frequency_correction_method=freq_method,
                timing_method="fractional_interp_dispersion",
                confidence=timing_info["confidence"],
                valid=timing_info["valid"],
            )

        # Default "raw" mode: simple, fast, robust I/Q coordinate normalization
        times_sec = np.arange(n, dtype=np.float64) / fs if n > 0 else np.array([], dtype=np.float64)
        return ConstellationResult(
            raw_points=raw_z,
            normalized_points=z_norm,
            symbol_points=z_norm,
            symbol_times_sec=times_sec,
            estimated_symbol_rate_baud=None,
            samples_per_symbol=None,
            timing_offset_samples=None,
            estimated_frequency_offset_hz=0.0,
            estimated_phase_offset_rad=0.0,
            dc_removed=dc_val,
            normalization_scale=scale_val,
            frequency_correction_method="none",
            timing_method="none",
            confidence=1.0 if n > 0 else 0.0,
            valid=n > 0,
        )


def compute_constellation_generation(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> ConstellationResult:
    """Functional convenience wrapper for ConstellationGeneration."""
    return ConstellationGeneration(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  ConstellationGeneration: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = ConstellationGeneration()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
