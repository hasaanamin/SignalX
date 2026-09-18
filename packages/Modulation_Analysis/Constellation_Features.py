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
from ._results import ConstellationFeaturesResult
from ._constellation_helpers import (
    _remove_dc,
    _normalize_complex_rms,
    _estimate_coarse_frequency_offset,
    _apply_frequency_correction,
    _recover_symbol_samples,
    _constellation_cluster_statistics,
)


@ModulationMetricFactory.register(["constellation_features", "constellation_metrics", "digital_features", "cluster_metrics"])
class ConstellationFeatures(BaseModulationMetric):
    """
    Constellation Features Module.
    Independently computes statistical, geometric, and cluster metrics of the digital constellation
    directly from raw complex IQ.
    Extracts complex radial moments, circularity, radial levels, IQ covariance eigenvalues,
    cluster compactness, and EVM (when reference is verified).
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        symbol_rate_baud: float | None = None,
        samples_per_symbol: float | None = None,
        frequency_correction: str = "auto",
        reference_constellation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.symbol_rate_baud = symbol_rate_baud
        self.samples_per_symbol = samples_per_symbol
        self.frequency_correction = str(frequency_correction)
        self.reference_constellation = reference_constellation

    @property
    def name(self) -> str:
        return "constellation_features"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        reference_constellation: Any | None = None,
        **kwargs: Any,
    ) -> ConstellationFeaturesResult:
        """Executes Constellation Features extraction directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate
        raw_z = np.asarray(iq_data)
        ref_const = reference_constellation if reference_constellation is not None else self.reference_constellation

        # 1. DC Removal & RMS Normalization
        z_dc, _ = _remove_dc(raw_z)
        z_norm, _ = _normalize_complex_rms(z_dc)

        # 2. Coarse CFO Correction
        if self.frequency_correction != "none":
            cfo_hz, _ = _estimate_coarse_frequency_offset(z_norm, sample_rate=fs, method=self.frequency_correction)
            z_corr = _apply_frequency_correction(z_norm, sample_rate=fs, freq_offset_hz=cfo_hz)
        else:
            z_corr = z_norm

        # 3. Fractional Symbol Recovery
        timing_info = _recover_symbol_samples(
            z_corr,
            sample_rate=fs,
            symbol_rate_baud=self.symbol_rate_baud,
            samples_per_symbol=self.samples_per_symbol,
        )

        symbols = timing_info["symbol_points"]

        # 4. Cluster Statistics
        stats = _constellation_cluster_statistics(symbols, sample_rate=fs, reference_constellation=ref_const)

        return ConstellationFeaturesResult(
            num_symbol_points=stats["num_symbol_points"],
            mean_complex=stats["mean_complex"],
            mean_power=stats["mean_power"],
            normalized_fourth_moment=stats["normalized_fourth_moment"],
            circularity_coefficient=stats["circularity_coefficient"],
            mean_radius=stats["mean_radius"],
            radius_std=stats["radius_std"],
            radius_cv=stats["radius_cv"],
            radius_skewness=stats["radius_skewness"],
            radius_kurtosis=stats["radius_kurtosis"],
            radial_level_count=stats["radial_level_count"],
            apparent_phase_state_count=stats["apparent_phase_state_count"],
            phase_concentration=stats["phase_concentration"],
            phase_occupancy_uniformity=stats["phase_occupancy_uniformity"],
            iq_covariance=stats["iq_covariance"],
            covariance_eigenvalues=stats["covariance_eigenvalues"],
            covariance_axis_ratio=stats["covariance_axis_ratio"],
            constellation_cluster_count=stats["constellation_cluster_count"],
            cluster_compactness=stats["cluster_compactness"],
            minimum_cluster_separation=stats["minimum_cluster_separation"],
            evm_rms=stats["evm_rms"],
            evm_percent=stats["evm_percent"],
            evm_valid=stats["evm_valid"],
            confidence=stats["confidence"],
            valid=stats["valid"],
        )


def compute_constellation_features(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> ConstellationFeaturesResult:
    """Functional convenience wrapper for ConstellationFeatures."""
    return ConstellationFeatures(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  ConstellationFeatures: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = ConstellationFeatures()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
