"""Tests for spectral flatness calculation."""

import numpy as np
import pytest

from signal_processing.core import PSDResult
from signal_processing.features.spectral import SpectralFeatureExtractor


class DummyPSDContext:
    def __init__(self, frequency_offset_hz: np.ndarray, power_linear: np.ndarray) -> None:
        self._psd = PSDResult(
            frequency_offset_hz=frequency_offset_hz,
            power_linear=power_linear,
            power_dbfs_per_hz=np.full_like(power_linear, -100.0),
        )

    def psd(self) -> PSDResult:
        return self._psd


class TestSpectralFlatness:
    """Test spectral flatness (geometric mean / arithmetic mean) calculations."""

    def test_case_a_single_dominant_bin_with_zero_bins_flatness_is_zero(self) -> None:
        """Test Case A: A spectrum with positive power and at least one zero bin yields 0.0 flatness."""
        freqs = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.flatness"]

        assert fv.valid is True
        assert fv.unit is None
        assert fv.reason is None
        assert isinstance(fv.value, float)
        assert type(fv.value) is float
        assert fv.value == 0.0

    def test_case_b_uniform_spectrum_flatness_equals_one(self) -> None:
        """Test Case B: Completely uniform power yields flatness of 1.0."""
        freqs = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
        powers = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.flatness"]

        assert fv.valid is True
        assert fv.unit is None
        assert pytest.approx(fv.value, abs=1e-12) == 1.0

    def test_case_c_known_two_level_spectrum_flatness(self) -> None:
        """Test Case C: Verify against independently computed two-level spectrum [1, 1, 4, 4]."""
        freqs = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
        powers = np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.flatness"]

        # Independent calculation:
        # Geometric mean: (1 * 1 * 4 * 4)**(1/4) = 16**(1/4) = 2.0
        # Arithmetic mean: (1 + 1 + 4 + 4) / 4 = 10 / 4 = 2.5
        # Expected ratio: 2.0 / 2.5 = 0.8
        assert fv.valid is True
        assert pytest.approx(fv.value, rel=1e-12) == 0.8

    def test_case_d_partially_zero_bins_flatness_is_zero_and_valid(self) -> None:
        """Test Case D: Spectrum with some zero bins [0, 1, 1, 0] produces exactly 0.0 with valid=True."""
        freqs = np.array([-150.0, -50.0, 50.0, 150.0], dtype=np.float64)
        powers = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.flatness"]

        assert fv.valid is True
        assert fv.reason is None
        assert fv.value == 0.0

    def test_case_f_one_bin_flatness_equals_one(self) -> None:
        """Test Case F: Single bin PSD has geometric mean equal to arithmetic mean -> flatness = 1.0."""
        freqs = np.array([100.0], dtype=np.float64)
        powers = np.array([2.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.flatness"]

        assert fv.valid is True
        assert fv.reason is None
        assert fv.value == 1.0

    def test_flatness_bounds_clamping(self) -> None:
        """Verify flatness stays strictly within [0.0, 1.0] across diverse spectral profiles."""
        rng = np.random.default_rng(123)
        freqs = np.linspace(-1000.0, 1000.0, 64)
        extractor = SpectralFeatureExtractor()

        for _ in range(10):
            powers = rng.uniform(0.01, 100.0, size=64)
            ctx = DummyPSDContext(freqs, powers)
            fset = extractor.extract(ctx)
            val = fset.features["spectral.flatness"].value
            assert 0.0 <= val <= 1.0
