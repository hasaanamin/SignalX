"""Tests for normalized Shannon spectral entropy calculation."""

import math

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


class TestSpectralEntropy:
    """Test normalized Shannon spectral entropy across varied spectral energy distributions."""

    def test_case_a_single_dominant_bin_entropy_near_zero(self) -> None:
        """Test Case A: Power concentrated in a single bin produces entropy approximately 0."""
        freqs = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.entropy"]

        assert fv.valid is True
        assert fv.unit is None
        assert fv.reason is None
        assert isinstance(fv.value, float)
        assert type(fv.value) is float
        assert pytest.approx(fv.value, abs=1e-12) == 0.0

    def test_case_b_uniform_spectrum_entropy_equals_one(self) -> None:
        """Test Case B: Uniform power across bins produces normalized entropy of exactly 1.0."""
        freqs = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
        powers = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.entropy"]

        assert fv.valid is True
        assert fv.unit is None
        assert pytest.approx(fv.value, abs=1e-12) == 1.0

    def test_case_c_known_nonuniform_entropy_formula(self) -> None:
        """Test Case C: Verify against an independently computed two-level spectrum [1, 1, 4, 4]."""
        freqs = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
        powers = np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.entropy"]

        # Independent manual calculation:
        # Total power = 10.0
        # Probabilities: [0.1, 0.1, 0.4, 0.4]
        # H = - (2 * 0.1 * ln(0.1) + 2 * 0.4 * ln(0.4))
        # Norm factor = ln(4)
        p = np.array([0.1, 0.1, 0.4, 0.4])
        expected_raw_h = -np.sum(p * np.log(p))
        expected_norm_h = expected_raw_h / np.log(4.0)

        assert fv.valid is True
        assert pytest.approx(fv.value, rel=1e-12) == expected_norm_h

    def test_partially_zero_bins_entropy(self) -> None:
        """Test Case D: Spectrum with some zero bins [0, 1, 1, 0] ignores 0*ln(0) correctly."""
        freqs = np.array([-150.0, -50.0, 50.0, 150.0], dtype=np.float64)
        powers = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.entropy"]

        # Independent calculation:
        # Total power = 2.0
        # Nonzero probabilities: [0.5, 0.5]
        # Raw H = - (0.5*ln(0.5) + 0.5*ln(0.5)) = ln(2)
        # N = 4 bins -> norm factor = ln(4) = 2*ln(2)
        # Expected = ln(2) / (2*ln(2)) = 0.5
        assert fv.valid is True
        assert pytest.approx(fv.value, abs=1e-12) == 0.5

    def test_case_f_one_bin_entropy_invalid(self) -> None:
        """Test Case F: Normalized entropy is undefined for single-bin PSD (ln(1) = 0)."""
        freqs = np.array([100.0], dtype=np.float64)
        powers = np.array([2.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["spectral.entropy"]

        assert fv.valid is False
        assert math.isnan(fv.value)
        assert fv.unit is None
        assert "At least two PSD bins are required" in fv.reason

    def test_entropy_bounds_clamping(self) -> None:
        """Verify entropy stays strictly within [0.0, 1.0] across diverse spectral profiles."""
        rng = np.random.default_rng(42)
        freqs = np.linspace(-1000.0, 1000.0, 64)
        extractor = SpectralFeatureExtractor()

        for _ in range(10):
            powers = rng.uniform(0.01, 100.0, size=64)
            ctx = DummyPSDContext(freqs, powers)
            fset = extractor.extract(ctx)
            val = fset.features["spectral.entropy"].value
            assert 0.0 <= val <= 1.0
