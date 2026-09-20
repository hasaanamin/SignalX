"""Tests for amplitude envelope population skewness calculation."""

import math

import numpy as np
import pytest

from signal_processing.features.statistics import StatisticsFeatureExtractor


class DummyEnvelopeContext:
    def __init__(self, envelope: np.ndarray) -> None:
        self._envelope = envelope

    def envelope(self) -> np.ndarray:
        return self._envelope


class TestAmplitudeSkewness:
    """Test amplitude envelope skewness calculations and edge cases."""

    def test_case_a_symmetric_distribution_skewness_zero(self) -> None:
        """Test Case A: Symmetric distribution [1, 2, 3, 4, 5] yields skewness of exactly 0.0."""
        env = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        ctx = DummyEnvelopeContext(env)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["stats.amplitude.skewness"]

        assert fv.valid is True
        assert fv.unit is None
        assert fv.reason is None
        assert isinstance(fv.value, float)
        assert type(fv.value) is float
        assert pytest.approx(fv.value, abs=1e-12) == 0.0

    def test_case_b_positively_skewed_distribution(self) -> None:
        """Test Case B: Right-tailed distribution [1, 1, 1, 2, 8] yields positive skewness matching formula."""
        env = np.array([1.0, 1.0, 1.0, 2.0, 8.0], dtype=np.float64)
        ctx = DummyEnvelopeContext(env)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["stats.amplitude.skewness"]

        # Independent calculation:
        # mu = 13 / 5 = 2.6
        # diff = [-1.6, -1.6, -1.6, -0.6, 5.4]
        # m2 = (3 * 2.56 + 0.36 + 29.16) / 5 = 37.20 / 5 = 7.44
        # m3 = (3 * (-4.096) + (-0.216) + 157.464) / 5 = 144.96 / 5 = 28.992
        # skew = 28.992 / (7.44 ** 1.5)
        expected_skew = 28.992 / (7.44 ** 1.5)

        assert fv.valid is True
        assert fv.value > 0.0
        assert pytest.approx(fv.value, rel=1e-12) == expected_skew

    def test_negatively_skewed_distribution(self) -> None:
        """Verify left-tailed distribution yields negative skewness."""
        env = np.array([1.0, 7.0, 8.0, 8.0, 8.0], dtype=np.float64)
        ctx = DummyEnvelopeContext(env)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["stats.amplitude.skewness"]

        assert fv.valid is True
        assert fv.value < 0.0

    def test_short_signals_skewness_requirements(self) -> None:
        """Skewness requires at least N >= 3 samples."""
        extractor = StatisticsFeatureExtractor()

        # N = 1
        ctx1 = DummyEnvelopeContext(np.array([5.0], dtype=np.float64))
        fset1 = extractor.extract(ctx1)
        fv1 = fset1.features["stats.amplitude.skewness"]
        assert fv1.valid is False
        assert math.isnan(fv1.value)
        assert "At least 3 samples are required" in fv1.reason

        # N = 2
        ctx2 = DummyEnvelopeContext(np.array([2.0, 5.0], dtype=np.float64))
        fset2 = extractor.extract(ctx2)
        fv2 = fset2.features["stats.amplitude.skewness"]
        assert fv2.valid is False
        assert math.isnan(fv2.value)
        assert "At least 3 samples are required" in fv2.reason

        # N = 3
        ctx3 = DummyEnvelopeContext(np.array([1.0, 2.0, 5.0], dtype=np.float64))
        fset3 = extractor.extract(ctx3)
        fv3 = fset3.features["stats.amplitude.skewness"]
        assert fv3.valid is True
        assert isinstance(fv3.value, float)

    def test_case_g_scale_invariance_skewness(self) -> None:
        """Test Case G: Multiplying envelope by a positive constant preserves skewness."""
        env_a = np.array([1.0, 2.0, 4.0, 7.0, 12.0], dtype=np.float64)
        env_b = 3.0 * env_a

        extractor = StatisticsFeatureExtractor()
        fset_a = extractor.extract(DummyEnvelopeContext(env_a))
        fset_b = extractor.extract(DummyEnvelopeContext(env_b))

        val_a = fset_a.features["stats.amplitude.skewness"].value
        val_b = fset_b.features["stats.amplitude.skewness"].value

        assert pytest.approx(val_a, rel=1e-12) == val_b

    def test_case_h_offset_invariance_skewness(self) -> None:
        """Test Case H: Adding a positive constant offset preserves skewness."""
        env_a = np.array([1.0, 2.0, 4.0, 7.0, 12.0], dtype=np.float64)
        env_b = env_a + 5.0

        extractor = StatisticsFeatureExtractor()
        fset_a = extractor.extract(DummyEnvelopeContext(env_a))
        fset_b = extractor.extract(DummyEnvelopeContext(env_b))

        val_a = fset_a.features["stats.amplitude.skewness"].value
        val_b = fset_b.features["stats.amplitude.skewness"].value

        assert pytest.approx(val_a, rel=1e-12) == val_b
