"""Tests for amplitude envelope population excess kurtosis calculation."""

import math

import numpy as np
import pytest

from signal_processing.features.statistics import StatisticsFeatureExtractor


class DummyEnvelopeContext:
    def __init__(self, envelope: np.ndarray) -> None:
        self._envelope = envelope

    def envelope(self) -> np.ndarray:
        return self._envelope


class TestAmplitudeExcessKurtosis:
    """Test amplitude envelope excess kurtosis calculations and edge cases."""

    def test_case_a_symmetric_distribution_excess_kurtosis(self) -> None:
        """Test Case A: Verify against independently computed excess kurtosis for [1, 2, 3, 4, 5]."""
        env = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        ctx = DummyEnvelopeContext(env)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["stats.amplitude.excess_kurtosis"]

        # Independent calculation:
        # mu = 3.0
        # diff = [-2, -1, 0, 1, 2]
        # m2 = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
        # m4 = (16 + 1 + 0 + 1 + 16) / 5 = 34 / 5 = 6.8
        # raw_kurtosis = 6.8 / (2.0 ** 2) = 6.8 / 4.0 = 1.7
        # excess_kurtosis = 1.7 - 3.0 = -1.3
        expected_excess = -1.3

        assert fv.valid is True
        assert fv.unit is None
        assert fv.reason is None
        assert isinstance(fv.value, float)
        assert type(fv.value) is float
        assert pytest.approx(fv.value, abs=1e-12) == expected_excess

    def test_heavy_tailed_distribution_excess_kurtosis(self) -> None:
        """Verify heavy-tailed distribution [1, 1, 1, 1, 10] has positive excess kurtosis."""
        env = np.array([1.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float64)
        ctx = DummyEnvelopeContext(env)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)
        fv = fset.features["stats.amplitude.excess_kurtosis"]

        assert fv.valid is True
        assert fv.value > 0.0

    def test_short_signals_excess_kurtosis_requirements(self) -> None:
        """Excess kurtosis requires at least N >= 4 samples."""
        extractor = StatisticsFeatureExtractor()

        # N = 1
        ctx1 = DummyEnvelopeContext(np.array([5.0], dtype=np.float64))
        fset1 = extractor.extract(ctx1)
        fv1 = fset1.features["stats.amplitude.excess_kurtosis"]
        assert fv1.valid is False
        assert math.isnan(fv1.value)
        assert "At least 4 samples are required" in fv1.reason

        # N = 2
        ctx2 = DummyEnvelopeContext(np.array([2.0, 5.0], dtype=np.float64))
        fset2 = extractor.extract(ctx2)
        fv2 = fset2.features["stats.amplitude.excess_kurtosis"]
        assert fv2.valid is False
        assert math.isnan(fv2.value)
        assert "At least 4 samples are required" in fv2.reason

        # N = 3
        ctx3 = DummyEnvelopeContext(np.array([1.0, 2.0, 5.0], dtype=np.float64))
        fset3 = extractor.extract(ctx3)
        fv3 = fset3.features["stats.amplitude.excess_kurtosis"]
        assert fv3.valid is False
        assert math.isnan(fv3.value)
        assert "At least 4 samples are required" in fv3.reason

        # N = 4
        ctx4 = DummyEnvelopeContext(np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float64))
        fset4 = extractor.extract(ctx4)
        fv4 = fset4.features["stats.amplitude.excess_kurtosis"]
        assert fv4.valid is True
        assert isinstance(fv4.value, float)

    def test_case_g_scale_invariance_excess_kurtosis(self) -> None:
        """Test Case G: Multiplying envelope by a positive constant preserves excess kurtosis."""
        env_a = np.array([1.0, 2.0, 4.0, 7.0, 12.0], dtype=np.float64)
        env_b = 3.0 * env_a

        extractor = StatisticsFeatureExtractor()
        fset_a = extractor.extract(DummyEnvelopeContext(env_a))
        fset_b = extractor.extract(DummyEnvelopeContext(env_b))

        val_a = fset_a.features["stats.amplitude.excess_kurtosis"].value
        val_b = fset_b.features["stats.amplitude.excess_kurtosis"].value

        assert pytest.approx(val_a, rel=1e-12) == val_b

    def test_case_h_offset_invariance_excess_kurtosis(self) -> None:
        """Test Case H: Adding a positive constant offset preserves excess kurtosis."""
        env_a = np.array([1.0, 2.0, 4.0, 7.0, 12.0], dtype=np.float64)
        env_b = env_a + 5.0

        extractor = StatisticsFeatureExtractor()
        fset_a = extractor.extract(DummyEnvelopeContext(env_a))
        fset_b = extractor.extract(DummyEnvelopeContext(env_b))

        val_a = fset_a.features["stats.amplitude.excess_kurtosis"].value
        val_b = fset_b.features["stats.amplitude.excess_kurtosis"].value

        assert pytest.approx(val_a, rel=1e-12) == val_b
