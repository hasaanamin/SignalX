"""Tests for amplitude envelope quantiles and interquartile range (IQR)."""

import numpy as np
import pytest

from signal_processing.features.statistics import StatisticsFeatureExtractor


class DummyEnvelopeContext:
    def __init__(self, envelope: np.ndarray) -> None:
        self._envelope = envelope

    def envelope(self) -> np.ndarray:
        return self._envelope


class TestAmplitudeQuantiles:
    """Test amplitude envelope quantiles (q01, q05, q25, q75, q95, q99) and IQR."""

    def test_case_f_deterministic_quantiles_and_iqr(self) -> None:
        """Test Case F: Verify against NumPy linear percentile convention on [0..9]."""
        env = np.arange(10, dtype=np.float64)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ctx = DummyEnvelopeContext(env)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)

        # Expected linear interpolation values for N=10:
        # q01 = 0 + 0.09 * 1 = 0.09
        # q05 = 0 + 0.45 * 1 = 0.45
        # q25 = 2 + 0.25 * 1 = 2.25
        # q75 = 6 + 0.75 * 1 = 6.75
        # q95 = 8 + 0.55 * 1 = 8.55
        # q99 = 8 + 0.91 * 1 = 8.91
        # IQR = 6.75 - 2.25 = 4.5
        assert pytest.approx(fset.features["stats.amplitude.q01"].value) == 0.09
        assert pytest.approx(fset.features["stats.amplitude.q05"].value) == 0.45
        assert pytest.approx(fset.features["stats.amplitude.q25"].value) == 2.25
        assert pytest.approx(fset.features["stats.amplitude.q75"].value) == 6.75
        assert pytest.approx(fset.features["stats.amplitude.q95"].value) == 8.55
        assert pytest.approx(fset.features["stats.amplitude.q99"].value) == 8.91
        assert pytest.approx(fset.features["stats.amplitude.iqr"].value) == 4.5

        # Check units and types
        for key in [
            "stats.amplitude.q01",
            "stats.amplitude.q05",
            "stats.amplitude.q25",
            "stats.amplitude.q75",
            "stats.amplitude.q95",
            "stats.amplitude.q99",
            "stats.amplitude.iqr",
        ]:
            fv = fset.features[key]
            assert fv.valid is True
            assert fv.unit == "linear"
            assert isinstance(fv.value, float)
            assert type(fv.value) is float

    def test_single_sample_quantiles_and_iqr(self) -> None:
        """N = 1 single sample: all quantiles equal the sample value and IQR = 0.0."""
        env = np.array([42.0], dtype=np.float64)
        ctx = DummyEnvelopeContext(env)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)

        for key in [
            "stats.amplitude.q01",
            "stats.amplitude.q05",
            "stats.amplitude.q25",
            "stats.amplitude.q75",
            "stats.amplitude.q95",
            "stats.amplitude.q99",
        ]:
            fv = fset.features[key]
            assert fv.valid is True
            assert fv.value == 42.0

        iqr_fv = fset.features["stats.amplitude.iqr"]
        assert iqr_fv.valid is True
        assert iqr_fv.value == 0.0

    def test_case_g_scale_behavior_quantiles_and_iqr(self) -> None:
        """Test Case G: Scaling envelope scales quantiles and IQR by the same factor."""
        env_a = np.array([1.0, 2.0, 4.0, 7.0, 12.0], dtype=np.float64)
        env_b = 3.0 * env_a

        extractor = StatisticsFeatureExtractor()
        fset_a = extractor.extract(DummyEnvelopeContext(env_a))
        fset_b = extractor.extract(DummyEnvelopeContext(env_b))

        for key in [
            "stats.amplitude.q01",
            "stats.amplitude.q05",
            "stats.amplitude.q25",
            "stats.amplitude.q75",
            "stats.amplitude.q95",
            "stats.amplitude.q99",
            "stats.amplitude.iqr",
        ]:
            val_a = fset_a.features[key].value
            val_b = fset_b.features[key].value
            assert pytest.approx(val_b, rel=1e-12) == 3.0 * val_a

    def test_case_h_offset_behavior_quantiles_and_iqr(self) -> None:
        """Test Case H: Offset c shifts quantiles by c, but leaves IQR invariant."""
        env_a = np.array([1.0, 2.0, 4.0, 7.0, 12.0], dtype=np.float64)
        c = 5.0
        env_b = env_a + c

        extractor = StatisticsFeatureExtractor()
        fset_a = extractor.extract(DummyEnvelopeContext(env_a))
        fset_b = extractor.extract(DummyEnvelopeContext(env_b))

        for key in [
            "stats.amplitude.q01",
            "stats.amplitude.q05",
            "stats.amplitude.q25",
            "stats.amplitude.q75",
            "stats.amplitude.q95",
            "stats.amplitude.q99",
        ]:
            val_a = fset_a.features[key].value
            val_b = fset_b.features[key].value
            assert pytest.approx(val_b, rel=1e-12) == val_a + c

        iqr_a = fset_a.features["stats.amplitude.iqr"].value
        iqr_b = fset_b.features["stats.amplitude.iqr"].value
        assert pytest.approx(iqr_b, rel=1e-12) == iqr_a

    def test_quantile_monotonicity(self) -> None:
        """Verify quantile values are monotonically non-decreasing."""
        rng = np.random.default_rng(999)
        extractor = StatisticsFeatureExtractor()

        for _ in range(10):
            env = rng.exponential(scale=2.0, size=50)
            fset = extractor.extract(DummyEnvelopeContext(env))

            q01 = fset.features["stats.amplitude.q01"].value
            q05 = fset.features["stats.amplitude.q05"].value
            q25 = fset.features["stats.amplitude.q25"].value
            q75 = fset.features["stats.amplitude.q75"].value
            q95 = fset.features["stats.amplitude.q95"].value
            q99 = fset.features["stats.amplitude.q99"].value
            iqr = fset.features["stats.amplitude.iqr"].value

            assert q01 <= q05 <= q25 <= q75 <= q95 <= q99
            assert iqr >= 0.0
            assert pytest.approx(iqr) == (q75 - q25)
