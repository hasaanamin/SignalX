"""Tests for AmplitudeFeatureExtractor mathematical accuracy, units, and error handling."""

import math

import numpy as np
import pytest

from signal_processing.core import FeatureSet
from signal_processing.features.amplitude import (
    AmplitudeFeatureConfig,
    AmplitudeFeatureError,
    AmplitudeFeatureExtractor,
)


class DummyContext:
    """Minimal duck-typed context providing an envelope."""

    def __init__(self, envelope: np.ndarray) -> None:
        self._envelope = envelope

    def envelope(self) -> np.ndarray:
        return self._envelope


class TestAmplitudeExtractor:
    """Test amplitude feature extraction on synthetic envelopes."""

    def test_constant_envelope(self) -> None:
        """Constant envelope [1.0, 1.0, 1.0, 1.0] must produce known exact metrics."""
        env = np.ones(100, dtype=np.float64)
        ctx = DummyContext(env)
        extractor = AmplitudeFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "amplitude"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 9

        assert fset.features["amplitude.mean"].value == 1.0
        assert fset.features["amplitude.mean"].unit == "linear"
        assert fset.features["amplitude.mean"].valid is True

        assert fset.features["amplitude.rms"].value == 1.0
        assert fset.features["amplitude.rms"].unit == "linear"

        assert fset.features["amplitude.peak"].value == 1.0
        assert fset.features["amplitude.peak"].unit == "linear"

        assert fset.features["amplitude.min"].value == 1.0
        assert fset.features["amplitude.min"].unit == "linear"

        assert fset.features["amplitude.std"].value == 0.0
        assert fset.features["amplitude.std"].unit == "linear"

        assert fset.features["amplitude.variance"].value == 0.0
        assert fset.features["amplitude.variance"].unit == "linear^2"

        assert fset.features["amplitude.median"].value == 1.0
        assert fset.features["amplitude.median"].unit == "linear"

        assert fset.features["amplitude.peak_to_peak"].value == 0.0
        assert fset.features["amplitude.peak_to_peak"].unit == "linear"

        assert fset.features["amplitude.crest_factor"].value == 1.0
        assert fset.features["amplitude.crest_factor"].unit is None
        assert fset.features["amplitude.crest_factor"].valid is True

    def test_known_varying_envelope(self) -> None:
        """Known envelope [1.0, 2.0, 3.0, 4.0] must match exact analytical definitions."""
        env = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        ctx = DummyContext(env)
        extractor = AmplitudeFeatureExtractor()

        fset = extractor.extract(ctx)

        # Expected analytical values:
        # mean = 10 / 4 = 2.5
        # rms = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5)
        # peak = 4.0
        # min = 1.0
        # std (ddof=0) = sqrt(1.25)
        # var (ddof=0) = 1.25
        # median = 2.5
        # peak_to_peak = 4.0 - 1.0 = 3.0
        # crest_factor = 4.0 / sqrt(7.5)

        expected_mean = 2.5
        expected_rms = float(math.sqrt(7.5))
        expected_peak = 4.0
        expected_min = 1.0
        expected_std = float(math.sqrt(1.25))
        expected_var = 1.25
        expected_median = 2.5
        expected_ptp = 3.0
        expected_cf = float(4.0 / math.sqrt(7.5))

        assert pytest.approx(fset.features["amplitude.mean"].value) == expected_mean
        assert pytest.approx(fset.features["amplitude.rms"].value) == expected_rms
        assert pytest.approx(fset.features["amplitude.peak"].value) == expected_peak
        assert pytest.approx(fset.features["amplitude.min"].value) == expected_min
        assert pytest.approx(fset.features["amplitude.std"].value) == expected_std
        assert pytest.approx(fset.features["amplitude.variance"].value) == expected_var
        assert pytest.approx(fset.features["amplitude.median"].value) == expected_median
        assert pytest.approx(fset.features["amplitude.peak_to_peak"].value) == expected_ptp
        assert pytest.approx(fset.features["amplitude.crest_factor"].value) == expected_cf

        # Check that output values are native Python float types
        for k, fv in fset.features.items():
            assert isinstance(fv.value, float), f"Feature {k} value is not float: {type(fv.value)}"
            assert type(fv.value) is float

    def test_custom_ddof_config(self) -> None:
        """Verify extractor applies custom sample degrees of freedom (ddof=1) if configured."""
        env = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        ctx = DummyContext(env)
        extractor = AmplitudeFeatureExtractor(config=AmplitudeFeatureConfig(ddof=1))

        fset = extractor.extract(ctx)

        expected_var_sample = float(np.var(env, ddof=1))
        expected_std_sample = float(np.std(env, ddof=1))

        assert pytest.approx(fset.features["amplitude.variance"].value) == expected_var_sample
        assert pytest.approx(fset.features["amplitude.std"].value) == expected_std_sample

    def test_reject_invalid_context_and_config(self) -> None:
        """Verify strict error handling on malformed contexts or configurations."""
        extractor = AmplitudeFeatureExtractor()

        with pytest.raises(AmplitudeFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(AmplitudeFeatureError, match="envelope"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

        with pytest.raises(AmplitudeFeatureError, match="AmplitudeFeatureConfig"):
            AmplitudeFeatureExtractor(config="invalid_config")  # type: ignore[arg-type]

    def test_reject_invalid_envelope_data(self) -> None:
        """Verify extractor rejects empty, multidimensional, non-finite, complex, or negative envelopes."""
        extractor = AmplitudeFeatureExtractor()

        # Access config property
        assert extractor.config.ddof == 0

        # Context envelope() raising exception
        class BrokenContext:
            def envelope(self) -> np.ndarray:
                raise RuntimeError("Hardware failure")

        with pytest.raises(AmplitudeFeatureError, match="Failed to obtain envelope"):
            extractor.extract(BrokenContext())

        with pytest.raises(AmplitudeFeatureError, match="non-empty"):
            extractor.extract(DummyContext(np.array([])))

        with pytest.raises(AmplitudeFeatureError, match="1-dimensional"):
            extractor.extract(DummyContext(np.zeros((5, 2))))

        with pytest.raises(AmplitudeFeatureError, match="real-valued"):
            extractor.extract(DummyContext(np.array([1.0 + 2.0j, 3.0])))

        with pytest.raises(AmplitudeFeatureError, match="non-finite"):
            extractor.extract(DummyContext(np.array([1.0, np.nan, 2.0])))

        with pytest.raises(AmplitudeFeatureError, match="non-finite"):
            extractor.extract(DummyContext(np.array([1.0, np.inf, 2.0])))

        with pytest.raises(AmplitudeFeatureError, match="negative"):
            extractor.extract(DummyContext(np.array([1.0, -0.5, 2.0])))

