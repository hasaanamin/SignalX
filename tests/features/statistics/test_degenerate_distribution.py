"""Tests for degenerate distributions (constant amplitude and all-zero signals)."""

import math

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import SignalEvent, StreamInfo
from signal_processing.features.statistics import StatisticsFeatureExtractor


class DummyEnvelopeContext:
    def __init__(self, envelope: np.ndarray) -> None:
        self._envelope = envelope

    def envelope(self) -> np.ndarray:
        return self._envelope


@pytest.fixture
def zero_event() -> SignalEvent:
    """Fixture producing an all-zero SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-stat-zero",
        source_id="synthetic-zero",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    total = 500
    iq = np.zeros(total, dtype=np.complex64)

    return SignalEvent(
        event_id="ev-stat-zero-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestDegenerateDistribution:
    """Test constant-amplitude and zero-signal distributions."""

    def test_case_c_constant_amplitude_distribution(self) -> None:
        """Test Case C: Constant amplitude [2, 2, 2, 2, 2] yields invalid standardized moments but valid quantiles."""
        env = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        ctx = DummyEnvelopeContext(env)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)

        # Skewness
        skew = fset.features["stats.amplitude.skewness"]
        assert skew.valid is False
        assert math.isnan(skew.value)
        assert skew.unit is None
        assert skew.reason == "Amplitude variance is zero"

        # Excess Kurtosis
        kurt = fset.features["stats.amplitude.excess_kurtosis"]
        assert kurt.valid is False
        assert math.isnan(kurt.value)
        assert kurt.unit is None
        assert kurt.reason == "Amplitude variance is zero"

        # Quantiles all equal 2.0
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
            assert fv.unit == "linear"
            assert fv.value == 2.0

        # IQR = 0.0
        iqr = fset.features["stats.amplitude.iqr"]
        assert iqr.valid is True
        assert iqr.unit == "linear"
        assert iqr.value == 0.0

    def test_case_d_zero_signal_real_context(self, zero_event: SignalEvent) -> None:
        """Test Case D: Real SignalContext with all-zero IQ yields valid zero quantiles and invalid moments."""
        ctx = SignalContext(zero_event)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)

        # Skewness and excess kurtosis must be invalid
        skew = fset.features["stats.amplitude.skewness"]
        assert skew.valid is False
        assert math.isnan(skew.value)
        assert skew.reason == "Amplitude variance is zero"

        kurt = fset.features["stats.amplitude.excess_kurtosis"]
        assert kurt.valid is False
        assert math.isnan(kurt.value)
        assert kurt.reason == "Amplitude variance is zero"

        # All quantiles must be 0.0 and valid
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
            assert fv.unit == "linear"
            assert fv.value == 0.0

        # IQR must be 0.0 and valid
        iqr = fset.features["stats.amplitude.iqr"]
        assert iqr.valid is True
        assert iqr.unit == "linear"
        assert iqr.value == 0.0
