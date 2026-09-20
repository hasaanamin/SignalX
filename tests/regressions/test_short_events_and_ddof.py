"""Regression tests for short events and amplitude degrees of freedom (ddof).

Requirements 3 & 4:
- N=1 and N=2 events execute through aggregation atomically without unhandled window errors.
- Short events mark spectral/frequency/bandwidth features invalid while retaining computable features.
- Amplitude ddof validation: N <= ddof marks amplitude.std and amplitude.variance invalid with reason,
  while keeping other amplitude features valid.
- Correct ddof=0 behavior.
- Window rejection for length < 3 without unrecorded fallback.
"""

import math

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureVector, SignalEvent, StreamInfo
from signal_processing.features import (
    AmplitudeFeatureConfig,
    AmplitudeFeatureExtractor,
    FeatureAggregator,
    build_default_feature_registry,
)


@pytest.fixture
def stream_info() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-short-events",
        source_id="synthetic",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


class TestShortEventsAggregation:
    """Validate deterministic aggregation on short (N=1, N=2) signal events."""

    def test_n_equals_one_event_aggregation(self, stream_info: StreamInfo) -> None:
        """N=1 event must complete aggregation without crashing halfway through."""
        iq = np.array([1.0 + 0.5j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-n1",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=1,
            detected_start_sample=0,
            detected_end_sample=1,
            trigger_sample=0,
        )

        ctx = SignalContext(event)
        aggregator = FeatureAggregator(build_default_feature_registry())
        vector = aggregator.extract(ctx)

        assert isinstance(vector, FeatureVector)
        assert len(vector.features) == 56

        # Computable temporal and power features remain valid
        assert vector.features["temporal.duration_samples"].valid is True
        assert vector.features["temporal.duration_samples"].value == 1.0
        assert vector.features["power.mean_linear"].valid is True
        assert vector.features["amplitude.mean"].valid is True
        assert pytest.approx(vector.features["amplitude.mean"].value, abs=1e-5) == abs(iq[0])

        # Spectral and frequency features requiring window >= 3 are marked invalid with clear reasons
        assert vector.features["spectral.entropy"].valid is False
        assert "PSD unavailable" in vector.features["spectral.entropy"].reason
        assert vector.features["bandwidth.occupied_99_hz"].valid is False
        assert vector.features["frequency.peak_offset_hz"].valid is False

    def test_n_equals_two_event_aggregation(self, stream_info: StreamInfo) -> None:
        """N=2 event reaches Hann window and must not crash; marks spectral features invalid."""
        iq = np.array([1.0 + 0.0j, 0.5 + 0.5j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-n2",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=2,
            detected_start_sample=0,
            detected_end_sample=2,
            trigger_sample=0,
        )

        ctx = SignalContext(event)
        aggregator = FeatureAggregator(build_default_feature_registry())
        vector = aggregator.extract(ctx)

        assert isinstance(vector, FeatureVector)
        assert len(vector.features) == 56

        # Temporal duration is valid
        assert vector.features["temporal.duration_samples"].valid is True
        assert vector.features["temporal.duration_samples"].value == 2.0

        # Spectral and frequency features are gracefully marked invalid
        assert vector.features["spectral.entropy"].valid is False
        assert "PSD unavailable" in vector.features["spectral.entropy"].reason
        assert vector.features["frequency.peak_offset_hz"].valid is False
        assert vector.features["bandwidth.occupied_99_hz"].valid is False


class TestAmplitudeDdof:
    """Validate amplitude feature extractor handling of sample count vs degrees of freedom."""

    def test_n_equals_one_ddof_one(self, stream_info: StreamInfo) -> None:
        """N=1 with ddof=1: std and variance are invalid; other features are valid."""
        iq = np.array([2.0 + 0.0j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-ddof-1",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=1,
            detected_start_sample=0,
            detected_end_sample=1,
            trigger_sample=0,
        )
        ctx = SignalContext(event)
        extractor = AmplitudeFeatureExtractor(AmplitudeFeatureConfig(ddof=1))
        fset = extractor.extract(ctx)

        assert fset.features["amplitude.std"].valid is False
        assert math.isnan(fset.features["amplitude.std"].value)
        assert "sample count (1) must be strictly greater than ddof (1)" in fset.features["amplitude.std"].reason.lower()

        assert fset.features["amplitude.variance"].valid is False
        assert math.isnan(fset.features["amplitude.variance"].value)

        # Computable features remain valid
        assert fset.features["amplitude.mean"].valid is True
        assert fset.features["amplitude.mean"].value == 2.0
        assert fset.features["amplitude.peak"].valid is True
        assert fset.features["amplitude.min"].valid is True

    def test_n_equals_two_ddof_two(self, stream_info: StreamInfo) -> None:
        """N=2 with ddof=2: std and variance are invalid; other features are valid."""
        iq = np.array([1.0 + 0j, 3.0 + 0j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-ddof-2",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=2,
            detected_start_sample=0,
            detected_end_sample=2,
            trigger_sample=0,
        )
        ctx = SignalContext(event)
        extractor = AmplitudeFeatureExtractor(AmplitudeFeatureConfig(ddof=2))
        fset = extractor.extract(ctx)

        assert fset.features["amplitude.std"].valid is False
        assert fset.features["amplitude.variance"].valid is False
        assert fset.features["amplitude.mean"].valid is True
        assert fset.features["amplitude.mean"].value == 2.0

    def test_ordinary_ddof_zero(self, stream_info: StreamInfo) -> None:
        """N=2 with ddof=0: std and variance are valid population statistics."""
        iq = np.array([1.0 + 0j, 3.0 + 0j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-ddof-0",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=2,
            detected_start_sample=0,
            detected_end_sample=2,
            trigger_sample=0,
        )
        ctx = SignalContext(event)
        extractor = AmplitudeFeatureExtractor(AmplitudeFeatureConfig(ddof=0))
        fset = extractor.extract(ctx)

        # mean = 2.0, deviations = [-1.0, 1.0], variance = 1.0, std = 1.0
        assert fset.features["amplitude.std"].valid is True
        assert fset.features["amplitude.std"].value == 1.0
        assert fset.features["amplitude.variance"].valid is True
        assert fset.features["amplitude.variance"].value == 1.0

    def test_supported_nonzero_ddof(self, stream_info: StreamInfo) -> None:
        """N=3 with ddof=1: sample std and variance are valid."""
        iq = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-ddof-n3",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=3,
            detected_start_sample=0,
            detected_end_sample=3,
            trigger_sample=0,
        )
        ctx = SignalContext(event)
        extractor = AmplitudeFeatureExtractor(AmplitudeFeatureConfig(ddof=1))
        fset = extractor.extract(ctx)

        # Sample variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / (3 - 1) = 2 / 2 = 1.0
        assert fset.features["amplitude.variance"].valid is True
        assert fset.features["amplitude.variance"].value == 1.0
        assert fset.features["amplitude.std"].valid is True
        assert fset.features["amplitude.std"].value == 1.0
