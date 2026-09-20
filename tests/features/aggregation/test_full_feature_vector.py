"""Full integration test for FeatureAggregator producing complete 56-feature FeatureVector."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureValue, FeatureVector, SignalEvent, StreamInfo
from signal_processing.features.aggregation import (
    FeatureAggregator,
    build_default_feature_registry,
)


@pytest.fixture
def realistic_signal_event() -> SignalEvent:
    """Create a deterministic, non-trivial multi-component RF pulse event."""
    stream = StreamInfo(
        stream_id="stream-full-vector",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    rng = np.random.RandomState(42)
    n = np.arange(1024)
    # Dual complex tone with random BPSK phase modulation and noise to ensure non-degenerate statistics
    carrier = np.exp(1j * 2.0 * np.pi * 50_000.0 * n / 1_000_000.0)
    sub = 0.5 * np.exp(1j * 2.0 * np.pi * 120_000.0 * n / 1_000_000.0)
    noise = (rng.randn(len(n)) + 1j * rng.randn(len(n))) * 0.05
    iq = (carrier + sub + noise).astype(np.complex64)
    total = len(iq)

    return SignalEvent(
        event_id="ev-full-pipeline-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=32,
        detected_end_sample=total - 32,
        trigger_sample=128,
    )


class TestFullFeatureVectorIntegration:
    """Validate end-to-end integration across all 10 feature extractor families producing 56 features."""

    def test_full_default_integration_and_fifty_six_features(
        self, realistic_signal_event: SignalEvent
    ) -> None:
        """Default aggregator extracts exactly 56 features with representative keys across all 10 families."""
        ctx = SignalContext(realistic_signal_event)
        registry = build_default_feature_registry()
        aggregator = FeatureAggregator(registry)

        vector = aggregator.extract(ctx)

        assert isinstance(vector, FeatureVector)
        assert vector.event_id == realistic_signal_event.event_id
        assert vector.feature_schema_version == "1.0"

        # Regression check: version-1.0 default registry must yield exactly 56 features
        assert len(vector.features) == 56

        # Representative keys from each of the 10 families must be present
        representative_keys = {
            "amplitude": "amplitude.mean",
            "power": "power.mean_dbfs",
            "temporal": "temporal.duration_s",
            "frequency": "frequency.center_offset_hz",
            "bandwidth": "bandwidth.occupied_99_hz",
            "spectral": "spectral.entropy",
            "phase": "phase.circular_variance",
            "statistics": "stats.amplitude.skewness",
            "higher_order": "higher_order.normalized_abs_c42",
            "correlation": "correlation.nonzero_peak_magnitude",
        }

        for family_name, rep_key in representative_keys.items():
            assert rep_key in vector.features, f"Missing representative key '{rep_key}' from family '{family_name}'"
            fv = vector.features[rep_key]
            assert isinstance(fv, FeatureValue)
            assert fv.valid is True
            assert isinstance(fv.value, float)

        # Ensure all 56 feature values are Python floats
        for key, fv in vector.features.items():
            assert isinstance(fv, FeatureValue)
            assert type(fv.value) is float, f"Feature '{key}' value has type {type(fv.value)}"

    def test_repeated_aggregation_deterministic(
        self, realistic_signal_event: SignalEvent
    ) -> None:
        """Repeated aggregation on the same context produces identical FeatureVectors."""
        ctx = SignalContext(realistic_signal_event)
        aggregator = FeatureAggregator(build_default_feature_registry())

        v1 = aggregator.extract(ctx)
        v2 = aggregator.extract(ctx)

        assert len(v1.features) == len(v2.features)
        for key in v1.features:
            assert v1.features[key].value == v2.features[key].value
            assert v1.features[key].valid == v2.features[key].valid
            assert v1.features[key].unit == v2.features[key].unit

    def test_no_signal_event_mutation(
        self, realistic_signal_event: SignalEvent
    ) -> None:
        """Aggregation leaves SignalEvent and SignalContext completely unmodified."""
        orig_iq = realistic_signal_event.iq.copy()
        orig_sig_iq = realistic_signal_event.signal_iq.copy()

        ctx = SignalContext(realistic_signal_event)
        aggregator = FeatureAggregator(build_default_feature_registry())
        _ = aggregator.extract(ctx)

        np.testing.assert_array_equal(realistic_signal_event.iq, orig_iq)
        np.testing.assert_array_equal(realistic_signal_event.signal_iq, orig_sig_iq)
        assert realistic_signal_event.iq.flags.writeable is False
        assert realistic_signal_event.signal_iq.flags.writeable is False
