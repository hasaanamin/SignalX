"""Architectural tests verifying separation of concerns, metadata-only extraction, and zero DSP execution."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.temporal import TemporalFeatureExtractor


class PureMetadataContext:
    """Mock context providing only timing metadata to prove no IQ or DSP transforms are required."""

    def __init__(self, num_samples: int, duration_s: float) -> None:
        self.signal_num_samples = num_samples
        self.signal_duration_s = duration_s


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a standard SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-arch-temporal",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(500)
    iq = np.exp(1j * 2.0 * np.pi * 100_000.0 * n / 1_000_000.0).astype(np.complex64)

    return SignalEvent(
        event_id="ev-arch-temporal-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=500,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestArchitectureAdherence:
    """Verify clean architectural decoupling and zero DSP overhead in temporal feature extraction."""

    def test_no_raw_iq_or_dsp_dependency(self) -> None:
        """Extractor must extract features using purely metadata without IQ or transform methods."""
        mock_ctx = PureMetadataContext(num_samples=400, duration_s=0.0004)
        extractor = TemporalFeatureExtractor()

        fset = extractor.extract(mock_ctx)

        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 2
        assert fset.features["temporal.duration_samples"].value == 400.0
        assert fset.features["temporal.duration_s"].value == 0.0004

    def test_zero_dsp_transforms_triggered(self, sample_event: SignalEvent) -> None:
        """Temporal feature extraction must trigger ZERO DSP transforms on SignalContext."""
        ctx = SignalContext(sample_event)
        extractor = TemporalFeatureExtractor()

        assert ctx.cached_keys == frozenset()

        _ = extractor.extract(ctx)

        # Transform cache must remain completely empty
        assert ctx.cached_keys == frozenset()
        assert "envelope" not in ctx.cached_keys
        assert "power" not in ctx.cached_keys
        assert "fft" not in ctx.cached_keys
        assert "psd" not in ctx.cached_keys
        assert "phase" not in ctx.cached_keys
        assert "instantaneous_frequency" not in ctx.cached_keys
        assert "autocorrelation" not in ctx.cached_keys

    def test_no_context_mutation(self, sample_event: SignalEvent) -> None:
        """Extractor must not alter SignalContext attributes or event arrays."""
        ctx = SignalContext(sample_event)
        orig_sig_iq = sample_event.signal_iq.copy()
        orig_event_iq = sample_event.iq.copy()

        extractor = TemporalFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(sample_event.signal_iq, orig_sig_iq)
        np.testing.assert_array_equal(sample_event.iq, orig_event_iq)
        assert sample_event.signal_iq.flags.writeable is False
        assert sample_event.iq.flags.writeable is False

    def test_no_capture_or_detector_metrics_in_featureset(self, sample_event: SignalEvent) -> None:
        """Ensure system/capture/detector metrics are strictly excluded from the FeatureSet."""
        ctx = SignalContext(sample_event)
        extractor = TemporalFeatureExtractor()

        fset = extractor.extract(ctx)

        # Only intrinsic RF signal temporal features allowed
        allowed_keys = {
            "temporal.duration_samples",
            "temporal.duration_s",
        }
        assert set(fset.features.keys()) == allowed_keys

        # Explicitly verify excluded system/buffer/detector metrics
        forbidden_keys = [
            "temporal.absolute_start_sample",
            "temporal.absolute_end_sample",
            "temporal.timestamp_ns",
            "temporal.trigger_sample",
            "temporal.pre_trigger_samples",
            "temporal.pre_trigger_s",
            "temporal.post_trigger_samples",
            "temporal.post_trigger_s",
            "temporal.trigger_delay",
            "temporal.duty_cycle",
            "temporal.pri",
            "temporal.prf",
            "temporal.rise_time",
            "temporal.fall_time",
        ]
        for key in forbidden_keys:
            assert key not in fset.features
