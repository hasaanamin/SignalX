"""Architectural tests verifying separation of concerns, transform isolation, and DSP non-duplication."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.amplitude import AmplitudeFeatureExtractor


class ContextWithoutSignalIq:
    """Mock context providing only envelope() to prove no direct raw-IQ access occurs."""

    def __init__(self, env: np.ndarray) -> None:
        self._env = env
        self.envelope_called = False

    def envelope(self) -> np.ndarray:
        self.envelope_called = True
        return self._env


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a standard SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-arch-test",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(500)
    iq = np.exp(1j * 2.0 * np.pi * 100_000.0 * n / 1_000_000.0).astype(np.complex64)

    return SignalEvent(
        event_id="ev-arch-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=500,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestArchitectureAdherence:
    """Verify clean architectural decoupling between feature extraction and DSP transforms."""

    def test_no_raw_iq_dependency(self) -> None:
        """Extractor must consume envelope() without requiring access to context.signal_iq."""
        env_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        mock_ctx = ContextWithoutSignalIq(env_data)

        extractor = AmplitudeFeatureExtractor()
        fset = extractor.extract(mock_ctx)

        assert mock_ctx.envelope_called is True
        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 9

    def test_only_envelope_transform_triggered(self, sample_event: SignalEvent) -> None:
        """Amplitude extraction must trigger ONLY ctx.envelope() and no other DSP transforms."""
        ctx = SignalContext(sample_event)
        extractor = AmplitudeFeatureExtractor()

        assert ctx.cached_keys == frozenset()

        _ = extractor.extract(ctx)

        # Only 'envelope' must be cached
        assert ctx.cached_keys == frozenset(["envelope"])
        assert "fft" not in ctx.cached_keys
        assert "psd" not in ctx.cached_keys
        assert "phase" not in ctx.cached_keys
        assert "power" not in ctx.cached_keys
        assert "instantaneous_frequency" not in ctx.cached_keys
        assert "autocorrelation" not in ctx.cached_keys

    def test_no_context_mutation(self, sample_event: SignalEvent) -> None:
        """Extractor must not alter SignalContext attributes or event arrays."""
        ctx = SignalContext(sample_event)
        orig_sig_iq = sample_event.signal_iq.copy()
        orig_event_iq = sample_event.iq.copy()

        extractor = AmplitudeFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(sample_event.signal_iq, orig_sig_iq)
        np.testing.assert_array_equal(sample_event.iq, orig_event_iq)
        assert sample_event.signal_iq.flags.writeable is False
        assert sample_event.iq.flags.writeable is False
