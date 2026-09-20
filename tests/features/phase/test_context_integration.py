"""Integration tests connecting SignalEvent -> SignalContext -> PhaseFeatureExtractor."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.phase import PhaseFeatureExtractor


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a standard 1 MHz multi-tone pulse event."""
    stream = StreamInfo(
        stream_id="stream-phase-int",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(400)
    iq = (
        0.7 * np.exp(1j * 2.0 * np.pi * 50_000.0 * n / 1_000_000.0)
        + 0.3 * np.exp(1j * 2.0 * np.pi * 120_000.0 * n / 1_000_000.0)
    ).astype(np.complex64)

    return SignalEvent(
        event_id="ev-phase-int-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=400,
        detected_start_sample=20,
        detected_end_sample=380,
        trigger_sample=50,
    )


class TestContextIntegration:
    """Validate integration with real SignalEvent and SignalContext objects."""

    def test_real_signal_context_integration(self, sample_event: SignalEvent) -> None:
        """Verify feature extraction on real SignalContext produces all 6 valid features."""
        ctx = SignalContext(sample_event)
        extractor = PhaseFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "phase"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 6

        for fv in fset.features.values():
            assert fv.valid is True
            assert isinstance(fv.value, float)

    def test_cache_reuse_and_immutability(self, sample_event: SignalEvent) -> None:
        """Verify context.phase() is cached and reused, and arrays remain read-only."""
        ctx = SignalContext(sample_event)
        extractor = PhaseFeatureExtractor()

        assert "phase" not in ctx.cached_keys

        fset1 = extractor.extract(ctx)

        assert "phase" in ctx.cached_keys
        phase_cached = ctx.phase()

        fset2 = extractor.extract(ctx)

        # Ensure PhaseResult object was reused from cache
        assert ctx.phase() is phase_cached

        for key in fset1.features:
            assert fset1.features[key].value == fset2.features[key].value

        # Immutability verification
        assert sample_event.iq.flags.writeable is False
        assert sample_event.signal_iq.flags.writeable is False
        assert phase_cached.wrapped_rad.flags.writeable is False
        assert phase_cached.unwrapped_rad.flags.writeable is False
