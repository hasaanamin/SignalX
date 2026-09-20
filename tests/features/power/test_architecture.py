"""Architectural tests verifying separation of concerns, transform isolation, and DSP non-duplication."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.power import PowerFeatureExtractor


class ContextWithoutSignalIq:
    """Mock context providing only power-domain methods to prove no direct raw-IQ access occurs."""

    def __init__(self, power_arr: np.ndarray) -> None:
        self._power = power_arr
        self.power_called = False

    def power(self) -> np.ndarray:
        self.power_called = True
        return self._power

    def mean_power_linear(self) -> float:
        return float(np.mean(self._power))

    def mean_power_dbfs(self) -> float:
        return float(10.0 * np.log10(np.mean(self._power)))


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
        event_id="ev-arch-power-001",
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
        """Extractor must consume power methods without requiring access to context.signal_iq."""
        power_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        mock_ctx = ContextWithoutSignalIq(power_data)

        extractor = PowerFeatureExtractor()
        fset = extractor.extract(mock_ctx)

        assert mock_ctx.power_called is True
        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 10

    def test_only_power_transforms_triggered(self, sample_event: SignalEvent) -> None:
        """Power extraction must trigger ONLY power methods and no other DSP transforms."""
        ctx = SignalContext(sample_event)
        extractor = PowerFeatureExtractor()

        assert ctx.cached_keys == frozenset()

        _ = extractor.extract(ctx)

        # Only power-domain keys must be cached in SignalContext
        for key in ctx.cached_keys:
            assert key in {"power", "mean_power_linear", "mean_power_dbfs"}

        assert "fft" not in ctx.cached_keys
        assert "psd" not in ctx.cached_keys
        assert "phase" not in ctx.cached_keys
        assert "envelope" not in ctx.cached_keys
        assert "instantaneous_frequency" not in ctx.cached_keys
        assert "autocorrelation" not in ctx.cached_keys

    def test_no_context_mutation(self, sample_event: SignalEvent) -> None:
        """Extractor must not alter SignalContext attributes or event arrays."""
        ctx = SignalContext(sample_event)
        orig_sig_iq = sample_event.signal_iq.copy()
        orig_event_iq = sample_event.iq.copy()

        extractor = PowerFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(sample_event.signal_iq, orig_sig_iq)
        np.testing.assert_array_equal(sample_event.iq, orig_event_iq)
        assert sample_event.signal_iq.flags.writeable is False
        assert sample_event.iq.flags.writeable is False
