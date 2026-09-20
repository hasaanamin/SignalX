"""Architectural tests verifying separation of concerns, PSD-only consumption, and transform isolation."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, PSDResult, SignalEvent, StreamInfo
from signal_processing.features.bandwidth import BandwidthFeatureExtractor


class ContextWithoutRawIq:
    """Mock context providing only psd() to prove no direct raw-IQ or extraneous transform access occurs."""

    def __init__(self, psd_result: PSDResult) -> None:
        self._psd = psd_result
        self.psd_called = False

    def psd(self) -> PSDResult:
        self.psd_called = True
        return self._psd


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a standard SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-arch-bw",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(500)
    iq = np.exp(1j * 2.0 * np.pi * 100_000.0 * n / 1_000_000.0).astype(np.complex64)

    return SignalEvent(
        event_id="ev-arch-bw-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=500,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestArchitectureAdherence:
    """Verify clean architectural decoupling between bandwidth feature extraction and DSP transforms."""

    def test_no_raw_iq_dependency(self) -> None:
        """Extractor must consume psd() without requiring access to context.signal_iq or raw transforms."""
        freqs = np.array([-100.0, 0.0, 100.0], dtype=np.float64)
        powers = np.array([1.0, 5.0, 1.0], dtype=np.float64)
        psd_obj = PSDResult(
            frequency_offset_hz=freqs,
            power_linear=powers,
            power_dbfs_per_hz=np.array([-10.0, 0.0, -10.0], dtype=np.float64),
        )
        mock_ctx = ContextWithoutRawIq(psd_obj)

        extractor = BandwidthFeatureExtractor()
        fset = extractor.extract(mock_ctx)

        assert mock_ctx.psd_called is True
        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 2
        assert fset.features["bandwidth.occupied_99_hz"].value == 300.0
        assert fset.features["bandwidth.three_db_hz"].value == 100.0

    def test_only_psd_dependency_chain_triggered(self, sample_event: SignalEvent) -> None:
        """Bandwidth feature extraction must trigger ONLY context.psd() and its internal FFT dependency."""
        ctx = SignalContext(sample_event)
        extractor = BandwidthFeatureExtractor()

        assert ctx.cached_keys == frozenset()

        _ = extractor.extract(ctx)

        # SignalContext computes FFT to calculate PSD; both are cached
        assert ctx.cached_keys == frozenset(["fft", "psd"])

        # Other DSP transforms must strictly NOT be computed
        assert "envelope" not in ctx.cached_keys
        assert "power" not in ctx.cached_keys
        assert "phase" not in ctx.cached_keys
        assert "instantaneous_frequency" not in ctx.cached_keys
        assert "autocorrelation" not in ctx.cached_keys

    def test_no_context_mutation(self, sample_event: SignalEvent) -> None:
        """Extractor must not alter SignalContext attributes or event arrays."""
        ctx = SignalContext(sample_event)
        orig_sig_iq = sample_event.signal_iq.copy()
        orig_event_iq = sample_event.iq.copy()

        extractor = BandwidthFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(sample_event.signal_iq, orig_sig_iq)
        np.testing.assert_array_equal(sample_event.iq, orig_event_iq)
        assert sample_event.signal_iq.flags.writeable is False
        assert sample_event.iq.flags.writeable is False

    def test_no_frequency_or_spectral_shape_features(self, sample_event: SignalEvent) -> None:
        """Ensure frequency center and spectral shape features are strictly excluded from the FeatureSet."""
        ctx = SignalContext(sample_event)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        allowed_keys = {
            "bandwidth.occupied_99_hz",
            "bandwidth.three_db_hz",
        }
        assert set(fset.features.keys()) == allowed_keys

        forbidden_keys = [
            "frequency.peak_offset_hz",
            "frequency.center_offset_hz",
            "spectral.spread",
            "spectral.entropy",
            "spectral.flatness",
            "spectral.rolloff",
            "spectral.kurtosis",
        ]
        for key in forbidden_keys:
            assert key not in fset.features
