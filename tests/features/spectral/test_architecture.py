"""Architectural tests verifying separation of concerns, PSD-only consumption, and transform isolation."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, PSDResult, SignalEvent, StreamInfo
from signal_processing.features.spectral import (
    SpectralFeatureError,
    SpectralFeatureExtractor,
)


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
        stream_id="stream-arch-spec",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(500)
    iq = np.exp(1j * 2.0 * np.pi * 100_000.0 * n / 1_000_000.0).astype(np.complex64)

    return SignalEvent(
        event_id="ev-arch-spec-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=500,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestArchitectureAdherence:
    """Verify clean architectural decoupling between spectral feature extraction and DSP transforms."""

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

        extractor = SpectralFeatureExtractor()
        fset = extractor.extract(mock_ctx)

        assert mock_ctx.psd_called is True
        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 4

    def test_only_psd_dependency_chain_triggered(self, sample_event: SignalEvent) -> None:
        """Spectral feature extraction must trigger ONLY context.psd() and its internal FFT dependency."""
        ctx = SignalContext(sample_event)
        extractor = SpectralFeatureExtractor()

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

        extractor = SpectralFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(sample_event.signal_iq, orig_sig_iq)
        np.testing.assert_array_equal(sample_event.iq, orig_event_iq)
        assert sample_event.signal_iq.flags.writeable is False
        assert sample_event.iq.flags.writeable is False

    def test_no_psd_array_mutation(self) -> None:
        """Extractor must not mutate PSDResult frequency or power arrays."""
        freqs = np.array([100.0, -100.0, 0.0], dtype=np.float64)
        powers = np.array([2.0, 1.0, 3.0], dtype=np.float64)
        psd_obj = PSDResult(
            frequency_offset_hz=freqs,
            power_linear=powers,
            power_dbfs_per_hz=np.full_like(powers, -100.0),
        )
        ctx = ContextWithoutRawIq(psd_obj)

        freqs_copy = freqs.copy()
        powers_copy = powers.copy()

        extractor = SpectralFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(psd_obj.frequency_offset_hz, freqs_copy)
        np.testing.assert_array_equal(psd_obj.power_linear, powers_copy)

    def test_no_frequency_bandwidth_or_spread_features(self, sample_event: SignalEvent) -> None:
        """Ensure frequency centroid, bandwidth, and spectral spread features are excluded."""
        ctx = SignalContext(sample_event)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)

        allowed_keys = {
            "spectral.entropy",
            "spectral.flatness",
            "spectral.rolloff_85_offset_hz",
            "spectral.rolloff_95_offset_hz",
        }
        assert set(fset.features.keys()) == allowed_keys

        forbidden_keys = [
            "frequency.peak_offset_hz",
            "frequency.center_offset_hz",
            "spectral.centroid_hz",
            "spectral.spread",
            "spectral.spread_hz",
            "spectral.skewness",
            "spectral.kurtosis",
            "bandwidth.occupied_99_hz",
            "bandwidth.three_db_hz",
        ]
        for key in forbidden_keys:
            assert key not in fset.features

    def test_reject_invalid_context_and_psd(self) -> None:
        """Verify SpectralFeatureError is raised for invalid contexts and malformed PSDs."""
        extractor = SpectralFeatureExtractor()

        with pytest.raises(SpectralFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(SpectralFeatureError, match="Expected SignalContext"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

        class BrokenContext:
            def psd(self) -> PSDResult:
                raise RuntimeError("Fail")

        with pytest.raises(SpectralFeatureError, match="Failed to obtain PSD"):
            extractor.extract(BrokenContext())

        class MissingAttrPSDContext:
            def psd(self) -> object:
                return object()

        with pytest.raises(SpectralFeatureError, match="PSDResult must contain"):
            extractor.extract(MissingAttrPSDContext())

        class EmptyPSD:
            frequency_offset_hz = np.array([], dtype=np.float64)
            power_linear = np.array([], dtype=np.float64)

        class EmptyContext:
            def psd(self) -> EmptyPSD:
                return EmptyPSD()

        with pytest.raises(SpectralFeatureError, match="must not be empty"):
            extractor.extract(EmptyContext())

        class MultiDimPSD:
            frequency_offset_hz = np.zeros((2, 2))
            power_linear = np.zeros((2, 2))

        class MultiDimContext:
            def psd(self) -> MultiDimPSD:
                return MultiDimPSD()

        with pytest.raises(SpectralFeatureError, match="1-dimensional"):
            extractor.extract(MultiDimContext())

        class LengthMismatchPSD:
            frequency_offset_hz = np.array([100.0, 200.0])
            power_linear = np.array([1.0])

        class LengthMismatchContext:
            def psd(self) -> LengthMismatchPSD:
                return LengthMismatchPSD()

        with pytest.raises(SpectralFeatureError, match="length mismatch"):
            extractor.extract(LengthMismatchContext())

        class NegativePowerPSD:
            frequency_offset_hz = np.array([100.0, 200.0])
            power_linear = np.array([1.0, -0.5])

        class NegativePowerContext:
            def psd(self) -> NegativePowerPSD:
                return NegativePowerPSD()

        with pytest.raises(SpectralFeatureError, match="non-negative values"):
            extractor.extract(NegativePowerContext())
