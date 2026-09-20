"""Architectural tests verifying separation of concerns, envelope-only consumption, and transform isolation."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.statistics import (
    StatisticsFeatureError,
    StatisticsFeatureExtractor,
)


class ContextWithoutRawIq:
    """Mock context providing only envelope() to prove no direct raw-IQ or extraneous transform access occurs."""

    def __init__(self, envelope: np.ndarray) -> None:
        self._envelope = envelope
        self.envelope_called = False

    def envelope(self) -> np.ndarray:
        self.envelope_called = True
        return self._envelope


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a standard SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-arch-stat",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(500)
    iq = (
        (1.0 + 0.3 * np.cos(2.0 * np.pi * 5_000.0 * n / 1_000_000.0))
        * np.exp(1j * 2.0 * np.pi * 100_000.0 * n / 1_000_000.0)
    ).astype(np.complex64)

    return SignalEvent(
        event_id="ev-arch-stat-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=500,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestArchitectureAdherence:
    """Verify clean architectural decoupling between statistics feature extraction and DSP transforms."""

    def test_no_raw_iq_dependency(self) -> None:
        """Extractor must consume envelope() without requiring access to context.signal_iq or raw transforms."""
        env = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mock_ctx = ContextWithoutRawIq(env)

        extractor = StatisticsFeatureExtractor()
        fset = extractor.extract(mock_ctx)

        assert mock_ctx.envelope_called is True
        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 9

    def test_only_envelope_dependency_triggered(self, sample_event: SignalEvent) -> None:
        """Statistics feature extraction must trigger ONLY context.envelope()."""
        ctx = SignalContext(sample_event)
        extractor = StatisticsFeatureExtractor()

        assert ctx.cached_keys == frozenset()

        _ = extractor.extract(ctx)

        # SignalContext computes only envelope
        assert ctx.cached_keys == frozenset(["envelope"])

        # Other DSP transforms must strictly NOT be computed
        assert "power" not in ctx.cached_keys
        assert "phase" not in ctx.cached_keys
        assert "fft" not in ctx.cached_keys
        assert "psd" not in ctx.cached_keys
        assert "instantaneous_frequency" not in ctx.cached_keys
        assert "autocorrelation" not in ctx.cached_keys

    def test_no_context_mutation(self, sample_event: SignalEvent) -> None:
        """Extractor must not alter SignalContext attributes or event arrays."""
        ctx = SignalContext(sample_event)
        orig_sig_iq = sample_event.signal_iq.copy()
        orig_event_iq = sample_event.iq.copy()

        extractor = StatisticsFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(sample_event.signal_iq, orig_sig_iq)
        np.testing.assert_array_equal(sample_event.iq, orig_event_iq)
        assert sample_event.signal_iq.flags.writeable is False
        assert sample_event.iq.flags.writeable is False

    def test_no_envelope_mutation(self) -> None:
        """Extractor must not mutate the envelope array."""
        env = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        env_copy = env.copy()
        ctx = ContextWithoutRawIq(env)

        extractor = StatisticsFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(env, env_copy)

    def test_no_duplicate_or_forbidden_features(self, sample_event: SignalEvent) -> None:
        """Ensure no duplicate amplitude, entropy, correlation, or higher-order cumulants are output."""
        ctx = SignalContext(sample_event)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)

        allowed_keys = {
            "stats.amplitude.skewness",
            "stats.amplitude.excess_kurtosis",
            "stats.amplitude.q01",
            "stats.amplitude.q05",
            "stats.amplitude.q25",
            "stats.amplitude.q75",
            "stats.amplitude.q95",
            "stats.amplitude.q99",
            "stats.amplitude.iqr",
        }
        assert set(fset.features.keys()) == allowed_keys

        forbidden_keys = [
            "stats.amplitude.mean",
            "stats.amplitude.variance",
            "stats.amplitude.std",
            "stats.amplitude.median",
            "stats.amplitude.min",
            "stats.amplitude.max",
            "stats.amplitude.q50",
            "stats.amplitude.entropy",
            "spectral.entropy",
            "stats.amplitude.covariance",
            "stats.amplitude.correlation",
            "stats.c20",
            "stats.c21",
            "stats.c40",
            "stats.c41",
            "stats.c42",
        ]
        for key in forbidden_keys:
            assert key not in fset.features

    def test_reject_invalid_context_and_envelope(self) -> None:
        """Verify StatisticsFeatureError is raised for invalid contexts and malformed envelope data."""
        extractor = StatisticsFeatureExtractor()

        with pytest.raises(StatisticsFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(StatisticsFeatureError, match="Expected SignalContext"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

        class BrokenContext:
            def envelope(self) -> np.ndarray:
                raise RuntimeError("Hardware failure")

        with pytest.raises(StatisticsFeatureError, match="Failed to obtain envelope"):
            extractor.extract(BrokenContext())

        class EmptyEnvelopeContext:
            def envelope(self) -> np.ndarray:
                return np.array([], dtype=np.float64)

        with pytest.raises(StatisticsFeatureError, match="non-empty 1-dimensional"):
            extractor.extract(EmptyEnvelopeContext())

        class MultiDimEnvelopeContext:
            def envelope(self) -> np.ndarray:
                return np.zeros((2, 2), dtype=np.float64)

        with pytest.raises(StatisticsFeatureError, match="non-empty 1-dimensional"):
            extractor.extract(MultiDimEnvelopeContext())

        class ComplexEnvelopeContext:
            def envelope(self) -> np.ndarray:
                return np.array([1.0 + 1j, 2.0], dtype=np.complex128)

        with pytest.raises(StatisticsFeatureError, match="real-valued numeric"):
            extractor.extract(ComplexEnvelopeContext())

        class NonFiniteEnvelopeContext:
            def envelope(self) -> np.ndarray:
                return np.array([1.0, float("nan"), 3.0], dtype=np.float64)

        with pytest.raises(StatisticsFeatureError, match="non-finite values"):
            extractor.extract(NonFiniteEnvelopeContext())

        class NegativeEnvelopeContext:
            def envelope(self) -> np.ndarray:
                return np.array([1.0, -2.0, 3.0], dtype=np.float64)

        with pytest.raises(StatisticsFeatureError, match="negative values"):
            extractor.extract(NegativeEnvelopeContext())
