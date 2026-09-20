"""Architectural tests verifying separation of concerns, direct signal_iq consumption, and transform isolation."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.higher_order import (
    HigherOrderFeatureError,
    HigherOrderFeatureExtractor,
)


class ContextWithOnlySignalIq:
    """Mock context providing only signal_iq to prove direct complex-IQ consumption without extraneous transforms."""

    def __init__(self, signal_iq: np.ndarray) -> None:
        self.signal_iq = signal_iq


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a standard SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-arch-hos",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(500)
    iq = np.exp(1j * 2.0 * np.pi * 50_000.0 * n / 1_000_000.0).astype(np.complex64)

    return SignalEvent(
        event_id="ev-arch-hos-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=500,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestArchitectureAdherence:
    """Verify clean architectural decoupling between higher-order feature extraction and DSP transforms."""

    def test_minimal_context_with_only_signal_iq(self) -> None:
        """Extractor must consume context.signal_iq without calling any transform methods."""
        iq = np.array([1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j, -1.0 + 1.0j], dtype=np.complex128)
        mock_ctx = ContextWithOnlySignalIq(iq)

        extractor = HigherOrderFeatureExtractor()
        fset = extractor.extract(mock_ctx)

        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 6

    def test_no_dsp_transforms_triggered(self, sample_event: SignalEvent) -> None:
        """Higher-order feature extraction must NOT trigger any transforms in SignalContext."""
        ctx = SignalContext(sample_event)
        extractor = HigherOrderFeatureExtractor()

        assert ctx.cached_keys == frozenset()

        _ = extractor.extract(ctx)

        # No transforms should have been computed or cached
        assert ctx.cached_keys == frozenset()

    def test_no_context_mutation(self, sample_event: SignalEvent) -> None:
        """Extractor must not alter SignalContext attributes or event arrays."""
        ctx = SignalContext(sample_event)
        orig_sig_iq = sample_event.signal_iq.copy()
        orig_event_iq = sample_event.iq.copy()

        extractor = HigherOrderFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(sample_event.signal_iq, orig_sig_iq)
        np.testing.assert_array_equal(sample_event.iq, orig_event_iq)
        assert sample_event.signal_iq.flags.writeable is False
        assert sample_event.iq.flags.writeable is False

    def test_exact_feature_keys_and_no_raw_or_classification_features(self, sample_event: SignalEvent) -> None:
        """Ensure only normalized absolute cumulants are exposed (no raw cumulants, no modulation labels)."""
        ctx = SignalContext(sample_event)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        allowed_keys = {
            "higher_order.normalized_abs_c20",
            "higher_order.normalized_abs_c40",
            "higher_order.normalized_abs_c41",
            "higher_order.normalized_abs_c42",
            "higher_order.normalized_abs_c60",
            "higher_order.normalized_abs_c63",
        }
        assert set(fset.features.keys()) == allowed_keys

        forbidden_keys = [
            "higher_order.c20",
            "higher_order.c21",
            "higher_order.c40",
            "higher_order.c41",
            "higher_order.c42",
            "higher_order.c60",
            "higher_order.c63",
            "higher_order.normalized_abs_c21",
            "modulation",
            "classification",
            "modulation_type",
            "c40.real",
            "c40.imag",
        ]
        for key in forbidden_keys:
            assert key not in fset.features

    def test_reject_invalid_context_and_iq(self) -> None:
        """Verify HigherOrderFeatureError is raised for invalid contexts and malformed IQ data."""
        extractor = HigherOrderFeatureExtractor()

        with pytest.raises(HigherOrderFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(HigherOrderFeatureError, match="Expected SignalContext"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

        class MissingAttrContext:
            pass

        with pytest.raises(HigherOrderFeatureError, match="Expected SignalContext"):
            extractor.extract(MissingAttrContext())

        class NoneIqContext:
            signal_iq = None

        with pytest.raises(HigherOrderFeatureError, match="signal_iq must not be None"):
            extractor.extract(NoneIqContext())

        class EmptyIqContext:
            signal_iq = np.array([], dtype=np.complex128)

        with pytest.raises(HigherOrderFeatureError, match="non-empty 1-dimensional"):
            extractor.extract(EmptyIqContext())

        class MultiDimIqContext:
            signal_iq = np.zeros((2, 2), dtype=np.complex128)

        with pytest.raises(HigherOrderFeatureError, match="non-empty 1-dimensional"):
            extractor.extract(MultiDimIqContext())

        class NonFiniteIqContext:
            signal_iq = np.array([1.0 + 1.0j, complex(float("nan"), 0.0)], dtype=np.complex128)

        with pytest.raises(HigherOrderFeatureError, match="non-finite values"):
            extractor.extract(NonFiniteIqContext())
