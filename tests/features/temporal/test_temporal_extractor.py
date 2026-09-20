"""Tests for TemporalFeatureExtractor mathematical accuracy, units, and error handling."""

import pytest

from signal_processing.core import FeatureSet
from signal_processing.features.temporal import (
    TemporalFeatureError,
    TemporalFeatureExtractor,
)


class DummyTemporalContext:
    """Minimal duck-typed context providing temporal metadata."""

    def __init__(
        self,
        signal_num_samples: float,
        signal_duration_s: float | None = None,
        sample_rate_hz: float | None = None,
    ) -> None:
        self.signal_num_samples = signal_num_samples
        if signal_duration_s is not None:
            self.signal_duration_s = signal_duration_s
        if sample_rate_hz is not None:
            self.sample_rate_hz = sample_rate_hz



class TestTemporalExtractor:
    """Test temporal feature extraction on synthetic timing configurations."""

    def test_case_a_basic_event(self) -> None:
        """Test Case A: 1000 samples at 1 MHz produces exact 1000 samples and 0.001 s."""
        ctx = DummyTemporalContext(
            signal_num_samples=1000,
            signal_duration_s=0.001,
        )
        extractor = TemporalFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "temporal"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 2

        # 1. duration_samples
        fv_samples = fset.features["temporal.duration_samples"]
        assert fv_samples.value == 1000.0
        assert fv_samples.unit == "samples"
        assert fv_samples.valid is True
        assert fv_samples.reason is None
        assert isinstance(fv_samples.value, float)
        assert type(fv_samples.value) is float

        # 2. duration_s
        fv_s = fset.features["temporal.duration_s"]
        assert pytest.approx(fv_s.value) == 0.001
        assert fv_s.unit == "s"
        assert fv_s.valid is True
        assert fv_s.reason is None
        assert isinstance(fv_s.value, float)
        assert type(fv_s.value) is float

    def test_case_b_different_sample_rate(self) -> None:
        """Test Case B: 500 samples at 2 MHz produces 0.00025 s."""
        ctx = DummyTemporalContext(
            signal_num_samples=500,
            sample_rate_hz=2_000_000.0,
        )
        extractor = TemporalFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["temporal.duration_samples"].value == 500.0
        assert pytest.approx(fset.features["temporal.duration_s"].value) == 0.00025

    def test_case_c_minimum_valid_signal(self) -> None:
        """Test Case C: 1-sample event at 1 MHz produces 1.0 sample and 1e-6 s."""
        ctx = DummyTemporalContext(
            signal_num_samples=1,
            sample_rate_hz=1_000_000.0,
        )
        extractor = TemporalFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["temporal.duration_samples"].value == 1.0
        assert pytest.approx(fset.features["temporal.duration_s"].value) == 1e-6

    def test_repeated_extraction_statelessness(self) -> None:
        """Repeated extraction on the same context produces identical FeatureSets."""
        ctx = DummyTemporalContext(signal_num_samples=750, signal_duration_s=0.00075)
        extractor = TemporalFeatureExtractor()

        fset1 = extractor.extract(ctx)
        fset2 = extractor.extract(ctx)

        for k in fset1.features:
            assert fset1.features[k].value == fset2.features[k].value
            assert fset1.features[k].unit == fset2.features[k].unit
            assert fset1.features[k].valid == fset2.features[k].valid
            assert fset1.features[k].reason == fset2.features[k].reason

    def test_duck_typed_event_access(self) -> None:
        """Verify extraction from a context exposing event with signal_samples and signal_duration_s."""
        class MockEvent:
            signal_samples = 300
            signal_duration_s = 0.0003

        class MockEventContext:
            event = MockEvent()

        extractor = TemporalFeatureExtractor()
        fset = extractor.extract(MockEventContext())

        assert fset.features["temporal.duration_samples"].value == 300.0
        assert pytest.approx(fset.features["temporal.duration_s"].value) == 0.0003

    def test_duck_typed_event_stream_sample_rate_fallback(self) -> None:
        """Verify extraction computes duration_s from event.stream.sample_rate_hz if duration_s missing."""
        class MockStream:
            sample_rate_hz = 500_000.0

        class MockEvent:
            signal_samples = 250
            stream = MockStream()

        class MockEventContext:
            event = MockEvent()

        extractor = TemporalFeatureExtractor()
        fset = extractor.extract(MockEventContext())

        assert fset.features["temporal.duration_samples"].value == 250.0
        assert pytest.approx(fset.features["temporal.duration_s"].value) == 0.0005

    def test_duck_typed_callable_properties(self) -> None:
        """Verify extraction handles callable methods for signal metadata."""
        class CallableContext:
            def signal_num_samples(self) -> int:
                return 400

            def signal_duration_s(self) -> float:
                return 0.0004

        extractor = TemporalFeatureExtractor()
        fset = extractor.extract(CallableContext())

        assert fset.features["temporal.duration_samples"].value == 400.0
        assert pytest.approx(fset.features["temporal.duration_s"].value) == 0.0004

    def test_reject_invalid_context(self) -> None:
        """Verify extractor raises TemporalFeatureError on None or unusable context."""
        extractor = TemporalFeatureExtractor()

        with pytest.raises(TemporalFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(TemporalFeatureError, match="Expected SignalContext"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

        class ContextWithoutDuration:
            signal_num_samples = 100

        with pytest.raises(TemporalFeatureError, match="Cannot determine signal duration"):
            extractor.extract(ContextWithoutDuration())

    def test_reject_invalid_samples(self) -> None:
        """Verify extractor raises TemporalFeatureError on non-numeric, zero, negative, or NaN sample counts."""
        extractor = TemporalFeatureExtractor()

        with pytest.raises(TemporalFeatureError, match="must be numeric"):
            extractor.extract(DummyTemporalContext(signal_num_samples="invalid", signal_duration_s=0.1))

        with pytest.raises(TemporalFeatureError, match="must be numeric"):
            extractor.extract(DummyTemporalContext(signal_num_samples=True, signal_duration_s=0.1))

        with pytest.raises(TemporalFeatureError, match="positive finite"):
            extractor.extract(DummyTemporalContext(signal_num_samples=0, signal_duration_s=0.1))

        with pytest.raises(TemporalFeatureError, match="positive finite"):
            extractor.extract(DummyTemporalContext(signal_num_samples=-10, signal_duration_s=0.1))

        with pytest.raises(TemporalFeatureError, match="positive finite"):
            extractor.extract(DummyTemporalContext(signal_num_samples=float("nan"), signal_duration_s=0.1))

    def test_reject_invalid_duration_s(self) -> None:
        """Verify extractor raises TemporalFeatureError on non-numeric, zero, negative, or NaN duration_s."""
        extractor = TemporalFeatureExtractor()

        with pytest.raises(TemporalFeatureError, match="must be numeric"):
            extractor.extract(DummyTemporalContext(signal_num_samples=100, signal_duration_s="bad_duration"))

        with pytest.raises(TemporalFeatureError, match="must be numeric"):
            extractor.extract(DummyTemporalContext(signal_num_samples=100, signal_duration_s=False))

        with pytest.raises(TemporalFeatureError, match="positive finite"):
            extractor.extract(DummyTemporalContext(signal_num_samples=100, signal_duration_s=0.0))

        with pytest.raises(TemporalFeatureError, match="positive finite"):
            extractor.extract(DummyTemporalContext(signal_num_samples=100, signal_duration_s=-0.05))

        with pytest.raises(TemporalFeatureError, match="positive finite"):
            extractor.extract(DummyTemporalContext(signal_num_samples=100, signal_duration_s=float("nan")))

    def test_reject_invalid_sample_rate(self) -> None:
        """Verify extractor raises TemporalFeatureError on invalid sample rates during fallback conversion."""
        extractor = TemporalFeatureExtractor()

        with pytest.raises(TemporalFeatureError, match="must be numeric"):
            extractor.extract(DummyTemporalContext(signal_num_samples=100, sample_rate_hz="bad_fs"))

        with pytest.raises(TemporalFeatureError, match="must be positive and finite"):
            extractor.extract(DummyTemporalContext(signal_num_samples=100, sample_rate_hz=0.0))

        with pytest.raises(TemporalFeatureError, match="must be positive and finite"):
            extractor.extract(DummyTemporalContext(signal_num_samples=100, sample_rate_hz=-1_000.0))

    def test_broken_property_exceptions(self) -> None:
        """Verify extractor wraps unexpected property exceptions in TemporalFeatureError."""
        extractor = TemporalFeatureExtractor()

        class BrokenSamplesContext:
            @property
            def signal_num_samples(self) -> int:
                raise RuntimeError("Hardware register read failure")

        with pytest.raises(TemporalFeatureError, match="Failed to obtain signal_num_samples"):
            extractor.extract(BrokenSamplesContext())

        class BrokenDurationContext:
            signal_num_samples = 100

            @property
            def signal_duration_s(self) -> float:
                raise RuntimeError("Timer failure")

        with pytest.raises(TemporalFeatureError, match="Failed to obtain signal_duration_s"):
            extractor.extract(BrokenDurationContext())

    def test_event_property_errors(self) -> None:
        """Verify error handling on malformed event properties."""
        extractor = TemporalFeatureExtractor()

        # Non-numeric event.signal_samples
        class BadEventSamples:
            signal_samples = "not_a_number"
        class BadEventContext:
            event = BadEventSamples()
        with pytest.raises(TemporalFeatureError, match="event.signal_samples must be numeric"):
            extractor.extract(BadEventContext())

        # Boolean event.signal_samples
        class BoolEventSamples:
            signal_samples = True
        class BoolEventContext:
            event = BoolEventSamples()
        with pytest.raises(TemporalFeatureError, match="event.signal_samples must be numeric"):
            extractor.extract(BoolEventContext())

        # Exception in event.signal_samples
        class ExplodingEventSamples:
            @property
            def signal_samples(self) -> int:
                raise RuntimeError("Event samples hardware fault")
        class ExplodingEventContext:
            event = ExplodingEventSamples()
        with pytest.raises(TemporalFeatureError, match="Failed to obtain signal_samples from context event"):
            extractor.extract(ExplodingEventContext())

        # Non-numeric event.signal_duration_s
        class BadEventDuration:
            signal_samples = 100
            signal_duration_s = "bad_s"
        class BadEventDurationContext:
            event = BadEventDuration()
        with pytest.raises(TemporalFeatureError, match="event.signal_duration_s must be numeric"):
            extractor.extract(BadEventDurationContext())

        # Boolean event.signal_duration_s
        class BoolEventDuration:
            signal_samples = 100
            signal_duration_s = True
        class BoolEventDurationContext:
            event = BoolEventDuration()
        with pytest.raises(TemporalFeatureError, match="event.signal_duration_s must be numeric"):
            extractor.extract(BoolEventDurationContext())

        # Exception in event.signal_duration_s
        class ExplodingEventDuration:
            signal_samples = 100
            @property
            def signal_duration_s(self) -> float:
                raise RuntimeError("Event duration clock fault")
        class ExplodingEventDurationContext:
            event = ExplodingEventDuration()
        with pytest.raises(TemporalFeatureError, match="Failed to obtain signal_duration_s from context event"):
            extractor.extract(ExplodingEventDurationContext())

    def test_sample_rate_fallback_errors(self) -> None:
        """Verify error handling on malformed sample rate fallback properties."""
        extractor = TemporalFeatureExtractor()

        # Exception in context.sample_rate_hz
        class ExplodingFsContext:
            signal_num_samples = 100
            @property
            def sample_rate_hz(self) -> float:
                raise RuntimeError("Oscillator read failure")
        with pytest.raises(TemporalFeatureError, match="Failed to compute duration_s from sample_rate_hz"):
            extractor.extract(ExplodingFsContext())

        # Callable sample_rate_hz returning boolean
        class CallableFsBoolContext:
            signal_num_samples = 100
            def sample_rate_hz(self) -> bool:
                return True
        with pytest.raises(TemporalFeatureError, match="sample_rate_hz must be numeric"):
            extractor.extract(CallableFsBoolContext())

        # Non-numeric stream.sample_rate_hz
        class BadStreamFs:
            sample_rate_hz = "invalid_fs"
        class BadStreamEvent:
            signal_samples = 100
            stream = BadStreamFs()
        class BadStreamContext:
            event = BadStreamEvent()
        with pytest.raises(TemporalFeatureError, match="stream.sample_rate_hz must be numeric"):
            extractor.extract(BadStreamContext())

        # Boolean stream.sample_rate_hz
        class BoolStreamFs:
            sample_rate_hz = False
        class BoolStreamEvent:
            signal_samples = 100
            stream = BoolStreamFs()
        class BoolStreamContext:
            event = BoolStreamEvent()
        with pytest.raises(TemporalFeatureError, match="stream.sample_rate_hz must be numeric"):
            extractor.extract(BoolStreamContext())

        # Non-positive stream.sample_rate_hz
        class ZeroStreamFs:
            sample_rate_hz = 0.0
        class ZeroStreamEvent:
            signal_samples = 100
            stream = ZeroStreamFs()
        class ZeroStreamContext:
            event = ZeroStreamEvent()
        with pytest.raises(TemporalFeatureError, match="sample_rate_hz must be positive and finite"):
            extractor.extract(ZeroStreamContext())

        # Exception in stream.sample_rate_hz
        class ExplodingStreamFs:
            @property
            def sample_rate_hz(self) -> float:
                raise RuntimeError("PLL lock error")
        class ExplodingStreamEvent:
            signal_samples = 100
            stream = ExplodingStreamFs()
        class ExplodingStreamContext:
            event = ExplodingStreamEvent()
        with pytest.raises(TemporalFeatureError, match="Failed to compute duration_s from stream sample_rate_hz"):
            extractor.extract(ExplodingStreamContext())

    def test_callable_event_and_stream_methods(self) -> None:
        """Verify extraction handles callable methods on event and stream objects."""
        class CallableStream:
            def sample_rate_hz(self) -> float:
                return 1_000_000.0

        class CallableEvent:
            def signal_samples(self) -> int:
                return 500

            def signal_duration_s(self) -> float:
                return 0.0005

            stream = CallableStream()

        class CallableEventContext:
            event = CallableEvent()

        extractor = TemporalFeatureExtractor()
        fset = extractor.extract(CallableEventContext())
        assert fset.features["temporal.duration_samples"].value == 500.0
        assert fset.features["temporal.duration_s"].value == 0.0005

        # Test event with callable signal_samples and stream with callable sample_rate_hz (no duration_s)
        class CallableEventNoDuration:
            def signal_samples(self) -> int:
                return 250

            stream = CallableStream()

        class ContextNoDuration:
            event = CallableEventNoDuration()

        fset_nodur = extractor.extract(ContextNoDuration())
        assert fset_nodur.features["temporal.duration_samples"].value == 250.0
        assert pytest.approx(fset_nodur.features["temporal.duration_s"].value) == 0.00025


