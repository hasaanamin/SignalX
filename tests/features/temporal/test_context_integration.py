"""Integration tests connecting SignalEvent -> SignalContext -> TemporalFeatureExtractor."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.temporal import TemporalFeatureExtractor


@pytest.fixture
def base_stream() -> StreamInfo:
    """Fixture providing standard StreamInfo at 1 MHz."""
    return StreamInfo(
        stream_id="stream-temporal-int",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


class TestContextIntegration:
    """Test temporal feature extraction on real SignalEvent and SignalContext instances."""

    def test_real_signal_context_integration(self, base_stream: StreamInfo) -> None:
        """Verify extraction on a real SignalContext wrapping a SignalEvent."""
        total_samples = 1000
        iq = np.ones(total_samples, dtype=np.complex64)

        event = SignalEvent(
            event_id="ev-temporal-001",
            stream=base_stream,
            iq=iq,
            event_start_sample=0,
            event_end_sample=total_samples,
            detected_start_sample=200,
            detected_end_sample=700,
            trigger_sample=300,
        )

        ctx = SignalContext(event)
        extractor = TemporalFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "temporal"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 2

        assert fset.features["temporal.duration_samples"].value == 500.0
        assert fset.features["temporal.duration_samples"].unit == "samples"
        assert fset.features["temporal.duration_samples"].valid is True

        assert pytest.approx(fset.features["temporal.duration_s"].value) == 0.0005
        assert fset.features["temporal.duration_s"].unit == "s"
        assert fset.features["temporal.duration_s"].valid is True

    def test_case_d_pre_post_buffer_independence(self, base_stream: StreamInfo) -> None:
        """Test Case D: Events with different pre/post capture margins but identical detected region produce identical features."""
        active_len = 400

        # Event A: Small pre/post margins (50 samples each)
        iq_a = np.ones(50 + active_len + 50, dtype=np.complex64)
        event_a = SignalEvent(
            event_id="ev-margin-small",
            stream=base_stream,
            iq=iq_a,
            event_start_sample=0,
            event_end_sample=50 + active_len + 50,
            detected_start_sample=50,
            detected_end_sample=50 + active_len,
            trigger_sample=100,
        )

        # Event B: Large pre/post margins (500 samples each)
        iq_b = np.ones(500 + active_len + 500, dtype=np.complex64)
        event_b = SignalEvent(
            event_id="ev-margin-large",
            stream=base_stream,
            iq=iq_b,
            event_start_sample=10_000,
            event_end_sample=10_000 + 500 + active_len + 500,
            detected_start_sample=10_000 + 500,
            detected_end_sample=10_000 + 500 + active_len,
            trigger_sample=10_000 + 700,
        )

        ctx_a = SignalContext(event_a)
        ctx_b = SignalContext(event_b)
        extractor = TemporalFeatureExtractor()

        fset_a = extractor.extract(ctx_a)
        fset_b = extractor.extract(ctx_b)

        assert fset_a.features["temporal.duration_samples"].value == float(active_len)
        assert fset_b.features["temporal.duration_samples"].value == float(active_len)
        assert fset_a.features["temporal.duration_samples"].value == fset_b.features["temporal.duration_samples"].value
        assert fset_a.features["temporal.duration_s"].value == fset_b.features["temporal.duration_s"].value

    def test_case_e_trigger_position_independence(self, base_stream: StreamInfo) -> None:
        """Test Case E: Events with different trigger positions inside identical detected regions produce identical features."""
        iq = np.ones(800, dtype=np.complex64)

        # Event 1: Trigger near the beginning
        event_1 = SignalEvent(
            event_id="ev-trig-early",
            stream=base_stream,
            iq=iq,
            event_start_sample=0,
            event_end_sample=800,
            detected_start_sample=100,
            detected_end_sample=700,
            trigger_sample=110,
        )

        # Event 2: Trigger near the end
        event_2 = SignalEvent(
            event_id="ev-trig-late",
            stream=base_stream,
            iq=iq,
            event_start_sample=0,
            event_end_sample=800,
            detected_start_sample=100,
            detected_end_sample=700,
            trigger_sample=690,
        )

        ctx_1 = SignalContext(event_1)
        ctx_2 = SignalContext(event_2)
        extractor = TemporalFeatureExtractor()

        fset_1 = extractor.extract(ctx_1)
        fset_2 = extractor.extract(ctx_2)

        assert fset_1.features["temporal.duration_samples"].value == 600.0
        assert fset_2.features["temporal.duration_samples"].value == 600.0
        assert fset_1.features["temporal.duration_samples"].value == fset_2.features["temporal.duration_samples"].value
        assert fset_1.features["temporal.duration_s"].value == fset_2.features["temporal.duration_s"].value

    def test_timestamp_independence(self, base_stream: StreamInfo) -> None:
        """Verify that wall-clock timestamp does not affect features or appear in FeatureSet."""
        iq = np.ones(500, dtype=np.complex64)

        event_no_ts = SignalEvent(
            event_id="ev-no-ts",
            stream=base_stream,
            iq=iq,
            event_start_sample=0,
            event_end_sample=500,
            detected_start_sample=100,
            detected_end_sample=400,
            trigger_sample=200,
            timestamp_ns=None,
        )

        event_with_ts = SignalEvent(
            event_id="ev-with-ts",
            stream=base_stream,
            iq=iq,
            event_start_sample=0,
            event_end_sample=500,
            detected_start_sample=100,
            detected_end_sample=400,
            trigger_sample=200,
            timestamp_ns=1_700_000_000_000_000_000,
        )

        ctx_no_ts = SignalContext(event_no_ts)
        ctx_with_ts = SignalContext(event_with_ts)
        extractor = TemporalFeatureExtractor()

        fset_no_ts = extractor.extract(ctx_no_ts)
        fset_with_ts = extractor.extract(ctx_with_ts)

        assert fset_no_ts.features["temporal.duration_samples"].value == fset_with_ts.features["temporal.duration_samples"].value
        assert fset_no_ts.features["temporal.duration_s"].value == fset_with_ts.features["temporal.duration_s"].value
        assert "temporal.timestamp_ns" not in fset_with_ts.features
        assert "temporal.absolute_start_sample" not in fset_with_ts.features
