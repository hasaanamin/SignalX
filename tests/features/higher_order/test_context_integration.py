"""Integration tests connecting SignalEvent -> SignalContext -> HigherOrderFeatureExtractor."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.higher_order import HigherOrderFeatureExtractor


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    """Fixture providing standard StreamInfo at 1 MHz sample rate."""
    return StreamInfo(
        stream_id="stream-hos-int",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


def make_qpsk_event(stream: StreamInfo, n_symbols: int = 250) -> SignalEvent:
    """Helper creating a deterministic QPSK SignalEvent."""
    symbols = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex64)
    iq = np.tile(symbols, n_symbols // 4)
    total = len(iq)

    return SignalEvent(
        event_id="ev-qpsk-hos-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=10,
        detected_end_sample=total - 10,
        trigger_sample=total // 4,
    )


class TestContextIntegration:
    """Test higher-order feature extraction with real SignalEvent and SignalContext instances."""

    def test_real_signal_context_integration(self, stream_1mhz: StreamInfo) -> None:
        """Real SignalContext with deterministic QPSK extracts all 6 valid higher-order features."""
        event = make_qpsk_event(stream_1mhz, n_symbols=400)
        ctx = SignalContext(event)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "higher_order"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 6

        # Check QPSK values on detected signal_iq (which contains an integer number of full QPSK cycles)
        c20 = fset.features["higher_order.normalized_abs_c20"]
        c40 = fset.features["higher_order.normalized_abs_c40"]
        c41 = fset.features["higher_order.normalized_abs_c41"]
        c42 = fset.features["higher_order.normalized_abs_c42"]
        c60 = fset.features["higher_order.normalized_abs_c60"]
        c63 = fset.features["higher_order.normalized_abs_c63"]

        assert c20.valid is True
        assert c20.unit is None
        assert pytest.approx(c20.value, abs=1e-6) == 0.0

        assert c40.valid is True
        assert c40.unit is None
        assert pytest.approx(c40.value, abs=1e-6) == 1.0

        assert c41.valid is True
        assert c41.unit is None
        assert pytest.approx(c41.value, abs=1e-6) == 0.0

        assert c42.valid is True
        assert c42.unit is None
        assert pytest.approx(c42.value, abs=1e-6) == 1.0

        assert c60.valid is True
        assert c60.unit is None
        assert pytest.approx(c60.value, abs=1e-6) == 0.0

        assert c63.valid is True
        assert c63.unit is None
        assert pytest.approx(c63.value, abs=1e-6) == 4.0

        # Repeated extraction on same context is identical
        fset2 = extractor.extract(ctx)
        for k in fset.features:
            assert fset.features[k].value == fset2.features[k].value
