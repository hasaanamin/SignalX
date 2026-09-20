"""Tests for zero-power signal handling and invalid spectral feature semantics."""

import math

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import PSDResult, SignalEvent, StreamInfo
from signal_processing.features.spectral import SpectralFeatureExtractor


class DummyZeroPSDContext:
    def __init__(self, size: int = 64) -> None:
        freqs = np.linspace(-500_000.0, 500_000.0, size, dtype=np.float64)
        powers = np.zeros(size, dtype=np.float64)
        self._psd = PSDResult(
            frequency_offset_hz=freqs,
            power_linear=powers,
            power_dbfs_per_hz=np.full(size, -300.0, dtype=np.float64),
        )

    def psd(self) -> PSDResult:
        return self._psd


@pytest.fixture
def zero_event() -> SignalEvent:
    """Fixture producing an all-zero SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-spec-zero",
        source_id="synthetic-zero",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    total = 500
    iq = np.zeros(total, dtype=np.complex64)

    return SignalEvent(
        event_id="ev-spec-zero-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestZeroPowerSpectral:
    """Test extraction on zero-power signals without throwing exceptions."""

    def test_case_e_all_zero_psd(self) -> None:
        """Test Case E: All-zero synthetic PSD produces invalid FeatureValues with NaN and descriptive reason."""
        ctx = DummyZeroPSDContext(size=128)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)

        # 1. Entropy
        entropy = fset.features["spectral.entropy"]
        assert entropy.valid is False
        assert math.isnan(entropy.value)
        assert entropy.unit is None
        assert entropy.reason == "PSD contains no positive signal power"

        # 2. Flatness
        flatness = fset.features["spectral.flatness"]
        assert flatness.valid is False
        assert math.isnan(flatness.value)
        assert flatness.unit is None
        assert flatness.reason == "PSD contains no positive signal power"

        # 3. Rolloff 85%
        r85 = fset.features["spectral.rolloff_85_offset_hz"]
        assert r85.valid is False
        assert math.isnan(r85.value)
        assert r85.unit == "Hz"
        assert r85.reason == "PSD contains no positive signal power"

        # 4. Rolloff 95%
        r95 = fset.features["spectral.rolloff_95_offset_hz"]
        assert r95.valid is False
        assert math.isnan(r95.value)
        assert r95.unit == "Hz"
        assert r95.reason == "PSD contains no positive signal power"

    def test_real_zero_signal_context_integration(self, zero_event: SignalEvent) -> None:
        """Real SignalContext on all-zero IQ produces invalid spectral features gracefully."""
        ctx = SignalContext(zero_event)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)

        for key in [
            "spectral.entropy",
            "spectral.flatness",
            "spectral.rolloff_85_offset_hz",
            "spectral.rolloff_95_offset_hz",
        ]:
            fv = fset.features[key]
            assert fv.valid is False
            assert math.isnan(fv.value)
            assert fv.reason == "PSD contains no positive signal power"
