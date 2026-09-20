"""Tests verifying rotation invariance of all 6 phase features."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import SignalEvent, StreamInfo
from signal_processing.features.phase import PhaseFeatureExtractor


def make_waveform_event(phase_offset_rad: float = 0.0) -> SignalEvent:
    """Helper creating a deterministic multi-component IQ event with a global phase rotation."""
    stream = StreamInfo(
        stream_id="stream-phase-rot",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(500)
    # Mix of tones and varying amplitude to exercise phase transitions
    sig = (
        0.6 * np.exp(1j * 2.0 * np.pi * 35_000.0 * n / 1_000_000.0)
        + 0.4 * np.exp(1j * 2.0 * np.pi * 80_000.0 * n / 1_000_000.0)
    )
    iq = (sig * np.exp(1j * phase_offset_rad)).astype(np.complex64)
    total = len(iq)

    return SignalEvent(
        event_id=f"ev-phase-rot-{phase_offset_rad:.3f}",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=0,
        detected_end_sample=total,
        trigger_sample=0,
    )


class TestRotationInvariance:
    """Validate that constant carrier phase offset does not alter any of the 6 phase features."""

    def test_case_d_rotation_invariance(self) -> None:
        """Test Case D: Adding constant phase rotation of 1.234 rad preserves all six phase features."""
        event_orig = make_waveform_event(phase_offset_rad=0.0)
        event_rot = make_waveform_event(phase_offset_rad=1.234)

        extractor = PhaseFeatureExtractor()
        fset_orig = extractor.extract(SignalContext(event_orig))
        fset_rot = extractor.extract(SignalContext(event_rot))

        for key in fset_orig.features:
            val_orig = fset_orig.features[key].value
            val_rot = fset_rot.features[key].value
            # Both values must be valid and approximately equal
            assert fset_orig.features[key].valid is True
            assert fset_rot.features[key].valid is True
            assert pytest.approx(val_rot, rel=1e-5, abs=1e-5) == val_orig

    @pytest.mark.parametrize("offset", [-np.pi / 2.0, np.pi / 3.0, 2.71828, -1.0])
    def test_arbitrary_rotations_invariant(self, offset: float) -> None:
        """Verify invariance across several positive and negative rotation angles."""
        event_base = make_waveform_event(phase_offset_rad=0.0)
        event_offset = make_waveform_event(phase_offset_rad=offset)

        extractor = PhaseFeatureExtractor()
        fset_base = extractor.extract(SignalContext(event_base))
        fset_offset = extractor.extract(SignalContext(event_offset))

        for key in fset_base.features:
            v_base = fset_base.features[key].value
            v_offset = fset_offset.features[key].value
            assert pytest.approx(v_offset, rel=1e-5, abs=1e-5) == v_base
