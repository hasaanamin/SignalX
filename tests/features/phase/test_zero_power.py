"""Tests verifying phase feature behavior on zero-power signals."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureValue, SignalEvent, StreamInfo
from signal_processing.features.phase import PhaseFeatureExtractor


def make_zero_iq_event() -> SignalEvent:
    """Create an event with completely zero IQ data."""
    stream = StreamInfo(
        stream_id="stream-zero-power",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    iq = np.zeros(256, dtype=np.complex64)
    total = len(iq)

    return SignalEvent(
        event_id="ev-zero-phase-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=0,
        detected_end_sample=total,
        trigger_sample=0,
    )


class ExplodingPhaseContext:
    """Mock context where phase() explodes to verify power precheck prevents phase transform call."""

    def mean_power_linear(self) -> float:
        return 0.0

    def phase(self) -> Any:
        raise AssertionError("context.phase() must NOT be called when mean_power_linear <= 0!")


class TestZeroPowerPhase:
    """Validate that zero-power signals cleanly produce invalid FeatureValues without exceptions."""

    def test_case_h_real_zero_signal_context(self) -> None:
        """Test Case H: All-zero IQ SignalContext returns all six features invalid with clear reason."""
        event = make_zero_iq_event()
        ctx = SignalContext(event)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        assert len(fset.features) == 6
        expected_units = {
            "phase.circular_resultant_length": None,
            "phase.circular_variance": None,
            "phase.increment_mean_rad": "rad",
            "phase.increment_std_rad": "rad",
            "phase.increment_rms_rad": "rad",
            "phase.increment_abs_mean_rad": "rad",
        }

        for key, expected_unit in expected_units.items():
            fv = fset.features[key]
            assert isinstance(fv, FeatureValue)
            assert fv.valid is False
            assert math.isnan(fv.value)
            assert fv.unit == expected_unit
            assert fv.reason is not None
            assert "zero power" in fv.reason.lower()

    def test_zero_power_precheck_skips_phase_call(self) -> None:
        """Verify that power <= 0 precheck returns invalid feature set before calling context.phase()."""
        ctx = ExplodingPhaseContext()
        extractor = PhaseFeatureExtractor()

        fset = extractor.extract(ctx)

        assert len(fset.features) == 6
        for fv in fset.features.values():
            assert fv.valid is False
            assert math.isnan(fv.value)
