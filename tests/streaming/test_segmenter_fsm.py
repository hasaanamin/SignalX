"""Behavioral tests for HysteresisSegmenter state machine."""


from signal_processing.core.contracts import DetectionMeasurement
from signal_processing.core.enums import SegmentState, SegmentTransition
from signal_processing.segmentation.segmenter import HysteresisSegmenter


def make_measurement(
    seq: int,
    start: int,
    end: int,
    signal_power: float,
    on_th: float = -40.0,
    off_th: float = -50.0,
) -> DetectionMeasurement:
    """Helper to construct valid DetectionMeasurement for segmenter testing."""
    above = signal_power >= on_th
    below = signal_power < off_th
    snr = max(0.0, signal_power - off_th)
    return DetectionMeasurement(
        stream_id="stream-fsm-01",
        block_sequence_number=seq,
        start_sample=start,
        end_sample=end,
        signal_power_dbfs=signal_power,
        noise_power_dbfs=off_th - 10.0,
        snr_estimate_db=snr,
        on_threshold_dbfs=on_th,
        off_threshold_dbfs=off_th,
        above_on_threshold=above,
        below_off_threshold=below,
    )


class TestSegmenterFSM:
    """Rigorous tests for hysteresis, persistence, spike cancellation, and reopening."""

    def test_start_persistence_and_spike_cancellation(self) -> None:
        """Isolated spike shorter than start_persistence is ignored; confirmed run emits START."""
        # Requires 2 consecutive hits
        seg = HysteresisSegmenter(start_persistence=2, end_persistence=3)

        # 1. Idle noise
        m0 = make_measurement(0, 0, 100, -60.0)
        d0 = seg.update(m0)
        assert d0.state == SegmentState.IDLE
        assert d0.transition == SegmentTransition.NONE

        # 2. Single isolated spike
        m1 = make_measurement(1, 100, 200, -20.0)  # > on_th (-40)
        d1 = seg.update(m1)
        assert d1.state == SegmentState.IDLE
        assert d1.transition == SegmentTransition.NONE
        assert seg.first_on_sample == 199

        # 3. Next sample drops back to noise -> spike cancelled!
        m2 = make_measurement(2, 200, 300, -60.0)
        d2 = seg.update(m2)
        assert d2.state == SegmentState.IDLE
        assert d2.transition == SegmentTransition.NONE
        assert seg.first_on_sample is None

        # 4. Two consecutive hits -> confirmed START
        m3 = make_measurement(3, 300, 400, -20.0)
        d3 = seg.update(m3)
        assert d3.state == SegmentState.IDLE
        assert d3.transition == SegmentTransition.NONE

        m4 = make_measurement(4, 400, 500, -20.0)
        d4 = seg.update(m4)
        assert d4.state == SegmentState.ACTIVE
        assert d4.transition == SegmentTransition.START
        assert d4.boundary_sample == 300  # Start of first on window
        assert seg.confirmation_sample == 499
        assert seg.trigger_sample == 399

    def test_hysteresis_gap_and_end_persistence(self) -> None:
        """Power in hysteresis gap maintains ACTIVE; confirmed OFF emits END."""
        # Requires 1 hit to start, 3 hits to end
        seg = HysteresisSegmenter(start_persistence=1, end_persistence=3)

        # Trigger START
        m0 = make_measurement(0, 0, 100, -20.0)
        d0 = seg.update(m0)
        assert d0.state == SegmentState.ACTIVE
        assert d0.transition == SegmentTransition.START

        # Power drops into hysteresis gap: -45 dBFS (between -40 and -50)
        # neither above ON nor below OFF -> maintains ACTIVE
        m1 = make_measurement(1, 100, 200, -45.0)
        d1 = seg.update(m1)
        assert d1.state == SegmentState.ACTIVE
        assert d1.transition == SegmentTransition.NONE

        # Power drops below OFF (-60 dBFS) for 1 hit: still ACTIVE
        m2 = make_measurement(2, 200, 300, -60.0)
        d2 = seg.update(m2)
        assert d2.state == SegmentState.ACTIVE
        assert d2.transition == SegmentTransition.NONE

        # Brief fade rebound into hysteresis gap: resets OFF counter!
        m3 = make_measurement(3, 300, 400, -45.0)
        d3 = seg.update(m3)
        assert d3.state == SegmentState.ACTIVE

        # Now 3 consecutive OFF hits
        m4 = make_measurement(4, 400, 500, -60.0)
        m5 = make_measurement(5, 500, 600, -60.0)
        m6 = make_measurement(6, 600, 700, -60.0)

        _ = seg.update(m4)
        _ = seg.update(m5)
        d6 = seg.update(m6)

        assert d6.state == SegmentState.POST_TRIGGER
        assert d6.transition == SegmentTransition.END
        assert d6.boundary_sample == 500  # End of first below-off window (m4.end_sample)
        assert seg.end_confirmation_sample == 699

    def test_burst_reopening_during_post_trigger(self) -> None:
        """Burst reopening during POST_TRIGGER hang time reverts to ACTIVE without new event."""
        seg = HysteresisSegmenter(start_persistence=1, end_persistence=1, allow_reopen=True)

        # Trigger START
        _ = seg.update(make_measurement(0, 0, 100, -20.0))
        # Trigger END -> state is POST_TRIGGER
        d_end = seg.update(make_measurement(1, 100, 200, -60.0))
        assert d_end.state == SegmentState.POST_TRIGGER
        assert d_end.transition == SegmentTransition.END

        # While waiting in POST_TRIGGER, power surges back up!
        d_reopen = seg.update(make_measurement(2, 200, 300, -10.0))
        assert d_reopen.state == SegmentState.ACTIVE
        assert d_reopen.transition == SegmentTransition.NONE
        assert seg.reopened_count == 1

    def test_max_duration_cap(self) -> None:
        """Burst exceeding max_event_samples forces END transition."""
        seg = HysteresisSegmenter(start_persistence=1, end_persistence=5, max_event_samples=500)

        # START at sample 0
        _ = seg.update(make_measurement(0, 0, 100, -20.0))

        # Continuous CW signal: keep feeding active measurements
        _ = seg.update(make_measurement(1, 100, 300, -20.0))
        d_over = seg.update(make_measurement(2, 300, 600, -20.0))  # 600 - 0 = 600 >= 500

        assert d_over.state == SegmentState.POST_TRIGGER
        assert d_over.transition == SegmentTransition.END
        assert seg.max_duration_reached is True
