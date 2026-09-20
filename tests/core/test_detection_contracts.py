"""Unit tests for detection and segmentation contracts."""

import pytest

from signal_processing.core import (
    ContractValidationError,
    DetectionMeasurement,
    NoiseEstimate,
    SegmentationDecision,
    SegmentState,
    SegmentTransition,
)

# ---------------------------------------------------------------------------
# NoiseEstimate Tests
# ---------------------------------------------------------------------------

def test_noise_estimate_valid() -> None:
    """Test valid NoiseEstimate construction."""
    ne = NoiseEstimate(
        stream_id="stream-01",
        block_sequence_number=5,
        noise_power_dbfs=-65.4,
        valid=True,
    )
    assert ne.stream_id == "stream-01"
    assert ne.block_sequence_number == 5
    assert ne.noise_power_dbfs == -65.4
    assert ne.valid is True
    assert ne.reason is None


def test_noise_estimate_invalid_with_reason() -> None:
    """Test invalid NoiseEstimate construction with valid explanation reason."""
    ne = NoiseEstimate(
        stream_id="stream-01",
        block_sequence_number=0,
        noise_power_dbfs=float("nan"),
        valid=False,
        reason="Warmup phase: insufficient history for noise estimation",
    )
    assert ne.valid is False
    assert ne.reason == "Warmup phase: insufficient history for noise estimation"


def test_noise_estimate_valid_with_unexpected_reason() -> None:
    """Test that valid=True rejects an error reason."""
    with pytest.raises(ContractValidationError, match="reason"):
        NoiseEstimate(
            stream_id="stream-01",
            block_sequence_number=1,
            noise_power_dbfs=-60.0,
            valid=True,
            reason="This reason should not exist on a valid estimate",
        )


def test_noise_estimate_invalid_without_reason() -> None:
    """Test that valid=False requires a non-blank reason."""
    with pytest.raises(ContractValidationError, match="reason"):
        NoiseEstimate(
            stream_id="stream-01",
            block_sequence_number=1,
            noise_power_dbfs=-60.0,
            valid=False,
            reason=None,
        )

    with pytest.raises(ContractValidationError, match="reason"):
        NoiseEstimate(
            stream_id="stream-01",
            block_sequence_number=1,
            noise_power_dbfs=-60.0,
            valid=False,
            reason="   ",
        )


def test_noise_estimate_non_finite_power_when_valid() -> None:
    """Test that valid=True rejects NaN and Inf noise_power_dbfs."""
    with pytest.raises(ContractValidationError, match="noise_power_dbfs"):
        NoiseEstimate(
            stream_id="stream-01",
            block_sequence_number=0,
            noise_power_dbfs=float("nan"),
            valid=True,
        )

    with pytest.raises(ContractValidationError, match="noise_power_dbfs"):
        NoiseEstimate(
            stream_id="stream-01",
            block_sequence_number=0,
            noise_power_dbfs=float("inf"),
            valid=True,
        )


def test_noise_estimate_negative_sequence_number() -> None:
    """Test rejection of negative sequence number."""
    with pytest.raises(ContractValidationError, match="block_sequence_number"):
        NoiseEstimate(
            stream_id="stream-01",
            block_sequence_number=-1,
            noise_power_dbfs=-60.0,
        )


def test_noise_estimate_blank_stream_id() -> None:
    """Test rejection of blank stream_id."""
    with pytest.raises(ContractValidationError, match="stream_id"):
        NoiseEstimate(
            stream_id="   ",
            block_sequence_number=0,
            noise_power_dbfs=-60.0,
        )


# ---------------------------------------------------------------------------
# DetectionMeasurement Tests
# ---------------------------------------------------------------------------

def test_detection_measurement_valid() -> None:
    """Test valid DetectionMeasurement construction."""
    dm = DetectionMeasurement(
        stream_id="stream-01",
        block_sequence_number=10,
        start_sample=10_000,
        end_sample=11_000,
        signal_power_dbfs=-35.0,
        noise_power_dbfs=-65.0,
        snr_estimate_db=30.0,
        on_threshold_dbfs=-50.0,
        off_threshold_dbfs=-55.0,
        above_on_threshold=True,
        below_off_threshold=False,
    )
    assert dm.stream_id == "stream-01"
    assert dm.block_sequence_number == 10
    assert dm.start_sample == 10_000
    assert dm.end_sample == 11_000
    assert dm.signal_power_dbfs == -35.0
    assert dm.noise_power_dbfs == -65.0
    assert dm.snr_estimate_db == 30.0
    assert dm.on_threshold_dbfs == -50.0
    assert dm.off_threshold_dbfs == -55.0
    assert dm.above_on_threshold is True
    assert dm.below_off_threshold is False


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, 100),    # negative start
        (100, 100),   # start == end (empty)
        (100, 90),    # end < start (backwards)
    ],
)
def test_detection_measurement_invalid_start_end(start: int, end: int) -> None:
    """Test rejection of invalid start and end sample intervals."""
    with pytest.raises(ContractValidationError):
        DetectionMeasurement(
            stream_id="stream-01",
            block_sequence_number=0,
            start_sample=start,
            end_sample=end,
            signal_power_dbfs=-30.0,
            noise_power_dbfs=-60.0,
            snr_estimate_db=30.0,
            on_threshold_dbfs=-45.0,
            off_threshold_dbfs=-50.0,
            above_on_threshold=True,
            below_off_threshold=False,
        )


@pytest.mark.parametrize(
    "bad_val",
    [float("nan"), float("inf"), float("-inf"), "not_a_float", True],
)
def test_detection_measurement_non_finite_fields(bad_val: object) -> None:
    """Test rejection of non-finite measurement and threshold values."""
    with pytest.raises(ContractValidationError, match="signal_power_dbfs"):
        DetectionMeasurement(
            stream_id="stream-01",
            block_sequence_number=0,
            start_sample=0,
            end_sample=100,
            signal_power_dbfs=bad_val,  # type: ignore[arg-type]
            noise_power_dbfs=-60.0,
            snr_estimate_db=30.0,
            on_threshold_dbfs=-45.0,
            off_threshold_dbfs=-50.0,
            above_on_threshold=True,
            below_off_threshold=False,
        )


    with pytest.raises(ContractValidationError, match="noise_power_dbfs"):
        DetectionMeasurement(
            stream_id="stream-01",
            block_sequence_number=0,
            start_sample=0,
            end_sample=100,
            signal_power_dbfs=-30.0,
            noise_power_dbfs=bad_val,
            snr_estimate_db=30.0,
            on_threshold_dbfs=-45.0,
            off_threshold_dbfs=-50.0,
            above_on_threshold=True,
            below_off_threshold=False,
        )

    with pytest.raises(ContractValidationError, match="snr_estimate_db"):
        DetectionMeasurement(
            stream_id="stream-01",
            block_sequence_number=0,
            start_sample=0,
            end_sample=100,
            signal_power_dbfs=-30.0,
            noise_power_dbfs=-60.0,
            snr_estimate_db=bad_val,
            on_threshold_dbfs=-45.0,
            off_threshold_dbfs=-50.0,
            above_on_threshold=True,
            below_off_threshold=False,
        )


# ---------------------------------------------------------------------------
# SegmentationDecision Tests
# ---------------------------------------------------------------------------

def test_segmentation_decision_none_case() -> None:
    """Test transition NONE requires boundary_sample to be None."""
    decision = SegmentationDecision(
        state=SegmentState.IDLE,
        transition=SegmentTransition.NONE,
        block_sequence_number=12,
        boundary_sample=None,
    )
    assert decision.state == SegmentState.IDLE
    assert decision.transition == SegmentTransition.NONE
    assert decision.boundary_sample is None

    # NONE with a boundary_sample must be rejected
    with pytest.raises(ContractValidationError, match="boundary_sample"):
        SegmentationDecision(
            state=SegmentState.IDLE,
            transition=SegmentTransition.NONE,
            block_sequence_number=12,
            boundary_sample=1200,
        )


def test_segmentation_decision_start_case() -> None:
    """Test transition START requires boundary_sample >= 0."""
    decision = SegmentationDecision(
        state=SegmentState.ACTIVE,
        transition=SegmentTransition.START,
        block_sequence_number=15,
        boundary_sample=15000,
    )
    assert decision.state == SegmentState.ACTIVE
    assert decision.transition == SegmentTransition.START
    assert decision.boundary_sample == 15000

    # START with missing boundary_sample must be rejected
    with pytest.raises(ContractValidationError, match="boundary_sample"):
        SegmentationDecision(
            state=SegmentState.ACTIVE,
            transition=SegmentTransition.START,
            block_sequence_number=15,
            boundary_sample=None,
        )

    # START with negative boundary_sample must be rejected
    with pytest.raises(ContractValidationError, match="boundary_sample"):
        SegmentationDecision(
            state=SegmentState.ACTIVE,
            transition=SegmentTransition.START,
            block_sequence_number=15,
            boundary_sample=-10,
        )


def test_segmentation_decision_end_case() -> None:
    """Test transition END requires boundary_sample >= 0."""
    decision = SegmentationDecision(
        state=SegmentState.POST_TRIGGER,
        transition=SegmentTransition.END,
        block_sequence_number=20,
        boundary_sample=20500,
    )
    assert decision.state == SegmentState.POST_TRIGGER
    assert decision.transition == SegmentTransition.END
    assert decision.boundary_sample == 20500

    # END with missing boundary_sample must be rejected
    with pytest.raises(ContractValidationError, match="boundary_sample"):
        SegmentationDecision(
            state=SegmentState.POST_TRIGGER,
            transition=SegmentTransition.END,
            block_sequence_number=20,
            boundary_sample=None,
        )


def test_noise_estimate_invalid_valid_type() -> None:
    """Test rejection when valid is not a boolean."""
    with pytest.raises(ContractValidationError, match="valid"):
        NoiseEstimate(
            stream_id="stream-01",
            block_sequence_number=0,
            noise_power_dbfs=-60.0,
            valid="yes",  # type: ignore[arg-type]
        )


def test_noise_estimate_invalid_power_type_when_invalid() -> None:
    """Test rejection when noise_power_dbfs is non-numeric even when valid=False."""
    with pytest.raises(ContractValidationError, match="noise_power_dbfs"):
        NoiseEstimate(
            stream_id="stream-01",
            block_sequence_number=0,
            noise_power_dbfs="not_a_number",  # type: ignore[arg-type]
            valid=False,
            reason="Noise estimate unavailable",
        )


def test_detection_measurement_invalid_end_sample_type() -> None:
    """Test rejection when end_sample is non-integer or boolean."""
    with pytest.raises(ContractValidationError, match="end_sample"):
        DetectionMeasurement(
            stream_id="stream-01",
            block_sequence_number=0,
            start_sample=0,
            end_sample=True,  # type: ignore[arg-type]
            signal_power_dbfs=-30.0,
            noise_power_dbfs=-60.0,
            snr_estimate_db=30.0,
            on_threshold_dbfs=-45.0,
            off_threshold_dbfs=-50.0,
            above_on_threshold=True,
            below_off_threshold=False,
        )


def test_detection_measurement_invalid_threshold_bool_flags() -> None:
    """Test rejection when above_on_threshold or below_off_threshold are not booleans."""
    with pytest.raises(ContractValidationError, match="above_on_threshold"):
        DetectionMeasurement(
            stream_id="stream-01",
            block_sequence_number=0,
            start_sample=0,
            end_sample=100,
            signal_power_dbfs=-30.0,
            noise_power_dbfs=-60.0,
            snr_estimate_db=30.0,
            on_threshold_dbfs=-45.0,
            off_threshold_dbfs=-50.0,
            above_on_threshold="true",  # type: ignore[arg-type]
            below_off_threshold=False,
        )

    with pytest.raises(ContractValidationError, match="below_off_threshold"):
        DetectionMeasurement(
            stream_id="stream-01",
            block_sequence_number=0,
            start_sample=0,
            end_sample=100,
            signal_power_dbfs=-30.0,
            noise_power_dbfs=-60.0,
            snr_estimate_db=30.0,
            on_threshold_dbfs=-45.0,
            off_threshold_dbfs=-50.0,
            above_on_threshold=True,
            below_off_threshold="false",  # type: ignore[arg-type]
        )


def test_segmentation_decision_invalid_state_and_transition_types() -> None:
    """Test rejection of non-enum state or transition."""
    with pytest.raises(ContractValidationError, match="state"):
        SegmentationDecision(
            state="IDLE",  # type: ignore[arg-type]
            transition=SegmentTransition.NONE,
            block_sequence_number=0,
        )

    with pytest.raises(ContractValidationError, match="transition"):
        SegmentationDecision(
            state=SegmentState.IDLE,
            transition="NONE",  # type: ignore[arg-type]
            block_sequence_number=0,
        )


@pytest.mark.parametrize("bad_reason", [123, [], object(), "", "   ", None])
def test_noise_estimate_invalid_reason_types(bad_reason: object) -> None:
    """Test that non-string or blank reasons raise ContractValidationError and never leak AttributeError."""
    with pytest.raises(ContractValidationError, match="reason"):
        NoiseEstimate(
            stream_id="stream-01",
            block_sequence_number=0,
            noise_power_dbfs=float("nan"),
            valid=False,
            reason=bad_reason,  # type: ignore[arg-type]
        )


def test_detection_measurement_hysteresis_ordering() -> None:
    """Test requirement that on_threshold_dbfs > off_threshold_dbfs."""
    # Valid: -40 > -45
    dm = DetectionMeasurement(
        stream_id="s1",
        block_sequence_number=0,
        start_sample=0,
        end_sample=10,
        signal_power_dbfs=-38.0,
        noise_power_dbfs=-70.0,
        snr_estimate_db=32.0,
        on_threshold_dbfs=-40.0,
        off_threshold_dbfs=-45.0,
        above_on_threshold=True,
        below_off_threshold=False,
    )
    assert dm.on_threshold_dbfs > dm.off_threshold_dbfs

    # Reversed: ON < OFF (-45 < -40)
    with pytest.raises(ContractValidationError, match="hysteresis"):
        DetectionMeasurement(
            stream_id="s1",
            block_sequence_number=0,
            start_sample=0,
            end_sample=10,
            signal_power_dbfs=-38.0,
            noise_power_dbfs=-70.0,
            snr_estimate_db=32.0,
            on_threshold_dbfs=-45.0,
            off_threshold_dbfs=-40.0,
            above_on_threshold=True,
            below_off_threshold=False,
        )

    # Equal: ON == OFF (-40 == -40)
    with pytest.raises(ContractValidationError, match="hysteresis"):
        DetectionMeasurement(
            stream_id="s1",
            block_sequence_number=0,
            start_sample=0,
            end_sample=10,
            signal_power_dbfs=-38.0,
            noise_power_dbfs=-70.0,
            snr_estimate_db=32.0,
            on_threshold_dbfs=-40.0,
            off_threshold_dbfs=-40.0,
            above_on_threshold=True,
            below_off_threshold=False,
        )


def test_detection_measurement_flag_consistency() -> None:
    """Test that above_on_threshold and below_off_threshold match signal_power_dbfs."""
    # signal_power (-30) >= on_threshold (-40), but above_on_threshold is False -> reject
    with pytest.raises(ContractValidationError, match="above_on_threshold"):
        DetectionMeasurement(
            stream_id="s1",
            block_sequence_number=0,
            start_sample=0,
            end_sample=10,
            signal_power_dbfs=-30.0,
            noise_power_dbfs=-70.0,
            snr_estimate_db=40.0,
            on_threshold_dbfs=-40.0,
            off_threshold_dbfs=-45.0,
            above_on_threshold=False,
            below_off_threshold=False,
        )

    # signal_power (-42) < on_threshold (-40), but above_on_threshold is True -> reject
    with pytest.raises(ContractValidationError, match="above_on_threshold"):
        DetectionMeasurement(
            stream_id="s1",
            block_sequence_number=0,
            start_sample=0,
            end_sample=10,
            signal_power_dbfs=-42.0,
            noise_power_dbfs=-70.0,
            snr_estimate_db=28.0,
            on_threshold_dbfs=-40.0,
            off_threshold_dbfs=-45.0,
            above_on_threshold=True,
            below_off_threshold=False,
        )

    # signal_power (-60) < off_threshold (-45), but below_off_threshold is False -> reject
    with pytest.raises(ContractValidationError, match="below_off_threshold"):
        DetectionMeasurement(
            stream_id="s1",
            block_sequence_number=0,
            start_sample=0,
            end_sample=10,
            signal_power_dbfs=-60.0,
            noise_power_dbfs=-70.0,
            snr_estimate_db=10.0,
            on_threshold_dbfs=-40.0,
            off_threshold_dbfs=-45.0,
            above_on_threshold=False,
            below_off_threshold=False,
        )

    # signal_power (-42) >= off_threshold (-45), but below_off_threshold is True -> reject
    with pytest.raises(ContractValidationError, match="below_off_threshold"):
        DetectionMeasurement(
            stream_id="s1",
            block_sequence_number=0,
            start_sample=0,
            end_sample=10,
            signal_power_dbfs=-42.0,
            noise_power_dbfs=-70.0,
            snr_estimate_db=28.0,
            on_threshold_dbfs=-40.0,
            off_threshold_dbfs=-45.0,
            above_on_threshold=False,
            below_off_threshold=True,
        )


def test_detection_measurement_boundary_equality() -> None:
    """Test boundary conditions: power == on_threshold and power == off_threshold."""
    # signal_power == on_threshold (-40.0) -> above_on_threshold must be True, below_off_threshold False
    dm_on = DetectionMeasurement(
        stream_id="s1",
        block_sequence_number=0,
        start_sample=0,
        end_sample=10,
        signal_power_dbfs=-40.0,
        noise_power_dbfs=-70.0,
        snr_estimate_db=30.0,
        on_threshold_dbfs=-40.0,
        off_threshold_dbfs=-45.0,
        above_on_threshold=True,
        below_off_threshold=False,
    )
    assert dm_on.above_on_threshold is True

    # signal_power == off_threshold (-45.0) -> below_off_threshold must be False, above_on_threshold False
    dm_off = DetectionMeasurement(
        stream_id="s1",
        block_sequence_number=0,
        start_sample=0,
        end_sample=10,
        signal_power_dbfs=-45.0,
        noise_power_dbfs=-70.0,
        snr_estimate_db=25.0,
        on_threshold_dbfs=-40.0,
        off_threshold_dbfs=-45.0,
        above_on_threshold=False,
        below_off_threshold=False,
    )
    assert dm_off.below_off_threshold is False


@pytest.mark.parametrize(
    ("transition", "invalid_state"),
    [
        (SegmentTransition.START, SegmentState.IDLE),
        (SegmentTransition.START, SegmentState.POST_TRIGGER),
        (SegmentTransition.END, SegmentState.IDLE),
        (SegmentTransition.END, SegmentState.ACTIVE),
    ],
)
def test_segmentation_decision_invalid_state_transition_combinations(
    transition: SegmentTransition, invalid_state: SegmentState
) -> None:
    """Test that START requires ACTIVE and END requires POST_TRIGGER."""
    with pytest.raises(ContractValidationError, match="requires state"):
        SegmentationDecision(
            state=invalid_state,
            transition=transition,
            block_sequence_number=1,
            boundary_sample=100,
        )


@pytest.mark.parametrize(
    "valid_state",
    [SegmentState.IDLE, SegmentState.ACTIVE, SegmentState.POST_TRIGGER],
)
def test_segmentation_decision_none_transition_allows_any_state(
    valid_state: SegmentState,
) -> None:
    """Test that transition NONE permits IDLE, ACTIVE, and POST_TRIGGER states."""
    dec = SegmentationDecision(
        state=valid_state,
        transition=SegmentTransition.NONE,
        block_sequence_number=0,
        boundary_sample=None,
    )
    assert dec.state == valid_state
    assert dec.transition == SegmentTransition.NONE


