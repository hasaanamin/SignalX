"""Stateful hysteresis persistence signal segmenter."""

from __future__ import annotations

import numpy as np

from signal_processing.core.contracts import DetectionMeasurement, SegmentationDecision
from signal_processing.core.enums import SegmentState, SegmentTransition
from signal_processing.core.interfaces import Resettable, SegmenterProtocol


class HysteresisSegmenter(SegmenterProtocol, Resettable):
    """Hysteresis and persistence signal segmentation state machine.

    Maintains:
      - IDLE: Accumulates consecutive ON-threshold hits up to start_persistence.
        Values below ON reset the hit counter.
      - ACTIVE: In burst. Accumulates consecutive below-OFF hits up to end_persistence.
        Values between OFF and ON maintain ACTIVE (hysteresis).
      - POST_TRIGGER: Waiting for trailing samples. Supports burst reopening if
        power rises above ON threshold during hang time.
    """

    def __init__(
        self,
        start_persistence: int = 2,
        end_persistence: int = 5,
        max_event_samples: int = 100_000,
        allow_reopen: bool = True,
    ) -> None:
        if isinstance(start_persistence, bool) or not isinstance(start_persistence, (int, np.integer)) or start_persistence < 1:
            raise ValueError(f"start_persistence must be an integer >= 1, got {start_persistence!r}")
        if isinstance(end_persistence, bool) or not isinstance(end_persistence, (int, np.integer)) or end_persistence < 1:
            raise ValueError(f"end_persistence must be an integer >= 1, got {end_persistence!r}")
        if isinstance(max_event_samples, bool) or not isinstance(max_event_samples, (int, np.integer)) or max_event_samples < 1:
            raise ValueError(f"max_event_samples must be an integer >= 1, got {max_event_samples!r}")
        if not isinstance(allow_reopen, bool):
            raise TypeError(f"allow_reopen must be a boolean, got {type(allow_reopen).__name__}")

        self._start_persistence = int(start_persistence)
        self._end_persistence = int(end_persistence)
        self._max_event_samples = int(max_event_samples)
        self._allow_reopen = bool(allow_reopen)

        self._state: SegmentState = SegmentState.IDLE
        self._on_count: int = 0
        self._off_count: int = 0

        # Boundary landmarks
        self._first_on_window: DetectionMeasurement | None = None
        self._first_on_sample: int | None = None
        self._confirmation_sample: int | None = None
        self._trigger_sample: int | None = None
        self._detected_start_sample: int | None = None

        self._first_off_window: DetectionMeasurement | None = None
        self._detected_end_sample: int | None = None
        self._end_confirmation_sample: int | None = None

        self._max_duration_reached: bool = False
        self._awaiting_off_rearm: bool = False
        self._reopened_count: int = 0

    @property
    def state(self) -> SegmentState:
        return self._state

    @property
    def start_persistence(self) -> int:
        return self._start_persistence

    @property
    def end_persistence(self) -> int:
        return self._end_persistence

    @property
    def first_on_sample(self) -> int | None:
        return self._first_on_sample

    @property
    def confirmation_sample(self) -> int | None:
        return self._confirmation_sample

    @property
    def trigger_sample(self) -> int | None:
        return self._trigger_sample

    @property
    def detected_start_sample(self) -> int | None:
        return self._detected_start_sample

    @property
    def detected_end_sample(self) -> int | None:
        return self._detected_end_sample

    @property
    def end_confirmation_sample(self) -> int | None:
        return self._end_confirmation_sample

    @property
    def max_duration_reached(self) -> bool:
        return self._max_duration_reached

    @property
    def awaiting_off_rearm(self) -> bool:
        return self._awaiting_off_rearm

    @property
    def reopened_count(self) -> int:
        return self._reopened_count

    def update(self, measurement: DetectionMeasurement) -> SegmentationDecision:
        """Update state machine with a new detection measurement."""
        seq = measurement.block_sequence_number

        if self._state == SegmentState.IDLE:
            if self._awaiting_off_rearm:
                if measurement.below_off_threshold:
                    self._awaiting_off_rearm = False
                else:
                    return SegmentationDecision(
                        state=SegmentState.IDLE,
                        transition=SegmentTransition.NONE,
                        block_sequence_number=seq,
                        boundary_sample=None,
                    )

            if measurement.above_on_threshold:
                if self._on_count == 0:
                    self._first_on_window = measurement
                    self._first_on_sample = measurement.end_sample - 1
                self._on_count += 1

                if self._on_count >= self._start_persistence:
                    # Confirmed START transition
                    self._state = SegmentState.ACTIVE
                    self._confirmation_sample = measurement.end_sample - 1
                    self._trigger_sample = self._first_on_sample
                    assert self._first_on_window is not None
                    self._detected_start_sample = self._first_on_window.start_sample
                    self._off_count = 0
                    self._max_duration_reached = False
                    self._first_off_window = None

                    return SegmentationDecision(
                        state=SegmentState.ACTIVE,
                        transition=SegmentTransition.START,
                        block_sequence_number=seq,
                        boundary_sample=self._detected_start_sample,
                    )
                else:
                    return SegmentationDecision(
                        state=SegmentState.IDLE,
                        transition=SegmentTransition.NONE,
                        block_sequence_number=seq,
                        boundary_sample=None,
                    )
            else:
                self._on_count = 0
                self._first_on_window = None
                self._first_on_sample = None
                return SegmentationDecision(
                    state=SegmentState.IDLE,
                    transition=SegmentTransition.NONE,
                    block_sequence_number=seq,
                    boundary_sample=None,
                )

        elif self._state == SegmentState.ACTIVE:
            # Check maximum burst duration limit
            if (
                self._detected_start_sample is not None
                and (measurement.end_sample - self._detected_start_sample) >= self._max_event_samples
            ):
                self._state = SegmentState.POST_TRIGGER
                self._max_duration_reached = True
                self._awaiting_off_rearm = True
                self._detected_end_sample = measurement.end_sample
                self._end_confirmation_sample = measurement.end_sample - 1
                return SegmentationDecision(
                    state=SegmentState.POST_TRIGGER,
                    transition=SegmentTransition.END,
                    block_sequence_number=seq,
                    boundary_sample=self._detected_end_sample,
                )

            if measurement.below_off_threshold:
                if self._off_count == 0:
                    self._first_off_window = measurement
                self._off_count += 1

                if self._off_count >= self._end_persistence:
                    # Confirmed END transition
                    self._state = SegmentState.POST_TRIGGER
                    self._end_confirmation_sample = measurement.end_sample - 1
                    assert self._first_off_window is not None
                    self._detected_end_sample = self._first_off_window.end_sample
                    return SegmentationDecision(
                        state=SegmentState.POST_TRIGGER,
                        transition=SegmentTransition.END,
                        block_sequence_number=seq,
                        boundary_sample=self._detected_end_sample,
                    )
                else:
                    return SegmentationDecision(
                        state=SegmentState.ACTIVE,
                        transition=SegmentTransition.NONE,
                        block_sequence_number=seq,
                        boundary_sample=None,
                    )
            else:
                # Signal still above off threshold (active or within hysteresis gap)
                self._off_count = 0
                self._first_off_window = None
                return SegmentationDecision(
                    state=SegmentState.ACTIVE,
                    transition=SegmentTransition.NONE,
                    block_sequence_number=seq,
                    boundary_sample=None,
                )

        elif self._state == SegmentState.POST_TRIGGER:
            if measurement.below_off_threshold:
                self._awaiting_off_rearm = False

            # Check if signal reopens during post-trigger margin
            if self._allow_reopen and not self._awaiting_off_rearm and measurement.above_on_threshold:
                self._state = SegmentState.ACTIVE
                self._reopened_count += 1
                self._off_count = 0
                self._first_off_window = None
                self._detected_end_sample = None
                self._end_confirmation_sample = None
                return SegmentationDecision(
                    state=SegmentState.ACTIVE,
                    transition=SegmentTransition.NONE,
                    block_sequence_number=seq,
                    boundary_sample=None,
                )
            else:
                return SegmentationDecision(
                    state=SegmentState.POST_TRIGGER,
                    transition=SegmentTransition.NONE,
                    block_sequence_number=seq,
                    boundary_sample=None,
                )

        return SegmentationDecision(
            state=self._state,
            transition=SegmentTransition.NONE,
            block_sequence_number=seq,
            boundary_sample=None,
        )

    def finish_event(self) -> None:
        """Mark event completion and return state machine to IDLE, clearing per-event landmarks."""
        self._state = SegmentState.IDLE
        self._on_count = 0
        self._off_count = 0
        self._first_on_window = None
        self._first_on_sample = None
        self._confirmation_sample = None
        self._trigger_sample = None
        self._detected_start_sample = None
        self._first_off_window = None
        self._detected_end_sample = None
        self._end_confirmation_sample = None
        self._max_duration_reached = False

    def reset(self) -> None:
        """Reset state machine to initial IDLE state."""
        self._state = SegmentState.IDLE
        self._on_count = 0
        self._off_count = 0
        self._first_on_window = None
        self._first_on_sample = None
        self._confirmation_sample = None
        self._trigger_sample = None
        self._detected_start_sample = None
        self._first_off_window = None
        self._detected_end_sample = None
        self._end_confirmation_sample = None
        self._max_duration_reached = False
        self._awaiting_off_rearm = False
        self._reopened_count = 0
