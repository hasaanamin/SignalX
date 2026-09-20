"""Shared enumerations for signal processing domain states and transitions."""

from enum import Enum


class SegmentState(Enum):
    """State of the signal segmentation state machine.

    States:
        IDLE: No signal detected; system is monitoring ambient noise/activity.
        ACTIVE: Signal power has exceeded detection threshold; signal is active.
        POST_TRIGGER: Signal power has dropped below turn-off threshold;
            capturing post-trigger margin samples.
    """

    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    POST_TRIGGER = "POST_TRIGGER"


class SegmentTransition(Enum):
    """Boundary transitions emitted by the segmentation state machine.

    Transitions:
        NONE: No boundary change in the current block.
        START: Transition into active signal (estimated signal-start boundary).
        END: Transition out of active signal (estimated signal-end boundary).
    """

    NONE = "NONE"
    START = "START"
    END = "END"
