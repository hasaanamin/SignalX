"""Event capture and assembly package."""

from signal_processing.capture.assembler import EventAssembler
from signal_processing.capture.exceptions import (
    CaptureCapacityError,
    CaptureDiscontinuityError,
    CaptureError,
)
from signal_processing.capture.sidecar import CaptureSidecar

__all__ = [
    "EventAssembler",
    "CaptureSidecar",
    "CaptureError",
    "CaptureCapacityError",
    "CaptureDiscontinuityError",
]
