"""Signal segmentation package."""

from signal_processing.segmentation.exceptions import (
    InvalidTransitionError,
    SegmentationError,
)
from signal_processing.segmentation.segmenter import HysteresisSegmenter

__all__ = [
    "HysteresisSegmenter",
    "SegmentationError",
    "InvalidTransitionError",
]
