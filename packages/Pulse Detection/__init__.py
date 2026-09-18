"""
Pulse Detection and Estimation Package (Factory & Abstract Base Class Architecture)
Provides enterprise Factory Pattern and self-contained classes for signal envelope extraction,
dual-threshold hysteresis detection, 5-state FSM pulse detection, boundary refinement,
sub-sample leading/trailing edge estimation, pulse width, and pulse waveform segmentation.
"""

from .base import BasePulseDetectionMetric, PulseDetectionFactory
from ._common import _as_signal_array

import importlib

# Dynamic module imports to register all metric classes into the Factory
_env_ext = importlib.import_module(".Envelope Extraction", package=__name__)
_thresh_det = importlib.import_module(".Threshold Detection", package=__name__)
_pulse_det = importlib.import_module(".Pulse Detection", package=__name__)
_pulse_bnd = importlib.import_module(".Pulse Boundaries", package=__name__)
_edge_det = importlib.import_module(".Leading Trailing Edge Detection", package=__name__)
_pulse_w = importlib.import_module(".Pulse Width", package=__name__)
_pulse_seg = importlib.import_module(".Pulse Segmentation", package=__name__)

# Exported Concrete Metric Classes
EnvelopeExtractionMetric = _env_ext.EnvelopeExtractionMetric
ThresholdDetectionMetric = _thresh_det.ThresholdDetectionMetric
PulseDetectionMetric = _pulse_det.PulseDetectionMetric
PulseBoundariesMetric = _pulse_bnd.PulseBoundariesMetric
LeadingTrailingEdgeDetectionMetric = _edge_det.LeadingTrailingEdgeDetectionMetric
PulseWidthMetric = _pulse_w.PulseWidthMetric
PulseSegmentationMetric = _pulse_seg.PulseSegmentationMetric
PulseFSMState = _pulse_det.PulseFSMState

# Exported Standalone Functional APIs
envelope_extraction = _env_ext.envelope_extraction
extract_envelope = _env_ext.extract_envelope
threshold_detection = _thresh_det.threshold_detection
detect_thresholds = _thresh_det.detect_thresholds
pulse_detection = _pulse_det.pulse_detection
detect_pulse_regions = _pulse_det.detect_pulse_regions
pulse_boundaries = _pulse_bnd.pulse_boundaries
refine_boundaries = _pulse_bnd.refine_boundaries
leading_trailing_edge_detection = _edge_det.leading_trailing_edge_detection
detect_edges = _edge_det.detect_edges
pulse_width = _pulse_w.pulse_width
calculate_pulse_width = _pulse_w.calculate_pulse_width
pulse_segmentation = _pulse_seg.pulse_segmentation
segment_pulses = _pulse_seg.segment_pulses
detect_and_segment = _pulse_seg.detect_and_segment

__all__ = [
    "BasePulseDetectionMetric",
    "PulseDetectionFactory",
    "PulseFSMState",
    "EnvelopeExtractionMetric",
    "ThresholdDetectionMetric",
    "PulseDetectionMetric",
    "PulseBoundariesMetric",
    "LeadingTrailingEdgeDetectionMetric",
    "PulseWidthMetric",
    "PulseSegmentationMetric",
    "envelope_extraction",
    "extract_envelope",
    "threshold_detection",
    "detect_thresholds",
    "pulse_detection",
    "detect_pulse_regions",
    "pulse_boundaries",
    "refine_boundaries",
    "leading_trailing_edge_detection",
    "detect_edges",
    "pulse_width",
    "calculate_pulse_width",
    "pulse_segmentation",
    "segment_pulses",
    "detect_and_segment",
]
