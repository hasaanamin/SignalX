from __future__ import annotations
import importlib

_pkg = importlib.import_module("packages.Pulse Detector")

BasePulseDetectorStage = _pkg.BasePulseDetectorStage
PulseDetectorFactory = _pkg.PulseDetectorFactory
PulseCandidate = _pkg.PulseCandidate

EnvelopeExtractionStage = _pkg.EnvelopeExtractionStage
extract_envelope = _pkg.extract_envelope

ThresholdDetectionStage = _pkg.ThresholdDetectionStage
detect_threshold = _pkg.detect_threshold

PulseDetectionStage = _pkg.PulseDetectionStage
detect_pulse_state = _pkg.detect_pulse_state

PulseBoundariesStage = _pkg.PulseBoundariesStage
find_pulse_boundaries = _pkg.find_pulse_boundaries

LeadingTrailingEdgeDetectionStage = _pkg.LeadingTrailingEdgeDetectionStage
refine_edges = _pkg.refine_edges

PulseWidthStage = _pkg.PulseWidthStage
evaluate_pulse_width = _pkg.evaluate_pulse_width

PulseSegmentationStage = _pkg.PulseSegmentationStage
segment_pulses = _pkg.segment_pulses

StreamingPulseDetector = _pkg.StreamingPulseDetector

detect_pulse_candidates = _pkg.detect_pulse_candidates
pulse_detector_summary = _pkg.pulse_detector_summary

__all__ = [
    "BasePulseDetectorStage",
    "PulseDetectorFactory",
    "PulseCandidate",
    "EnvelopeExtractionStage",
    "extract_envelope",
    "ThresholdDetectionStage",
    "detect_threshold",
    "PulseDetectionStage",
    "detect_pulse_state",
    "PulseBoundariesStage",
    "find_pulse_boundaries",
    "LeadingTrailingEdgeDetectionStage",
    "refine_edges",
    "PulseWidthStage",
    "evaluate_pulse_width",
    "PulseSegmentationStage",
    "segment_pulses",
    "StreamingPulseDetector",
    "detect_pulse_candidates",
    "pulse_detector_summary",
]

