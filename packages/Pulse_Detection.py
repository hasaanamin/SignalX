from __future__ import annotations
import numpy as np
import pulse_detector as _pd

# Export top-level classes and factories
RealTimePulseDetectorConfig = _pd.RealTimePulseDetectorConfig
RealTimePulseDetector = _pd.RealTimePulseDetector
PulseDetectorFactory = _pd.PulseDetectorFactory
IQRingBuffer = _pd.IQRingBuffer

EnvelopeConfig = _pd.EnvelopeConfig
EnvelopeResult = _pd.EnvelopeResult
EnvelopeExtractorFactory = _pd.EnvelopeExtractorFactory

ThresholdConfig = _pd.ThresholdConfig
ThresholdResult = _pd.ThresholdResult
ThresholdContext = _pd.ThresholdContext
ThresholdDetectorFactory = _pd.ThresholdDetectorFactory

PulseDetectionConfig = _pd.PulseDetectionConfig
PulseEvent = _pd.PulseEvent
PulseEventType = _pd.PulseEventType
PulseDetectionResult = _pd.PulseDetectionResult
PulseDetectionFactory = _pd.PulseDetectionFactory

BoundaryConfig = _pd.BoundaryConfig
BoundaryResult = _pd.BoundaryResult
PulseBoundaryFactory = _pd.PulseBoundaryFactory

EdgeConfig = _pd.EdgeConfig
EdgeResult = _pd.EdgeResult
EdgeDetectorFactory = _pd.EdgeDetectorFactory

PulseWidthConfig = _pd.PulseWidthConfig
PulseWidthResult = _pd.PulseWidthResult
PulseWidthFactory = _pd.PulseWidthFactory

SegmentationConfig = _pd.SegmentationConfig
DetectedPulse = _pd.DetectedPulse
PulseSegmenterFactory = _pd.PulseSegmenterFactory


# Functional DSP wrappers for backward compatibility
def envelope_extraction(signal: np.ndarray, smoothing_window: int = 4, **kwargs) -> dict:
    cfg = _pd.EnvelopeConfig(edge_window_samples=smoothing_window, **kwargs)
    extractor = _pd.EnvelopeExtractorFactory.create(cfg)
    res = extractor.process(signal)
    return {
        "raw_envelope": np.sqrt(res.power_fast),
        "smoothed_envelope": np.sqrt(res.power_edge),
        "energy_envelope": res.power_detect,
        "detection_statistic": res.power_detect,
        "amplitude_envelope": np.sqrt(res.power_fast),
        "power_envelope": res.power_fast,
    }


def extract_envelope(signal: np.ndarray, **kwargs) -> np.ndarray:
    return envelope_extraction(signal, **kwargs)["smoothed_envelope"]


def threshold_detection(detection_statistic: np.ndarray, **kwargs) -> dict:
    cfg = _pd.ThresholdConfig(**kwargs)
    detector = _pd.ThresholdDetectorFactory.create(cfg)
    env_res = _pd.EnvelopeResult(
        power_fast=detection_statistic,
        power_edge=detection_statistic,
        power_detect=detection_statistic,
        start_index=0,
    )
    res = detector.process(env_res)
    return {
        "high_threshold": res.threshold_high,
        "low_threshold": res.threshold_low,
        "noise_floor": res.noise_power,
    }


def detect_thresholds(detection_statistic: np.ndarray, **kwargs) -> dict:
    return threshold_detection(detection_statistic, **kwargs)


def pulse_detection(detection_statistic: np.ndarray, **kwargs) -> list:
    cfg = _pd.PulseDetectionConfig(**kwargs)
    detector = _pd.PulseDetectionFactory.create(cfg)
    th_dict = threshold_detection(detection_statistic)
    th_res = _pd.ThresholdResult(
        noise_power=th_dict["noise_floor"],
        threshold_high=th_dict["high_threshold"],
        threshold_low=th_dict["low_threshold"],
        above_high=(detection_statistic >= th_dict["high_threshold"]),
        above_low=(detection_statistic >= th_dict["low_threshold"]),
        start_index=0,
    )
    res = detector.process(th_res)
    return res.events


def detect_pulse_regions(detection_statistic: np.ndarray, **kwargs) -> list:
    return pulse_detection(detection_statistic, **kwargs)


def pulse_segmentation(signal: np.ndarray, sample_rate: float = 1e6, **kwargs) -> list[DetectedPulse]:
    cfg = _pd.RealTimePulseDetectorConfig(sample_rate=sample_rate)
    detector = _pd.PulseDetectorFactory.create(cfg)
    return detector.process(signal)


def segment_pulses(signal: np.ndarray, **kwargs) -> list[DetectedPulse]:
    return pulse_segmentation(signal, **kwargs)


def detect_and_segment(signal: np.ndarray, **kwargs) -> list[DetectedPulse]:
    return pulse_segmentation(signal, **kwargs)


__all__ = [
    "RealTimePulseDetectorConfig",
    "RealTimePulseDetector",
    "PulseDetectorFactory",
    "IQRingBuffer",
    "EnvelopeExtractorFactory",
    "ThresholdDetectorFactory",
    "PulseDetectionFactory",
    "PulseBoundaryFactory",
    "EdgeDetectorFactory",
    "PulseWidthFactory",
    "PulseSegmenterFactory",
    "envelope_extraction",
    "extract_envelope",
    "threshold_detection",
    "detect_thresholds",
    "pulse_detection",
    "detect_pulse_regions",
    "pulse_segmentation",
    "segment_pulses",
    "detect_and_segment",
]
