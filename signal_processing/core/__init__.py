"""Foundational domain contracts, enums, interfaces, exceptions, and validation for signal processing."""

from signal_processing.core.contracts import (
    AnalysisResult,
    AutocorrelationResult,
    CaptureQuality,
    DetectionMeasurement,
    ExtractionPolicy,
    FeatureSet,
    FeatureValue,
    FeatureVector,
    IQBlock,
    NoiseEstimate,
    PhaseResult,
    PSDResult,
    SegmentationDecision,
    SignalEvent,
    StreamInfo,
)
from signal_processing.core.enums import (
    SegmentState,
    SegmentTransition,
)
from signal_processing.core.exceptions import (
    ContractValidationError,
    SignalProcessingError,
    StreamConsistencyError,
)
from signal_processing.core.interfaces import (
    DetectorProtocol,
    IQSource,
    NoiseEstimatorProtocol,
    Resettable,
    SegmenterProtocol,
)
from signal_processing.core.validation import (
    validate_consecutive_blocks,
)

__all__ = [
    "AnalysisResult",
    "AutocorrelationResult",
    "CaptureQuality",
    "ContractValidationError",
    "DetectionMeasurement",
    "DetectorProtocol",
    "ExtractionPolicy",
    "FeatureSet",
    "FeatureValue",
    "FeatureVector",
    "IQBlock",
    "IQSource",
    "NoiseEstimate",
    "NoiseEstimatorProtocol",
    "PSDResult",
    "PhaseResult",
    "Resettable",
    "SegmentState",
    "SegmentTransition",
    "SegmentationDecision",
    "SegmenterProtocol",
    "SignalEvent",
    "SignalProcessingError",
    "StreamConsistencyError",
    "StreamInfo",
    "validate_consecutive_blocks",
]
