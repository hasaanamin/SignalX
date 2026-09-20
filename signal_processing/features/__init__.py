"""Feature extraction packages for RF signal processing."""

from signal_processing.features.aggregation import (
    DEFAULT_EXTRACTOR_NAMES,
    FeatureAggregationError,
    FeatureAggregator,
    FeatureExtractor,
    FeatureRegistry,
    FeatureRegistryError,
    build_default_feature_registry,
)
from signal_processing.features.amplitude import (
    AmplitudeFeatureConfig,
    AmplitudeFeatureError,
    AmplitudeFeatureExtractor,
)
from signal_processing.features.bandwidth import (
    BandwidthFeatureError,
    BandwidthFeatureExtractor,
)
from signal_processing.features.correlation import (
    CorrelationFeatureError,
    CorrelationFeatureExtractor,
)
from signal_processing.features.frequency import (
    FrequencyFeatureError,
    FrequencyFeatureExtractor,
)
from signal_processing.features.higher_order import (
    HigherOrderFeatureError,
    HigherOrderFeatureExtractor,
)
from signal_processing.features.phase import (
    PhaseFeatureError,
    PhaseFeatureExtractor,
)
from signal_processing.features.power import (
    PowerFeatureError,
    PowerFeatureExtractor,
)
from signal_processing.features.spectral import (
    SpectralFeatureError,
    SpectralFeatureExtractor,
)
from signal_processing.features.statistics import (
    StatisticsFeatureError,
    StatisticsFeatureExtractor,
)
from signal_processing.features.temporal import (
    TemporalFeatureError,
    TemporalFeatureExtractor,
)

__all__ = [
    "DEFAULT_EXTRACTOR_NAMES",
    "AmplitudeFeatureConfig",
    "AmplitudeFeatureError",
    "AmplitudeFeatureExtractor",
    "BandwidthFeatureError",
    "BandwidthFeatureExtractor",
    "CorrelationFeatureError",
    "CorrelationFeatureExtractor",
    "FeatureAggregationError",
    "FeatureAggregator",
    "FeatureExtractor",
    "FeatureRegistry",
    "FeatureRegistryError",
    "FrequencyFeatureError",
    "FrequencyFeatureExtractor",
    "HigherOrderFeatureError",
    "HigherOrderFeatureExtractor",
    "PhaseFeatureError",
    "PhaseFeatureExtractor",
    "PowerFeatureError",
    "PowerFeatureExtractor",
    "SpectralFeatureError",
    "SpectralFeatureExtractor",
    "StatisticsFeatureError",
    "StatisticsFeatureExtractor",
    "TemporalFeatureError",
    "TemporalFeatureExtractor",
    "build_default_feature_registry",
]
