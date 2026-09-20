"""Feature aggregation layer for RF signal processing."""

from signal_processing.features.aggregation.aggregator import FeatureAggregator
from signal_processing.features.aggregation.defaults import (
    DEFAULT_EXTRACTOR_NAMES,
    build_default_feature_registry,
)
from signal_processing.features.aggregation.exceptions import (
    FeatureAggregationError,
    FeatureRegistryError,
)
from signal_processing.features.aggregation.protocols import FeatureExtractor
from signal_processing.features.aggregation.registry import FeatureRegistry

__all__ = [
    "DEFAULT_EXTRACTOR_NAMES",
    "FeatureAggregationError",
    "FeatureAggregator",
    "FeatureExtractor",
    "FeatureRegistry",
    "FeatureRegistryError",
    "build_default_feature_registry",
]
