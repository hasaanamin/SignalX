"""Default feature extractor registry factory."""

from signal_processing.features.aggregation.registry import FeatureRegistry
from signal_processing.features.amplitude import AmplitudeFeatureExtractor
from signal_processing.features.bandwidth import BandwidthFeatureExtractor
from signal_processing.features.correlation import CorrelationFeatureExtractor
from signal_processing.features.frequency import FrequencyFeatureExtractor
from signal_processing.features.higher_order import HigherOrderFeatureExtractor
from signal_processing.features.phase import PhaseFeatureExtractor
from signal_processing.features.power import PowerFeatureExtractor
from signal_processing.features.spectral import SpectralFeatureExtractor
from signal_processing.features.statistics import StatisticsFeatureExtractor
from signal_processing.features.temporal import TemporalFeatureExtractor

DEFAULT_EXTRACTOR_NAMES: tuple[str, ...] = (
    "amplitude",
    "power",
    "temporal",
    "frequency",
    "bandwidth",
    "spectral",
    "phase",
    "statistics",
    "higher_order",
    "correlation",
)


def build_default_feature_registry() -> FeatureRegistry:
    """Construct a fresh FeatureRegistry containing all 10 canonical feature extractors.

    The extractors are ordered deterministically:
        1. amplitude
        2. power
        3. temporal
        4. frequency
        5. bandwidth
        6. spectral
        7. phase
        8. statistics
        9. higher_order
        10. correlation

    Returns:
        A new FeatureRegistry instance populated with fresh extractor instances.
    """
    extractors = [
        AmplitudeFeatureExtractor(),
        PowerFeatureExtractor(),
        TemporalFeatureExtractor(),
        FrequencyFeatureExtractor(),
        BandwidthFeatureExtractor(),
        SpectralFeatureExtractor(),
        PhaseFeatureExtractor(),
        StatisticsFeatureExtractor(),
        HigherOrderFeatureExtractor(),
        CorrelationFeatureExtractor(),
    ]
    return FeatureRegistry(extractors)
