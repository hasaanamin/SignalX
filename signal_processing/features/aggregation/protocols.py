"""Protocols defining the feature extractor interface."""

from typing import Protocol, runtime_checkable

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet


@runtime_checkable
class FeatureExtractor(Protocol):
    """Structural interface for feature extractors.

    Any extractor adhering to this interface exposes:
        name: Unique non-empty string identifier for the feature family.
        version: Non-empty semantic version string.
        extract(context: SignalContext) -> FeatureSet: Callable producing a FeatureSet from a SignalContext.
    """

    name: str
    version: str

    def extract(self, context: SignalContext) -> FeatureSet:
        """Extract a FeatureSet from a SignalContext."""
        ...
