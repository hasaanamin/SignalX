"""Tests verifying fail-fast rejection of duplicate feature keys across extractors."""

import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.aggregation import (
    FeatureAggregationError,
    FeatureAggregator,
    FeatureRegistry,
)


class DummyContext:
    def __init__(self, event_id: str = "ev-dup-001") -> None:
        self.event_id = event_id


class StaticExtractor:
    """Mock extractor returning a pre-defined dictionary of FeatureValues."""

    def __init__(self, name: str, features: dict[str, FeatureValue]) -> None:
        self.name = name
        self.version = "1.0"
        self._features = features

    def extract(self, context: SignalContext) -> FeatureSet:
        return FeatureSet(extractor=self.name, extractor_version=self.version, features=self._features)


class TestDuplicateFeatureKeys:
    """Verify that duplicate feature keys across FeatureSets are rejected and never overwritten."""

    def test_case_h_duplicate_feature_key_rejected(self) -> None:
        """Test Case H: Two extractors returning the same key 'duplicate.metric' must fail with FeatureAggregationError."""
        fv1 = FeatureValue(value=1.0, unit="V", valid=True, reason=None)
        fv2 = FeatureValue(value=2.0, unit="V", valid=True, reason=None)

        ext_a = StaticExtractor("extractor_a", {"duplicate.metric": fv1, "family_a.unique": fv1})
        ext_b = StaticExtractor("extractor_b", {"duplicate.metric": fv2, "family_b.unique": fv2})

        registry = FeatureRegistry([ext_a, ext_b])
        aggregator = FeatureAggregator(registry)

        with pytest.raises(FeatureAggregationError, match="duplicate.metric") as exc_info:
            aggregator.extract(DummyContext())

        # Verify duplicate key name is explicitly mentioned in error message
        assert "duplicate.metric" in str(exc_info.value)
        assert "extractor_b" in str(exc_info.value) or "extractor_a" in str(exc_info.value)
