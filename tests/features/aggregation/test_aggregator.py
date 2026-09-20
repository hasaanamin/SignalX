"""Tests for FeatureAggregator orchestration, error handling, and FeatureVector assembly."""

from typing import Any

import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue, FeatureVector
from signal_processing.features.aggregation import (
    FeatureAggregationError,
    FeatureAggregator,
    FeatureRegistry,
)


class DummyContext:
    """Minimal mock context providing an event_id."""

    def __init__(self, event_id: str = "ev-test-123") -> None:
        self.event_id = event_id


class ConfigurableExtractor:
    """Mock extractor returning custom FeatureSets or raising exceptions."""

    def __init__(
        self,
        name: str,
        version: str = "1.0",
        features: dict[str, FeatureValue] | None = None,
        should_raise: Exception | None = None,
        reported_extractor: str | None = None,
        reported_version: str | None = None,
        raw_result: Any = None,
    ) -> None:
        self.name = name
        self.version = version
        self._features = features or {
            f"{name}.feat": FeatureValue(value=1.0, unit=None, valid=True, reason=None)
        }
        self._should_raise = should_raise
        self._reported_extractor = reported_extractor if reported_extractor is not None else name
        self._reported_version = reported_version if reported_version is not None else version
        self._raw_result = raw_result
        self.contexts_received: list[Any] = []

    def extract(self, context: SignalContext) -> FeatureSet:
        self.contexts_received.append(context)
        if self._should_raise:
            raise self._should_raise
        if self._raw_result is not None:
            return self._raw_result  # type: ignore[return-value]
        return FeatureSet(
            extractor=self._reported_extractor,
            extractor_version=self._reported_version,
            features=self._features,
        )


class TestFeatureAggregator:
    """Validate feature aggregator execution, validation, and error policies."""

    def test_case_c_custom_subset_registry(self) -> None:
        """Test Case C: Custom registry with two extractors returns FeatureVector with exactly those two families."""
        ext_a = ConfigurableExtractor("family_a", features={
            "family_a.x": FeatureValue(value=10.0, unit="Hz", valid=True, reason=None)
        })
        ext_b = ConfigurableExtractor("family_b", features={
            "family_b.y": FeatureValue(value=20.0, unit="s", valid=True, reason=None)
        })

        registry = FeatureRegistry([ext_a, ext_b])
        aggregator = FeatureAggregator(registry)
        ctx = DummyContext("event-custom-001")

        vector = aggregator.extract(ctx)

        assert isinstance(vector, FeatureVector)
        assert vector.event_id == "event-custom-001"
        assert len(vector.features) == 2
        assert set(vector.features.keys()) == {"family_a.x", "family_b.y"}
        assert vector.features["family_a.x"].value == 10.0
        assert vector.features["family_b.y"].value == 20.0

    def test_case_e_extractor_failure_fails_aggregation_atomically(self) -> None:
        """Test Case E: An extractor raising RuntimeError fails the aggregation atomically without partial output."""
        ext_a = ConfigurableExtractor("ext_a")
        ext_fail = ConfigurableExtractor("ext_failing", should_raise=RuntimeError("Hardware failure"))
        ext_b = ConfigurableExtractor("ext_b")

        registry = FeatureRegistry([ext_a, ext_fail, ext_b])
        aggregator = FeatureAggregator(registry)
        ctx = DummyContext()

        with pytest.raises(FeatureAggregationError, match="Extractor 'ext_failing' unexpectedly raised an exception") as exc_info:
            aggregator.extract(ctx)

        # Original exception chained via __cause__
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "Hardware failure" in str(exc_info.value.__cause__)

        # Ensure subsequent extractors were not executed
        assert len(ext_b.contexts_received) == 0

    def test_case_f_wrong_featureset_extractor_name(self) -> None:
        """Test Case F: Mismatch between registered extractor name and FeatureSet.extractor raises FeatureAggregationError."""
        ext = ConfigurableExtractor("alpha", reported_extractor="beta")
        registry = FeatureRegistry([ext])
        aggregator = FeatureAggregator(registry)

        with pytest.raises(FeatureAggregationError, match="Extractor name mismatch: extractor registered as 'alpha' returned FeatureSet with extractor='beta'"):
            aggregator.extract(DummyContext())

    def test_case_g_wrong_extractor_version(self) -> None:
        """Test Case G: Mismatch between registered version and FeatureSet.extractor_version raises FeatureAggregationError."""
        ext = ConfigurableExtractor("alpha", version="1.0", reported_version="2.0")
        registry = FeatureRegistry([ext])
        aggregator = FeatureAggregator(registry)

        with pytest.raises(FeatureAggregationError, match="Extractor version mismatch for 'alpha': registered version='1.0', returned extractor_version='2.0'"):
            aggregator.extract(DummyContext())

    def test_case_i_event_id_propagation(self) -> None:
        """Test Case I: FeatureVector inherits context.event_id exactly."""
        registry = FeatureRegistry([ConfigurableExtractor("test_fam")])
        aggregator = FeatureAggregator(registry)

        vec = aggregator.extract(DummyContext("event-specific-id-999"))
        assert vec.event_id == "event-specific-id-999"

    def test_case_j_schema_version_propagation_and_validation(self) -> None:
        """Test Case J: Default schema version is '1.0', custom version is preserved, blank is rejected."""
        registry = FeatureRegistry([ConfigurableExtractor("test_fam")])

        # Default schema version
        agg_default = FeatureAggregator(registry)
        vec_default = agg_default.extract(DummyContext())
        assert vec_default.feature_schema_version == "1.0"

        # Custom schema version
        agg_custom = FeatureAggregator(registry, feature_schema_version="2.1-custom")
        vec_custom = agg_custom.extract(DummyContext())
        assert vec_custom.feature_schema_version == "2.1-custom"

        # Blank or non-string schema version rejected
        with pytest.raises(FeatureAggregationError, match="non-blank string"):
            FeatureAggregator(registry, feature_schema_version="")

        with pytest.raises(FeatureAggregationError, match="non-blank string"):
            FeatureAggregator(registry, feature_schema_version="   ")

        with pytest.raises(FeatureAggregationError, match="non-blank string"):
            FeatureAggregator(registry, feature_schema_version=123)  # type: ignore[arg-type]

    def test_same_context_passed_to_every_extractor(self) -> None:
        """Aggregator must pass the exact same SignalContext object reference to every extractor."""
        ext1 = ConfigurableExtractor("ext1")
        ext2 = ConfigurableExtractor("ext2")
        ext3 = ConfigurableExtractor("ext3")

        registry = FeatureRegistry([ext1, ext2, ext3])
        aggregator = FeatureAggregator(registry)
        ctx = DummyContext("ev-shared-ctx")

        _ = aggregator.extract(ctx)

        assert ext1.contexts_received[0] is ctx
        assert ext2.contexts_received[0] is ctx
        assert ext3.contexts_received[0] is ctx

    def test_reject_empty_featureset(self) -> None:
        """Extractor returning an empty FeatureSet must raise FeatureAggregationError."""
        empty_fset = object.__new__(FeatureSet)
        object.__setattr__(empty_fset, "extractor", "empty_ext")
        object.__setattr__(empty_fset, "extractor_version", "1.0")
        object.__setattr__(empty_fset, "features", {})

        ext = ConfigurableExtractor("empty_ext", raw_result=empty_fset)
        registry = FeatureRegistry([ext])
        aggregator = FeatureAggregator(registry)

        with pytest.raises(FeatureAggregationError, match="returned an empty FeatureSet"):
            aggregator.extract(DummyContext())

    def test_reject_non_featureset_result(self) -> None:
        """Extractor returning a non-FeatureSet object must raise FeatureAggregationError."""
        ext = ConfigurableExtractor("bad_type", raw_result="not_a_featureset")
        registry = FeatureRegistry([ext])
        aggregator = FeatureAggregator(registry)

        with pytest.raises(FeatureAggregationError, match="expected FeatureSet instance"):
            aggregator.extract(DummyContext())

    def test_reject_non_featurevalue_in_featureset(self) -> None:
        """FeatureSet containing a non-FeatureValue value must raise FeatureAggregationError."""
        bad_fset = object.__new__(FeatureSet)
        object.__setattr__(bad_fset, "extractor", "bad_val")
        object.__setattr__(bad_fset, "extractor_version", "1.0")
        object.__setattr__(bad_fset, "features", {"bad.key": "raw_float_value"})

        ext = ConfigurableExtractor("bad_val", raw_result=bad_fset)
        registry = FeatureRegistry([ext])
        aggregator = FeatureAggregator(registry)

        with pytest.raises(FeatureAggregationError, match="expected FeatureValue instance"):
            aggregator.extract(DummyContext())

    def test_reject_invalid_context(self) -> None:
        """Extractor must reject None or context missing event_id."""
        aggregator = FeatureAggregator(FeatureRegistry([]))

        with pytest.raises(FeatureAggregationError, match="Context must not be None"):
            aggregator.extract(None)

        class MissingEventId:
            pass

        with pytest.raises(FeatureAggregationError, match="non-blank string 'event_id'"):
            aggregator.extract(MissingEventId())

        class BlankEventId:
            event_id = "   "

        with pytest.raises(FeatureAggregationError, match="non-blank string 'event_id'"):
            aggregator.extract(BlankEventId())
