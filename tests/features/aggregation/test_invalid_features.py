"""Tests verifying exact preservation of invalid FeatureValues during aggregation."""

import math

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue, FeatureVector
from signal_processing.features.aggregation import (
    FeatureAggregator,
    FeatureRegistry,
)


class DummyContext:
    def __init__(self, event_id: str = "ev-inv-001") -> None:
        self.event_id = event_id


class ExtractorWithInvalidFeatures:
    """Mock extractor returning a mixture of valid and mathematically undefined invalid features."""

    name = "mixed_family"
    version = "1.0"

    def extract(self, context: SignalContext) -> FeatureSet:
        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features={
                "mixed.valid_feature": FeatureValue(
                    value=42.0,
                    unit="dB",
                    valid=True,
                    reason=None,
                ),
                "mixed.zero_power_undefined": FeatureValue(
                    value=float("nan"),
                    unit="Hz",
                    valid=False,
                    reason="Signal has zero power; frequency estimate undefined",
                ),
                "mixed.single_sample_undefined": FeatureValue(
                    value=float("nan"),
                    unit="rad",
                    valid=False,
                    reason="At least two phase samples are required",
                ),
            },
        )


class TestInvalidFeaturePreservation:
    """Validate that invalid FeatureValues are preserved verbatim without imputation or dropping."""

    def test_case_d_invalid_feature_preservation(self) -> None:
        """Test Case D: Aggregation succeeds when extractors yield invalid features; semantics and reasons survive intact."""
        registry = FeatureRegistry([ExtractorWithInvalidFeatures()])
        aggregator = FeatureAggregator(registry)

        vector = aggregator.extract(DummyContext("event-invalid-test"))

        assert isinstance(vector, FeatureVector)
        assert len(vector.features) == 3

        # Valid feature
        fv_valid = vector.features["mixed.valid_feature"]
        assert fv_valid.valid is True
        assert fv_valid.value == 42.0
        assert fv_valid.unit == "dB"
        assert fv_valid.reason is None

        # Zero power invalid feature
        fv_zero_p = vector.features["mixed.zero_power_undefined"]
        assert fv_zero_p.valid is False
        assert math.isnan(fv_zero_p.value)
        assert fv_zero_p.unit == "Hz"
        assert fv_zero_p.reason == "Signal has zero power; frequency estimate undefined"

        # Single sample invalid feature
        fv_single = vector.features["mixed.single_sample_undefined"]
        assert fv_single.valid is False
        assert math.isnan(fv_single.value)
        assert fv_single.unit == "rad"
        assert fv_single.reason == "At least two phase samples are required"

        # Verify no sentinel replacements (0, -1, None)
        assert fv_zero_p.value != 0
        assert fv_zero_p.value is not -1
        assert fv_zero_p.value is not None
