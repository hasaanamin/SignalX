"""Unit tests for feature contracts (FeatureValue, FeatureSet, FeatureVector)."""

from dataclasses import FrozenInstanceError

import pytest

from signal_processing.core import (
    ContractValidationError,
    FeatureSet,
    FeatureValue,
    FeatureVector,
)

# ---------------------------------------------------------------------------
# FeatureValue Tests
# ---------------------------------------------------------------------------

def test_feature_value_valid() -> None:
    """Test valid FeatureValue construction."""
    fv = FeatureValue(value=10500.5, unit="Hz", valid=True)
    assert fv.value == 10500.5
    assert fv.unit == "Hz"
    assert fv.valid is True
    assert fv.reason is None


def test_feature_value_invalid_with_reason() -> None:
    """Test invalid FeatureValue construction with error reason."""
    fv = FeatureValue(
        value=float("nan"),
        unit="dB",
        valid=False,
        reason="Division by zero: zero signal energy",
    )
    assert fv.valid is False
    assert fv.reason == "Division by zero: zero signal energy"


def test_feature_value_valid_with_nan_rejected() -> None:
    """Test that valid=True rejects NaN values."""
    with pytest.raises(ContractValidationError, match="finite"):
        FeatureValue(value=float("nan"), unit="Hz", valid=True)


def test_feature_value_valid_with_reason_rejected() -> None:
    """Test that valid=True rejects an error reason."""
    with pytest.raises(ContractValidationError, match="reason"):
        FeatureValue(value=10.0, unit="dB", valid=True, reason="Should not be here")


def test_feature_value_invalid_without_reason_rejected() -> None:
    """Test that valid=False requires a non-blank reason."""
    with pytest.raises(ContractValidationError, match="reason"):
        FeatureValue(value=float("nan"), unit="dB", valid=False, reason=None)

    with pytest.raises(ContractValidationError, match="reason"):
        FeatureValue(value=float("nan"), unit="dB", valid=False, reason="   ")


def test_feature_value_immutability() -> None:
    """Test that FeatureValue is frozen."""
    fv = FeatureValue(value=1.0, unit="s")
    with pytest.raises(FrozenInstanceError):
        fv.value = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FeatureSet Tests
# ---------------------------------------------------------------------------

def test_feature_set_valid_and_defensive_copy() -> None:
    """Test valid FeatureSet, defensive copy, and mapping immutability."""
    raw_dict = {
        "centroid_hz": FeatureValue(value=15000.0, unit="Hz"),
        "spread_hz": FeatureValue(value=2500.0, unit="Hz"),
    }

    fset = FeatureSet(
        extractor="spectral",
        extractor_version="1.0.0",
        features=raw_dict,
    )

    assert fset.extractor == "spectral"
    assert fset.extractor_version == "1.0.0"
    assert len(fset.features) == 2
    assert fset.features["centroid_hz"].value == 15000.0

    # 1. Defensively copied: mutating raw_dict does not mutate fset.features
    raw_dict["centroid_hz"] = FeatureValue(value=999.0, unit="Hz")
    assert fset.features["centroid_hz"].value == 15000.0

    # 2. Immutable mapping: assigning to features mapping raises TypeError
    with pytest.raises(TypeError):
        fset.features["new_feat"] = FeatureValue(value=1.0)  # type: ignore[index]


def test_feature_set_blank_extractor() -> None:
    """Test rejection of blank extractor."""
    with pytest.raises(ContractValidationError, match="extractor"):
        FeatureSet(
            extractor="   ",
            extractor_version="1.0",
            features={"f1": FeatureValue(value=1.0)},
        )


def test_feature_set_empty_features() -> None:
    """Test rejection of empty features dictionary."""
    with pytest.raises(ContractValidationError, match="features"):
        FeatureSet(
            extractor="power",
            extractor_version="1.0",
            features={},
        )


def test_feature_set_invalid_feature_value_type() -> None:
    """Test rejection when mapping values are not FeatureValue instances."""
    with pytest.raises(ContractValidationError, match="FeatureValue"):
        FeatureSet(
            extractor="power",
            extractor_version="1.0",
            features={"mean_dbfs": 12.5},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# FeatureVector Tests
# ---------------------------------------------------------------------------

def test_feature_vector_valid() -> None:
    """Test valid FeatureVector creation."""
    features = {
        "spectral.centroid_hz": FeatureValue(value=15000.0, unit="Hz"),
        "power.mean_dbfs": FeatureValue(value=-45.2, unit="dBFS"),
    }
    vec = FeatureVector(
        event_id="ev-001",
        feature_schema_version="v1",
        features=features,
    )

    assert vec.event_id == "ev-001"
    assert vec.feature_schema_version == "v1"
    assert len(vec.features) == 2


def test_feature_vector_from_feature_sets() -> None:
    """Test assembling FeatureVector from multiple FeatureSet instances."""
    spectral_set = FeatureSet(
        extractor="spectral",
        extractor_version="1.0",
        features={
            "spectral.centroid_hz": FeatureValue(value=15000.0, unit="Hz"),
            "spectral.entropy": FeatureValue(value=0.85, unit=None),
        },
    )
    power_set = FeatureSet(
        extractor="power",
        extractor_version="1.0",
        features={
            "power.mean_dbfs": FeatureValue(value=-40.0, unit="dBFS"),
            "power.peak_dbfs": FeatureValue(value=-32.0, unit="dBFS"),
        },
    )

    vec = FeatureVector.from_feature_sets(
        event_id="ev-002",
        feature_schema_version="v1",
        feature_sets=[spectral_set, power_set],
    )

    assert vec.event_id == "ev-002"
    assert len(vec.features) == 4
    assert vec.features["spectral.centroid_hz"].value == 15000.0
    assert vec.features["power.mean_dbfs"].value == -40.0


def test_feature_vector_from_feature_sets_duplicate_rejection() -> None:
    """Test that duplicate feature names across sets are rejected without silent overwriting."""
    set_a = FeatureSet(
        extractor="module_a",
        extractor_version="1.0",
        features={"shared.metric": FeatureValue(value=1.0)},
    )
    set_b = FeatureSet(
        extractor="module_b",
        extractor_version="1.0",
        features={"shared.metric": FeatureValue(value=2.0)},
    )

    with pytest.raises(ContractValidationError, match="Duplicate feature name"):
        FeatureVector.from_feature_sets(
            event_id="ev-003",
            feature_schema_version="v1",
            feature_sets=[set_a, set_b],
        )


def test_feature_value_invalid_types_and_blank_unit() -> None:
    """Test rejection of non-bool valid, non-numeric value, and blank unit."""
    with pytest.raises(ContractValidationError, match="valid"):
        FeatureValue(value=1.0, valid="yes")  # type: ignore[arg-type]

    with pytest.raises(ContractValidationError, match="value"):
        FeatureValue(value="not_numeric", valid=True)  # type: ignore[arg-type]

    with pytest.raises(ContractValidationError, match="unit"):
        FeatureValue(value=1.0, unit="   ")


def test_feature_set_invalid_mapping_and_blank_key() -> None:
    """Test rejection when features is not a mapping or has a blank key."""
    with pytest.raises(ContractValidationError, match="mapping"):
        FeatureSet(
            extractor="test",
            extractor_version="1.0",
            features="not_a_mapping",  # type: ignore[arg-type]
        )

    with pytest.raises(ContractValidationError, match="Feature key"):
        FeatureSet(
            extractor="test",
            extractor_version="1.0",
            features={"   ": FeatureValue(value=1.0)},
        )


def test_feature_vector_invalid_mapping_and_types() -> None:
    """Test rejection when FeatureVector features is not a mapping or contains non-FeatureValue."""
    with pytest.raises(ContractValidationError, match="mapping"):
        FeatureVector(
            event_id="ev-01",
            feature_schema_version="v1",
            features="not_a_mapping",  # type: ignore[arg-type]
        )

    with pytest.raises(ContractValidationError, match="Feature value"):
        FeatureVector(
            event_id="ev-01",
            feature_schema_version="v1",
            features={"f1": 123.4},  # type: ignore[dict-item]
        )

    with pytest.raises(ContractValidationError, match="FeatureSet instance"):
        FeatureVector.from_feature_sets(
            event_id="ev-01",
            feature_schema_version="v1",
            feature_sets=["not_a_feature_set"],  # type: ignore[list-item]
        )


def test_feature_vector_blank_key() -> None:
    """Test rejection when FeatureVector feature key is blank."""
    with pytest.raises(ContractValidationError, match="Feature key"):
        FeatureVector(
            event_id="ev-01",
            feature_schema_version="v1",
            features={"   ": FeatureValue(value=1.0)},
        )


@pytest.mark.parametrize("bad_reason", [123, [], object(), "", "   ", None])
def test_feature_value_invalid_reason_types(bad_reason: object) -> None:
    """Test that invalid reason types raise ContractValidationError and never leak AttributeError."""
    with pytest.raises(ContractValidationError, match="reason"):
        FeatureValue(
            value=float("nan"),
            valid=False,
            reason=bad_reason,  # type: ignore[arg-type]
        )




