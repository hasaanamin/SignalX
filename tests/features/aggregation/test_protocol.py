"""Tests verifying the FeatureExtractor structural Protocol."""

from typing import Any

import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet
from signal_processing.features import (
    AmplitudeFeatureExtractor,
    BandwidthFeatureExtractor,
    CorrelationFeatureExtractor,
    FrequencyFeatureExtractor,
    HigherOrderFeatureExtractor,
    PhaseFeatureExtractor,
    PowerFeatureExtractor,
    SpectralFeatureExtractor,
    StatisticsFeatureExtractor,
    TemporalFeatureExtractor,
)
from signal_processing.features.aggregation.protocols import FeatureExtractor


class ValidDummyExtractor:
    """Mock extractor conforming structurally to FeatureExtractor."""

    name: str = "dummy"
    version: str = "1.0"

    def extract(self, context: SignalContext) -> FeatureSet:
        return FeatureSet(extractor="dummy", extractor_version="1.0", features={})


class MissingNameExtractor:
    version: str = "1.0"

    def extract(self, context: SignalContext) -> FeatureSet:
        return FeatureSet(extractor="dummy", extractor_version="1.0", features={})


class MissingVersionExtractor:
    name: str = "dummy"

    def extract(self, context: SignalContext) -> FeatureSet:
        return FeatureSet(extractor="dummy", extractor_version="1.0", features={})


class MissingExtractExtractor:
    name: str = "dummy"
    version: str = "1.0"


class TestFeatureExtractorProtocol:
    """Verify runtime checkability and adherence of extractors to FeatureExtractor protocol."""

    @pytest.mark.parametrize(
        "extractor_cls",
        [
            AmplitudeFeatureExtractor,
            PowerFeatureExtractor,
            TemporalFeatureExtractor,
            FrequencyFeatureExtractor,
            BandwidthFeatureExtractor,
            SpectralFeatureExtractor,
            PhaseFeatureExtractor,
            StatisticsFeatureExtractor,
            HigherOrderFeatureExtractor,
            CorrelationFeatureExtractor,
        ],
    )
    def test_canonical_extractors_satisfy_protocol(self, extractor_cls: type[Any]) -> None:
        """Every canonical feature extractor in the repository satisfies FeatureExtractor protocol."""
        ext = extractor_cls()
        assert isinstance(ext, FeatureExtractor)
        assert isinstance(ext.name, str) and ext.name.strip()
        assert isinstance(ext.version, str) and ext.version.strip()
        assert callable(ext.extract)

    def test_valid_dummy_satisfies_protocol(self) -> None:
        """A plain class implementing name, version, and extract satisfies FeatureExtractor."""
        dummy = ValidDummyExtractor()
        assert isinstance(dummy, FeatureExtractor)

    def test_missing_attributes_fail_protocol(self) -> None:
        """Classes missing required structural attributes fail FeatureExtractor check."""
        assert not isinstance(MissingNameExtractor(), FeatureExtractor)
        assert not isinstance(MissingVersionExtractor(), FeatureExtractor)
        assert not isinstance(MissingExtractExtractor(), FeatureExtractor)
