"""Tests for FeatureRegistry ordering, validation, immutability, and lookup."""

from typing import Any

import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet
from signal_processing.features.aggregation import (
    FeatureRegistry,
    FeatureRegistryError,
)


class MockExtractor:
    """Configurable mock extractor for registry testing."""

    def __init__(self, name: str, version: str = "1.0") -> None:
        self.name = name
        self.version = version

    def extract(self, context: SignalContext) -> FeatureSet:
        return FeatureSet(extractor=self.name, extractor_version=self.version, features={})


class TestFeatureRegistry:
    """Validate FeatureRegistry construction, ordering, immutability, and lookup."""

    def test_case_a_registry_order(self) -> None:
        """Test Case A: Constructor ordering of extractors (A, B, C) is strictly preserved."""
        ext_a = MockExtractor("extractor_a")
        ext_b = MockExtractor("extractor_b")
        ext_c = MockExtractor("extractor_c")

        registry = FeatureRegistry([ext_a, ext_b, ext_c])

        assert registry.names == ("extractor_a", "extractor_b", "extractor_c")
        assert registry.extractors == (ext_a, ext_b, ext_c)
        assert len(registry) == 3

        # Indexing and iteration preserve order
        assert registry[0] is ext_a
        assert registry[1] is ext_b
        assert registry[2] is ext_c
        assert list(registry) == [ext_a, ext_b, ext_c]

    def test_case_b_duplicate_extractor_name_rejected(self) -> None:
        """Test Case B: Constructing registry with duplicate extractor names raises FeatureRegistryError."""
        ext1 = MockExtractor("test_name")
        ext2 = MockExtractor("test_name")

        with pytest.raises(FeatureRegistryError, match="Duplicate extractor name 'test_name'"):
            FeatureRegistry([ext1, ext2])

    def test_immutable_extractor_tuple(self) -> None:
        """Verify registry.extractors returns an immutable tuple."""
        registry = FeatureRegistry([MockExtractor("ext1")])
        assert isinstance(registry.extractors, tuple)
        with pytest.raises((AttributeError, TypeError)):
            registry.extractors.append(MockExtractor("ext2"))  # type: ignore[attr-defined]

    def test_registry_lookup_success_and_failure(self) -> None:
        """Verify registry.get finds existing extractor and raises FeatureRegistryError for unknown names."""
        ext = MockExtractor("alpha")
        registry = FeatureRegistry([ext])

        assert registry.get("alpha") is ext
        assert registry["alpha"] is ext
        assert "alpha" in registry
        assert ext in registry

        with pytest.raises(FeatureRegistryError, match="No extractor registered with name 'beta'"):
            registry.get("beta")

        with pytest.raises(FeatureRegistryError, match="No extractor registered with name 'beta'"):
            _ = registry["beta"]

        assert "beta" not in registry

    def test_reject_invalid_extractor_types(self) -> None:
        """Verify FeatureRegistryError is raised for non-compliant extractors."""
        with pytest.raises(FeatureRegistryError, match="must not be None"):
            FeatureRegistry([None])  # type: ignore[list-item]

        class BlankName:
            name = "   "
            version = "1.0"
            def extract(self, ctx: Any) -> Any: pass

        with pytest.raises(FeatureRegistryError, match="non-blank string 'name'"):
            FeatureRegistry([BlankName()])

        class NonStringName:
            name = 123
            version = "1.0"
            def extract(self, ctx: Any) -> Any: pass

        with pytest.raises(FeatureRegistryError, match="non-blank string 'name'"):
            FeatureRegistry([NonStringName()])

        class BlankVersion:
            name = "valid"
            version = ""
            def extract(self, ctx: Any) -> Any: pass

        with pytest.raises(FeatureRegistryError, match="non-blank string 'version'"):
            FeatureRegistry([BlankVersion()])

        class UncallableExtract:
            name = "valid"
            version = "1.0"
            extract = "not_callable"

        with pytest.raises(FeatureRegistryError, match="callable 'extract'"):
            FeatureRegistry([UncallableExtract()])

    def test_empty_registry_is_valid(self) -> None:
        """An empty FeatureRegistry can be constructed and behaves consistently."""
        registry = FeatureRegistry([])
        assert len(registry) == 0
        assert registry.names == ()
        assert registry.extractors == ()

    def test_reject_non_iterable_argument(self) -> None:
        """Passing non-iterable to FeatureRegistry raises FeatureRegistryError."""
        with pytest.raises(FeatureRegistryError, match="Expected iterable"):
            FeatureRegistry(12345)  # type: ignore[arg-type]
