"""Tests for default feature extractor registry construction and ordering."""

from signal_processing.features.aggregation import (
    DEFAULT_EXTRACTOR_NAMES,
    FeatureExtractor,
    FeatureRegistry,
    build_default_feature_registry,
)


class TestDefaultRegistry:
    """Validate canonical default registry composition and stateless factory behavior."""

    def test_default_registry_contains_exactly_ten_families(self) -> None:
        """Default registry contains exactly 10 feature extractor instances."""
        registry = build_default_feature_registry()
        assert isinstance(registry, FeatureRegistry)
        assert len(registry) == 10

    def test_default_registry_exact_order(self) -> None:
        """Default registry ordering matches canonical version-1.0 sequence exactly."""
        expected_order = (
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
        registry = build_default_feature_registry()

        assert registry.names == expected_order
        assert DEFAULT_EXTRACTOR_NAMES == expected_order

        # Verify each extractor in registry is an instance of FeatureExtractor protocol
        for ext in registry.extractors:
            assert isinstance(ext, FeatureExtractor)
            assert ext.version == ("2.0" if ext.name == "bandwidth" else "1.0")

    def test_fresh_registry_per_call(self) -> None:
        """build_default_feature_registry returns fresh instances on every invocation without shared mutable state."""
        reg1 = build_default_feature_registry()
        reg2 = build_default_feature_registry()

        assert reg1 is not reg2
        assert len(reg1) == len(reg2)

        # Extractor instances themselves are fresh objects
        for ext1, ext2 in zip(reg1.extractors, reg2.extractors, strict=True):
            assert ext1 is not ext2
            assert ext1.name == ext2.name
            assert ext1.version == ext2.version
