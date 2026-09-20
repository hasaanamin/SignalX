"""Ordered immutable registry of feature extractors."""

from collections.abc import Iterable, Iterator
from typing import Any

from signal_processing.features.aggregation.exceptions import FeatureRegistryError
from signal_processing.features.aggregation.protocols import FeatureExtractor


class FeatureRegistry:
    """Ordered immutable collection of validated FeatureExtractor instances.

    Guarantees:
        - Exact constructor ordering preserved.
        - Immutable internal tuple of extractors.
        - Extractor name uniqueness enforced.
        - Structural extractor validation (name: non-empty str, version: non-empty str, extract: callable).
    """

    def __init__(self, extractors: Iterable[FeatureExtractor | Any] | None = None) -> None:
        """Construct a FeatureRegistry with the given extractors.

        Args:
            extractors: Optional iterable of FeatureExtractor instances.

        Raises:
            FeatureRegistryError: If extractors is not an iterable, any extractor fails
                validation, or duplicate extractor names are encountered.
        """
        if extractors is None:
            extractors = ()

        if not isinstance(extractors, Iterable):
            raise FeatureRegistryError(
                f"Expected iterable of extractors, got {type(extractors).__name__}"
            )

        validated: list[FeatureExtractor] = []
        seen_names: dict[str, FeatureExtractor] = {}

        for idx, ext in enumerate(extractors):
            self._validate_extractor(ext, idx)
            name = ext.name
            if name in seen_names:
                raise FeatureRegistryError(
                    f"Duplicate extractor name '{name}' detected at index {idx}; extractor names must be unique"
                )
            seen_names[name] = ext
            validated.append(ext)

        self._extractors: tuple[FeatureExtractor, ...] = tuple(validated)
        self._by_name: dict[str, FeatureExtractor] = seen_names

    @staticmethod
    def _validate_extractor(ext: Any, idx: int) -> None:
        """Validate structural compliance of an extractor instance."""
        if ext is None:
            raise FeatureRegistryError(f"Extractor at index {idx} must not be None")

        name = getattr(ext, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise FeatureRegistryError(
                f"Extractor at index {idx} must have a non-blank string 'name', got {name!r}"
            )

        version = getattr(ext, "version", None)
        if not isinstance(version, str) or not version.strip():
            raise FeatureRegistryError(
                f"Extractor '{name}' at index {idx} must have a non-blank string 'version', got {version!r}"
            )

        extract_fn = getattr(ext, "extract", None)
        if not callable(extract_fn):
            raise FeatureRegistryError(
                f"Extractor '{name}' at index {idx} must have a callable 'extract' method"
            )

    @property
    def extractors(self) -> tuple[FeatureExtractor, ...]:
        """Immutable ordered tuple of registered feature extractors."""
        return self._extractors

    @property
    def names(self) -> tuple[str, ...]:
        """Immutable tuple of extractor names in registration order."""
        return tuple(e.name for e in self._extractors)

    def get(self, name: str) -> FeatureExtractor:
        """Retrieve a registered extractor by its unique name.

        Args:
            name: Name of the feature extractor.

        Returns:
            The matching FeatureExtractor instance.

        Raises:
            FeatureRegistryError: If no extractor is registered under the given name.
        """
        if not isinstance(name, str) or name not in self._by_name:
            raise FeatureRegistryError(f"No extractor registered with name '{name}'")
        return self._by_name[name]

    def __len__(self) -> int:
        """Return the number of registered feature extractors."""
        return len(self._extractors)

    def __iter__(self) -> Iterator[FeatureExtractor]:
        """Iterate over feature extractors in registration order."""
        return iter(self._extractors)

    def __contains__(self, item: Any) -> bool:
        """Check if an extractor name or extractor instance is present in the registry."""
        if isinstance(item, str):
            return item in self._by_name
        return item in self._extractors

    def __getitem__(self, key: int | str) -> FeatureExtractor:
        """Access extractor by index or by name."""
        if isinstance(key, int):
            return self._extractors[key]
        return self.get(key)

    def __repr__(self) -> str:
        names_str = ", ".join(repr(name) for name in self.names)
        return f"FeatureRegistry([{names_str}])"
