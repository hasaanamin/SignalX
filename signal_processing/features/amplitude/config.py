"""Configuration for amplitude feature extraction."""

from dataclasses import dataclass

import numpy as np

from signal_processing.features.amplitude.exceptions import AmplitudeFeatureError


@dataclass(frozen=True)
class AmplitudeFeatureConfig:
    """Immutable configuration for amplitude feature extraction.

    Attributes:
        ddof: Delta degrees of freedom for standard deviation and variance calculations
            (default: 0 for population statistics).
    """

    ddof: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.ddof, bool) or not isinstance(self.ddof, (int, np.integer)) or self.ddof < 0:
            raise AmplitudeFeatureError(f"ddof must be a non-negative integer, got {self.ddof!r}")
        object.__setattr__(self, "ddof", int(self.ddof))
