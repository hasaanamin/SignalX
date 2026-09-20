"""Tests for AmplitudeFeatureConfig immutability and validation."""

from dataclasses import FrozenInstanceError

import pytest

from signal_processing.features.amplitude import (
    AmplitudeFeatureConfig,
    AmplitudeFeatureError,
)


class TestAmplitudeConfig:
    """Test defaults, customization, immutability, and validation of AmplitudeFeatureConfig."""

    def test_default_config(self) -> None:
        config = AmplitudeFeatureConfig()
        assert config.ddof == 0

    def test_custom_config(self) -> None:
        config = AmplitudeFeatureConfig(ddof=1)
        assert config.ddof == 1

    def test_immutability(self) -> None:
        config = AmplitudeFeatureConfig()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            config.ddof = 1  # type: ignore[misc]

    @pytest.mark.parametrize("bad_ddof", [-1, -5, True, False, 1.5, "0", None, [0]])
    def test_reject_invalid_ddof(self, bad_ddof: object) -> None:
        with pytest.raises(AmplitudeFeatureError, match="ddof"):
            AmplitudeFeatureConfig(ddof=bad_ddof)  # type: ignore[arg-type]
