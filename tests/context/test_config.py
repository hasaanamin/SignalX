"""Tests for AnalysisConfig immutability, defaults, and validation."""

from dataclasses import FrozenInstanceError

import pytest

from signal_processing.context import AnalysisConfig, ContextError


class TestAnalysisConfig:
    """Test defaults, customization, immutability, and validation of AnalysisConfig."""

    def test_default_values(self) -> None:
        """Verify standard default parameters for AnalysisConfig."""
        config = AnalysisConfig()

        assert config.nfft is None
        assert config.spectral_window == "hann"
        assert config.centered_spectrum is True
        assert config.power_floor_dbfs == -300.0
        assert config.psd_floor_dbfs_per_hz == -300.0
        assert config.autocorrelation_max_lag_samples is None
        assert config.autocorrelation_normalized is True

    def test_custom_values(self) -> None:
        """Verify custom configurations are properly set and types normalized."""
        config = AnalysisConfig(
            nfft=2048,
            spectral_window="Blackman",
            centered_spectrum=False,
            power_floor_dbfs=-150.0,
            psd_floor_dbfs_per_hz=-200.0,
            autocorrelation_max_lag_samples=64,
            autocorrelation_normalized=False,
        )

        assert config.nfft == 2048
        assert config.spectral_window == "blackman"
        assert config.centered_spectrum is False
        assert config.power_floor_dbfs == -150.0
        assert config.psd_floor_dbfs_per_hz == -200.0
        assert config.autocorrelation_max_lag_samples == 64
        assert config.autocorrelation_normalized is False

    @pytest.mark.parametrize(
        ("win_input", "expected_normalized"),
        [
            ("rectangular", "rectangular"),
            ("RECTANGULAR", "rectangular"),
            ("  hann  ", "hann"),
            ("Hamming", "hamming"),
            ("BLACKMAN", "blackman"),
        ],
    )
    def test_supported_windows_case_insensitive(
        self, win_input: str, expected_normalized: str
    ) -> None:
        """All supported windows must be canonicalized to lowercase strings."""
        config = AnalysisConfig(spectral_window=win_input)
        assert config.spectral_window == expected_normalized

    def test_immutability(self) -> None:
        """AnalysisConfig must be frozen and reject attribute mutation."""
        config = AnalysisConfig()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            config.nfft = 1024  # type: ignore[misc]

        with pytest.raises((FrozenInstanceError, AttributeError)):
            config.spectral_window = "hamming"  # type: ignore[misc]

    @pytest.mark.parametrize("bad_nfft", [0, -1, -512, True, False, 1.5, "1024", [1024]])
    def test_reject_invalid_nfft(self, bad_nfft: object) -> None:
        """nfft must be a strictly positive integer or None."""
        with pytest.raises(ContextError, match="nfft"):
            AnalysisConfig(nfft=bad_nfft)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_window",
        ["chebyshev", "kaiser", "flattop", "gaussian", "", "   ", None, 123, True],
    )
    def test_reject_unsupported_window(self, bad_window: object) -> None:
        """spectral_window must be one of the supported transform windows."""
        with pytest.raises(ContextError, match="spectral_window"):
            AnalysisConfig(spectral_window=bad_window)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_centered", [1, 0, "True", "False", None, [True]])
    def test_reject_invalid_centered_spectrum(self, bad_centered: object) -> None:
        """centered_spectrum must be a boolean."""
        with pytest.raises(ContextError, match="centered_spectrum"):
            AnalysisConfig(centered_spectrum=bad_centered)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_power_floor",
        [True, False, float("nan"), float("inf"), float("-inf"), "floor", [100.0]],
    )
    def test_reject_invalid_power_floor_dbfs(self, bad_power_floor: object) -> None:
        """power_floor_dbfs must be a finite numeric float."""
        with pytest.raises(ContextError, match="power_floor_dbfs"):
            AnalysisConfig(power_floor_dbfs=bad_power_floor)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_psd_floor",
        [True, False, float("nan"), float("inf"), float("-inf"), "floor", None],
    )
    def test_reject_invalid_psd_floor_dbfs_per_hz(self, bad_psd_floor: object) -> None:
        """psd_floor_dbfs_per_hz must be a finite numeric float."""
        with pytest.raises(ContextError, match="psd_floor_dbfs_per_hz"):
            AnalysisConfig(psd_floor_dbfs_per_hz=bad_psd_floor)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_lag",
        [-1, -10, True, False, 1.5, "64", [64]],
    )
    def test_reject_invalid_autocorrelation_max_lag(self, bad_lag: object) -> None:
        """autocorrelation_max_lag_samples must be a non-negative integer or None."""
        with pytest.raises(ContextError, match="autocorrelation_max_lag_samples"):
            AnalysisConfig(autocorrelation_max_lag_samples=bad_lag)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_normalized",
        [1, 0, "yes", "no", None, [True]],
    )
    def test_reject_invalid_autocorrelation_normalized(self, bad_normalized: object) -> None:
        """autocorrelation_normalized must be a boolean."""
        with pytest.raises(ContextError, match="autocorrelation_normalized"):
            AnalysisConfig(autocorrelation_normalized=bad_normalized)  # type: ignore[arg-type]
