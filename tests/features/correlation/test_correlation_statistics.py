"""Tests for mean and RMS nonzero autocorrelation magnitude statistics."""

import numpy as np
import pytest

from signal_processing.core import AutocorrelationResult
from signal_processing.features.correlation import CorrelationFeatureExtractor


class DummyCorrelationContext:
    def __init__(
        self,
        lag_samples: np.ndarray,
        values: np.ndarray,
        sample_rate_hz: float = 1000.0,
        mean_power: float = 1.0,
    ) -> None:
        self._ac = AutocorrelationResult(lag_samples=lag_samples, values=values)
        self.sample_rate_hz = sample_rate_hz
        self._mean_power = mean_power

    def autocorrelation(self) -> AutocorrelationResult:
        return self._ac

    def mean_power_linear(self) -> float:
        return self._mean_power


class TestCorrelationStatistics:
    """Test mean and RMS normalized autocorrelation magnitude calculations across lags > 0."""

    def test_case_l_mean_and_rms_nonzero_magnitude(self) -> None:
        """Test Case L: Independent calculation of mean and RMS magnitudes across positive lags."""
        lags = np.array([0, 1, 2, 3], dtype=np.int64)
        values = np.array([1.0 + 0.0j, 0.2 + 0.0j, 0.8 + 0.0j, 0.4 + 0.0j], dtype=np.complex128)
        ctx = DummyCorrelationContext(lags, values)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        mean_fv = fset.features["correlation.mean_nonzero_magnitude"]
        rms_fv = fset.features["correlation.rms_nonzero_magnitude"]

        expected_mean = (0.2 + 0.8 + 0.4) / 3.0
        expected_rms = np.sqrt((0.2**2 + 0.8**2 + 0.4**2) / 3.0)

        assert mean_fv.valid is True
        assert mean_fv.unit is None
        assert isinstance(mean_fv.value, float)
        assert pytest.approx(mean_fv.value) == expected_mean

        assert rms_fv.valid is True
        assert rms_fv.unit is None
        assert isinstance(rms_fv.value, float)
        assert pytest.approx(rms_fv.value) == expected_rms

    def test_mean_and_rms_with_complex_phases(self) -> None:
        """Verify magnitude statistics correctly use absolute values for complex autocorrelation coefficients."""
        lags = np.array([0, 1, 2, 3], dtype=np.int64)
        # Amplitudes are [1.0, 0.3, 0.6, 0.9] with arbitrary complex phases
        values = np.array(
            [
                1.0 + 0.0j,
                0.3 * np.exp(1j * 0.5),
                0.6 * np.exp(1j * -1.2),
                0.9 * np.exp(1j * 2.1),
            ],
            dtype=np.complex128,
        )
        ctx = DummyCorrelationContext(lags, values)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        mean_fv = fset.features["correlation.mean_nonzero_magnitude"]
        rms_fv = fset.features["correlation.rms_nonzero_magnitude"]

        expected_mean = (0.3 + 0.6 + 0.9) / 3.0
        expected_rms = np.sqrt((0.3**2 + 0.6**2 + 0.9**2) / 3.0)

        assert pytest.approx(mean_fv.value) == expected_mean
        assert pytest.approx(rms_fv.value) == expected_rms
