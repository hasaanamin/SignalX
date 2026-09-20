"""Tests for nonzero autocorrelation peak magnitude, lag (samples and seconds), and phase."""

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


class TestPeakCorrelation:
    """Test nonzero-lag peak extraction, units, and phase conventions."""

    def test_case_a_known_fake_autocorrelation(self) -> None:
        """Test Case A: Zero lag is excluded; peak at lag 2 with exact magnitude, lag, seconds, and phase."""
        lags = np.array([0, 1, 2, 3], dtype=np.int64)
        values = np.array([1.0 + 0.0j, 0.2 + 0.0j, 0.8 + 0.0j, 0.4 + 0.0j], dtype=np.complex128)
        ctx = DummyCorrelationContext(lags, values, sample_rate_hz=1000.0, mean_power=1.0)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        # Lag 0 has magnitude 1.0, but zero-lag MUST be excluded!
        # Max in positive lags is at lag 2 with magnitude 0.8
        mag_fv = fset.features["correlation.nonzero_peak_magnitude"]
        lag_smp_fv = fset.features["correlation.nonzero_peak_lag_samples"]
        lag_s_fv = fset.features["correlation.nonzero_peak_lag_s"]
        phase_fv = fset.features["correlation.nonzero_peak_phase_rad"]

        assert mag_fv.valid is True
        assert mag_fv.unit is None
        assert isinstance(mag_fv.value, float)
        assert pytest.approx(mag_fv.value) == 0.8

        assert lag_smp_fv.valid is True
        assert lag_smp_fv.unit == "samples"
        assert isinstance(lag_smp_fv.value, float)
        assert pytest.approx(lag_smp_fv.value) == 2.0

        assert lag_s_fv.valid is True
        assert lag_s_fv.unit == "s"
        assert isinstance(lag_s_fv.value, float)
        assert pytest.approx(lag_s_fv.value) == 0.002

        assert phase_fv.valid is True
        assert phase_fv.unit == "rad"
        assert isinstance(phase_fv.value, float)
        assert pytest.approx(phase_fv.value, abs=1e-12) == 0.0

    def test_case_c_complex_phase_at_peak(self) -> None:
        """Test Case C: Peak at lag 2 has complex phase angle equal to pi/3."""
        lags = np.array([0, 1, 2, 3], dtype=np.int64)
        phase_expected = np.pi / 3.0
        values = np.array(
            [
                1.0 + 0.0j,
                0.2 + 0.0j,
                0.5 * np.exp(1j * phase_expected),
                0.1 + 0.0j,
            ],
            dtype=np.complex128,
        )
        ctx = DummyCorrelationContext(lags, values, sample_rate_hz=2000.0)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        mag_fv = fset.features["correlation.nonzero_peak_magnitude"]
        phase_fv = fset.features["correlation.nonzero_peak_phase_rad"]

        assert pytest.approx(mag_fv.value) == 0.5
        assert pytest.approx(phase_fv.value) == phase_expected

    def test_case_d_deterministic_tie_behavior(self) -> None:
        """Test Case D: Multiple positive lags with equal maximum magnitude selects the first (smallest) lag."""
        lags = np.array([0, 1, 2, 3], dtype=np.int64)
        values = np.array([1.0 + 0.0j, 0.7 + 0.0j, 0.7 + 0.0j, 0.2 + 0.0j], dtype=np.complex128)
        ctx = DummyCorrelationContext(lags, values, sample_rate_hz=1000.0)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        lag_smp_fv = fset.features["correlation.nonzero_peak_lag_samples"]
        assert pytest.approx(lag_smp_fv.value) == 1.0

    def test_case_f_complex_tone_phase_convention(self) -> None:
        """Test Case F: Complex tone x[n] = exp(j*2*pi*f0*n/Fs) yields phase 2*pi*f0*k/Fs at lag k=1."""
        fs = 1_000_000.0
        f0 = 100_000.0
        # Positive lag autocorrelation of tone: Rxx[k] = exp(j * 2 * pi * f0 * k / fs)
        lags = np.array([0, 1, 2], dtype=np.int64)
        k = lags.astype(np.float64)
        values = np.exp(1j * 2.0 * np.pi * f0 * k / fs)
        ctx = DummyCorrelationContext(lags, values, sample_rate_hz=fs)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        phase_fv = fset.features["correlation.nonzero_peak_phase_rad"]
        expected_phase = 2.0 * np.pi * f0 * 1.0 / fs  # 0.2 * pi = 0.62831853...
        assert pytest.approx(phase_fv.value) == expected_phase
