"""Tests for unwrapped phase increment features."""

import math

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import PhaseResult, SignalEvent, StreamInfo
from signal_processing.features.phase import PhaseFeatureExtractor


class DummyPhaseContext:
    """Mock context providing phase() and mean_power_linear()."""

    def __init__(self, wrapped_rad: np.ndarray, unwrapped_rad: np.ndarray, mean_power: float = 1.0) -> None:
        self._phase = PhaseResult(wrapped_rad=wrapped_rad, unwrapped_rad=unwrapped_rad)
        self._mean_power = mean_power

    def phase(self) -> PhaseResult:
        return self._phase

    def mean_power_linear(self) -> float:
        return self._mean_power


def make_tone_event(f0: float, fs: float = 1_000_000.0, n_samples: int = 1000) -> SignalEvent:
    """Helper to create a deterministic complex exponential tone event."""
    stream = StreamInfo(
        stream_id="stream-phase-tone",
        source_id="synthetic-src",
        sample_rate_hz=fs,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(n_samples)
    iq = np.exp(1j * 2.0 * np.pi * f0 * n / fs).astype(np.complex64)

    return SignalEvent(
        event_id=f"ev-tone-{int(f0)}",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=n_samples,
        detected_start_sample=0,
        detected_end_sample=n_samples,
        trigger_sample=0,
    )


class TestPhaseIncrements:
    """Validate sample-to-sample unwrapped phase increment statistics."""

    def test_case_a_constant_phase_zero_increments(self) -> None:
        """Test Case A: Constant phase has zero increment mean, std, rms, and abs_mean."""
        wrapped = np.zeros(50, dtype=np.float64)
        unwrapped = np.zeros(50, dtype=np.float64)
        ctx = DummyPhaseContext(wrapped, unwrapped)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        for key in (
            "phase.increment_mean_rad",
            "phase.increment_std_rad",
            "phase.increment_rms_rad",
            "phase.increment_abs_mean_rad",
        ):
            fv = fset.features[key]
            assert fv.valid is True
            assert fv.unit == "rad"
            assert pytest.approx(fv.value, abs=1e-12) == 0.0

    def test_case_b_positive_complex_tone(self) -> None:
        """Test Case B: +100 kHz tone at 1 MHz sample rate has delta_phi ~= +0.2*pi rad."""
        fs = 1_000_000.0
        f0 = 100_000.0
        expected_inc = 2.0 * math.pi * f0 / fs  # ~0.6283185307 rad

        event = make_tone_event(f0=f0, fs=fs, n_samples=1000)
        ctx = SignalContext(event)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        mean_fv = fset.features["phase.increment_mean_rad"]
        std_fv = fset.features["phase.increment_std_rad"]
        rms_fv = fset.features["phase.increment_rms_rad"]
        abs_mean_fv = fset.features["phase.increment_abs_mean_rad"]

        assert mean_fv.valid is True
        assert pytest.approx(mean_fv.value, rel=1e-5) == expected_inc

        assert std_fv.valid is True
        assert pytest.approx(std_fv.value, abs=1e-4) == 0.0

        assert rms_fv.valid is True
        assert pytest.approx(rms_fv.value, rel=1e-5) == expected_inc

        assert abs_mean_fv.valid is True
        assert pytest.approx(abs_mean_fv.value, rel=1e-5) == expected_inc

    def test_case_c_negative_complex_tone(self) -> None:
        """Test Case C: -100 kHz tone at 1 MHz sample rate has negative mean increment and positive rms."""
        fs = 1_000_000.0
        f0 = -100_000.0
        expected_mag = 2.0 * math.pi * abs(f0) / fs
        expected_mean = -expected_mag

        event = make_tone_event(f0=f0, fs=fs, n_samples=1000)
        ctx = SignalContext(event)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        mean_fv = fset.features["phase.increment_mean_rad"]
        std_fv = fset.features["phase.increment_std_rad"]
        rms_fv = fset.features["phase.increment_rms_rad"]
        abs_mean_fv = fset.features["phase.increment_abs_mean_rad"]

        assert mean_fv.valid is True
        assert pytest.approx(mean_fv.value, rel=1e-5) == expected_mean

        assert std_fv.valid is True
        assert pytest.approx(std_fv.value, abs=1e-4) == 0.0

        assert rms_fv.valid is True
        assert pytest.approx(rms_fv.value, rel=1e-5) == expected_mag

        assert abs_mean_fv.valid is True
        assert pytest.approx(abs_mean_fv.value, rel=1e-5) == expected_mag

    def test_case_g_known_increment_vector(self) -> None:
        """Test Case G: Synthetic unwrapped phase [0.0, 0.1, 0.3, 0.6] with exact formula checks."""
        unwrapped = np.array([0.0, 0.1, 0.3, 0.6], dtype=np.float64)
        wrapped = unwrapped.copy()
        ctx = DummyPhaseContext(wrapped, unwrapped)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        # delta = [0.1, 0.2, 0.3]
        # mean = (0.1 + 0.2 + 0.3) / 3 = 0.2
        # pop std = sqrt(((0.1 - 0.2)^2 + 0 + (0.3 - 0.2)^2) / 3) = sqrt(0.02 / 3)
        # rms = sqrt((0.01 + 0.04 + 0.09) / 3) = sqrt(0.14 / 3)
        # abs_mean = 0.2
        expected_mean = 0.2
        expected_std = math.sqrt(0.02 / 3.0)
        expected_rms = math.sqrt(0.14 / 3.0)
        expected_abs_mean = 0.2

        assert pytest.approx(fset.features["phase.increment_mean_rad"].value, rel=1e-12) == expected_mean
        assert pytest.approx(fset.features["phase.increment_std_rad"].value, rel=1e-12) == expected_std
        assert pytest.approx(fset.features["phase.increment_rms_rad"].value, rel=1e-12) == expected_rms
        assert pytest.approx(fset.features["phase.increment_abs_mean_rad"].value, rel=1e-12) == expected_abs_mean
