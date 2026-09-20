"""Tests for instantaneous frequency calculation primitive."""

import numpy as np
import pytest

from signal_processing.core import PhaseResult
from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.frequency import compute_instantaneous_frequency
from signal_processing.transforms.phase import compute_phase


class TestInstantaneousFrequency:
    """Test instantaneous frequency computation from PhaseResult."""

    def test_positive_frequency_tone(self) -> None:
        """Tone at +100 kHz sampled at 1 MHz must yield +100 kHz."""
        fs = 1_000_000.0
        f0 = 100_000.0
        n = np.arange(500)
        iq = np.exp(1j * 2.0 * np.pi * f0 * n / fs).astype(np.complex128)

        phase = compute_phase(iq)
        f_inst = compute_instantaneous_frequency(phase, fs)

        assert len(f_inst) == len(iq) - 1
        assert np.isrealobj(f_inst)
        assert np.all(np.isfinite(f_inst))
        np.testing.assert_allclose(f_inst, f0, atol=1e-6)

    def test_negative_frequency_tone(self) -> None:
        """Tone at -100 kHz sampled at 1 MHz must yield -100 kHz."""
        fs = 1_000_000.0
        f0 = -100_000.0
        n = np.arange(500)
        iq = np.exp(1j * 2.0 * np.pi * f0 * n / fs).astype(np.complex128)

        phase = compute_phase(iq)
        f_inst = compute_instantaneous_frequency(phase, fs)

        assert len(f_inst) == len(iq) - 1
        np.testing.assert_allclose(f_inst, f0, atol=1e-6)

    def test_chirp_frequency_progression(self) -> None:
        """Linear frequency chirp must produce linearly increasing instantaneous frequency."""
        fs = 100_000.0
        n = np.arange(200)
        f_start = 10_000.0
        f_slope = 50.0  # Hz per sample
        # Continuous phase: phi[n] = 2*pi * (f_start * n + 0.5 * f_slope * n^2) / fs
        phi = 2.0 * np.pi * (f_start * n + 0.5 * f_slope * (n ** 2)) / fs
        iq = np.exp(1j * phi).astype(np.complex128)

        phase = compute_phase(iq)
        f_inst = compute_instantaneous_frequency(phase, fs)

        # Expected instantaneous frequency at intervals: f_start + f_slope * (n + 0.5)
        expected_f = f_start + f_slope * (n[:-1] + 0.5)
        np.testing.assert_allclose(f_inst, expected_f, atol=1e-5)

    def test_output_properties_and_immutability(self) -> None:
        """Instantaneous frequency output must be read-only."""
        wrapped = np.array([0.0, 1.0, 2.0])
        unwrapped = np.array([0.0, 1.0, 2.0])
        phase = PhaseResult(wrapped_rad=wrapped, unwrapped_rad=unwrapped)

        f_inst = compute_instantaneous_frequency(phase, 1000.0)
        assert f_inst.flags.writeable is False

        with pytest.raises((ValueError, TypeError)):
            f_inst[0] = 0.0  # type: ignore[misc]


class TestInstantaneousFrequencyValidation:
    """Test validation of inputs to compute_instantaneous_frequency."""

    def test_reject_non_phaseresult(self) -> None:
        with pytest.raises(TransformError, match="PhaseResult"):
            compute_instantaneous_frequency(np.array([1.0, 2.0]), 1000.0)  # type: ignore[arg-type]

    def test_reject_short_phase_under_2_samples(self) -> None:
        phase = PhaseResult(
            wrapped_rad=np.array([0.0]),
            unwrapped_rad=np.array([0.0]),
        )
        with pytest.raises(TransformError, match="at least 2 phase samples"):
            compute_instantaneous_frequency(phase, 1000.0)

    @pytest.mark.parametrize("bad_fs", [0.0, -100.0, float("nan"), float("inf"), "1000", True])
    def test_reject_invalid_sample_rate(self, bad_fs: object) -> None:
        phase = PhaseResult(
            wrapped_rad=np.array([0.0, 1.0]),
            unwrapped_rad=np.array([0.0, 1.0]),
        )
        with pytest.raises(TransformError):
            compute_instantaneous_frequency(phase, bad_fs)  # type: ignore[arg-type]
