"""Regression tests for phase/frequency precision, undefined phase, and rotation invariance.

Requirements 1 & 2:
- Promotion before angle and unwrap to complex128/float64.
- 1M-sample complex64 tone reproduction: frequency mean error and std dev < 1 Hz.
- Explicit phase sample validity and adjacent-pair validity.
- Mixed-zero inputs and global rotation invariance.
- Configurable phase amplitude threshold.
- Instantaneous frequency array with gaps handling.
"""

import math

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import SignalEvent, StreamInfo
from signal_processing.features.phase import PhaseFeatureExtractor
from signal_processing.transforms import (
    TransformError,
    compute_instantaneous_frequency,
    compute_phase,
)


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-precision-test",
        source_id="synthetic",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


class TestPhaseFrequencyPrecision:
    """Validate phase/frequency precision on canonical complex64 inputs."""

    def test_reproduction_1m_sample_tone_precision(self, stream_1mhz: StreamInfo) -> None:
        """Reproduction: exp(2j*pi*0.2*arange(1_000_000)).astype(complex64), Fs = 1 MHz.

        Actual frequency is 200 kHz.
        Frequency mean error and standard deviation must each be below 1 Hz.
        """
        fs = 1_000_000.0
        f0 = 200_000.0
        n_samples = 1_000_000
        n = np.arange(n_samples, dtype=np.float64)

        # Generate as complex64 canonical capture
        iq_c64 = np.exp(2j * np.pi * (f0 / fs) * n).astype(np.complex64)

        event = SignalEvent(
            event_id="ev-precision-1m",
            stream=stream_1mhz,
            iq=iq_c64,
            event_start_sample=0,
            event_end_sample=n_samples,
            detected_start_sample=0,
            detected_end_sample=n_samples,
            trigger_sample=0,
        )

        ctx = SignalContext(event)
        inst_freq = ctx.instantaneous_frequency()

        assert len(inst_freq) == n_samples - 1
        assert inst_freq.dtype == np.float64

        mean_freq = float(np.mean(inst_freq))
        std_freq = float(np.std(inst_freq))

        mean_error_hz = abs(mean_freq - f0)
        assert mean_error_hz < 1.0, f"Mean frequency error {mean_error_hz} Hz >= 1.0 Hz"
        assert std_freq < 1.0, f"Frequency standard deviation {std_freq} Hz >= 1.0 Hz"

    def test_negative_frequency_tone_precision(self, stream_1mhz: StreamInfo) -> None:
        """Negative frequency tone: -350 kHz at Fs = 1 MHz."""
        fs = 1_000_000.0
        f0 = -350_000.0
        n_samples = 100_000
        n = np.arange(n_samples, dtype=np.float64)

        iq_c64 = np.exp(2j * np.pi * (f0 / fs) * n).astype(np.complex64)
        event = SignalEvent(
            event_id="ev-neg-tone",
            stream=stream_1mhz,
            iq=iq_c64,
            event_start_sample=0,
            event_end_sample=n_samples,
            detected_start_sample=0,
            detected_end_sample=n_samples,
            trigger_sample=0,
        )

        ctx = SignalContext(event)
        inst_freq = ctx.instantaneous_frequency()
        mean_freq = float(np.mean(inst_freq))
        std_freq = float(np.std(inst_freq))

        assert abs(mean_freq - f0) < 1.0
        assert std_freq < 1.0

    def test_linear_chirp_precision(self, stream_1mhz: StreamInfo) -> None:
        """Linear chirp from -200 kHz to +200 kHz over 100,000 samples."""
        fs = 1_000_000.0
        n_samples = 100_000
        t = np.arange(n_samples, dtype=np.float64) / fs
        t_end = (n_samples - 1) / fs
        f_start = -200_000.0
        f_end = 200_000.0
        k = (f_end - f_start) / t_end

        # Phase integral: 2*pi * (f_start * t + 0.5 * k * t^2)
        phi = 2.0 * np.pi * (f_start * t + 0.5 * k * (t ** 2))
        iq = np.exp(1j * phi).astype(np.complex64)

        event = SignalEvent(
            event_id="ev-chirp",
            stream=stream_1mhz,
            iq=iq,
            event_start_sample=0,
            event_end_sample=n_samples,
            detected_start_sample=0,
            detected_end_sample=n_samples,
            trigger_sample=0,
        )

        ctx = SignalContext(event)
        inst_freq = ctx.instantaneous_frequency()

        # Midpoint times for N-1 intervals
        t_mid = 0.5 * (t[:-1] + t[1:])
        expected_freq = f_start + k * t_mid

        max_err = float(np.max(np.abs(inst_freq - expected_freq)))
        assert max_err < 1.0, f"Max chirp frequency error {max_err} Hz >= 1.0 Hz"


class TestUndefinedPhaseAndRotationInvariance:
    """Validate explicit handling of undefined phase, gaps, and rotation invariance."""

    def test_mixed_zero_rotation_invariance(self, stream_1mhz: StreamInfo) -> None:
        """[1j, 0, 1j, 0, 1j] and its rotated copy exp(1j * pi / 4) * [...]

        Must give equal valid circular features (R=1.0, Var=0.0) and identical invalid
        increment features.
        """
        iq1 = np.array([1.0j, 0.0, 1.0j, 0.0, 1.0j], dtype=np.complex64)
        rot_angle = np.pi / 4.0
        iq2 = (iq1 * np.exp(1j * rot_angle)).astype(np.complex64)

        event1 = SignalEvent(
            event_id="ev-zero-1",
            stream=stream_1mhz,
            iq=iq1,
            event_start_sample=0,
            event_end_sample=5,
            detected_start_sample=0,
            detected_end_sample=5,
            trigger_sample=0,
        )
        event2 = SignalEvent(
            event_id="ev-zero-2",
            stream=stream_1mhz,
            iq=iq2,
            event_start_sample=0,
            event_end_sample=5,
            detected_start_sample=0,
            detected_end_sample=5,
            trigger_sample=0,
        )

        extractor = PhaseFeatureExtractor()
        fset1 = extractor.extract(SignalContext(event1))
        fset2 = extractor.extract(SignalContext(event2))

        # Circular features are valid and invariant
        r1 = fset1.features["phase.circular_resultant_length"]
        r2 = fset2.features["phase.circular_resultant_length"]
        assert r1.valid is True
        assert r2.valid is True
        assert pytest.approx(r1.value, abs=1e-6) == 1.0
        assert pytest.approx(r2.value, abs=1e-6) == 1.0

        v1 = fset1.features["phase.circular_variance"]
        v2 = fset2.features["phase.circular_variance"]
        assert v1.valid is True
        assert v2.valid is True
        assert pytest.approx(v1.value, abs=1e-6) == 0.0
        assert pytest.approx(v2.value, abs=1e-6) == 0.0

        # Increment features are invalid because there are NO adjacent valid pairs (zeros in between)
        for key in [
            "phase.increment_mean_rad",
            "phase.increment_std_rad",
            "phase.increment_rms_rad",
            "phase.increment_abs_mean_rad",
        ]:
            fv1 = fset1.features[key]
            fv2 = fset2.features[key]
            assert fv1.valid is False
            assert fv2.valid is False
            assert math.isnan(fv1.value)
            assert math.isnan(fv2.value)

    def test_configurable_amplitude_threshold(self, stream_1mhz: StreamInfo) -> None:
        """Phase samples with amplitude below threshold are masked out as invalid."""
        # Samples with amplitudes: 0.01, 1.0, 0.005, 1.0
        iq = np.array([0.01 + 0j, 1.0 + 0j, 0.005 + 0j, 1.0 + 0j], dtype=np.complex64)

        # Threshold 0.05 masks samples 0 and 2
        phase_res = compute_phase(iq, amplitude_threshold=0.05)
        assert phase_res.valid_mask is not None
        np.testing.assert_array_equal(phase_res.valid_mask, [False, True, False, True])

        # Without threshold (0.0), all non-zero samples are valid
        phase_res_zero = compute_phase(iq, amplitude_threshold=0.0)
        np.testing.assert_array_equal(phase_res_zero.valid_mask, [True, True, True, True])

    def test_instantaneous_frequency_gaps_policy(self, stream_1mhz: StreamInfo) -> None:
        """With gaps, instantaneous frequency returns NaN when allow_gaps=True, raises TransformError when allow_gaps=False."""
        iq = np.array([1.0 + 0j, 0.0, 1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
        phase_res = compute_phase(iq, amplitude_threshold=0.0)

        # allow_gaps=False raises TransformError
        with pytest.raises(TransformError, match="undefined across gaps"):
            compute_instantaneous_frequency(phase_res, stream_1mhz.sample_rate_hz, allow_gaps=False)

        # allow_gaps=True produces aligned array with NaN on invalid intervals
        ifreq = compute_instantaneous_frequency(phase_res, stream_1mhz.sample_rate_hz, allow_gaps=True)
        assert len(ifreq) == len(iq) - 1
        assert math.isnan(ifreq[0])  # interval (0, 1) has zero at 1
        assert math.isnan(ifreq[1])  # interval (1, 2) has zero at 1
        assert not math.isnan(ifreq[2])  # interval (2, 3) has both valid
