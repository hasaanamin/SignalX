"""Regression tests for array immutability and arithmetic safety.

Requirement 9:
- Hardening against mutation: attempting setflags(write=True) raises ValueError.
- Caller-owned array mutation after contract instantiation does not mutate contract data.
- Power calculation promotion prevents complex64 squaring overflow on large finite values ([1e20+0j]).
- No leaking of Inf as valid output.
"""

import numpy as np
import pytest

from signal_processing.core import (
    IQBlock,
    SignalEvent,
    StreamInfo,
)
from signal_processing.transforms import (
    compute_envelope,
    compute_fft,
    compute_phase,
    compute_power,
)


@pytest.fixture
def stream_info() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-safety",
        source_id="synthetic",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


class TestStrictArrayImmutability:
    """Verify that stored arrays cannot be modified in-place or via setflags(write=True)."""

    def test_iqblock_immutable_against_setflags(self, stream_info: StreamInfo) -> None:
        """Attempting setflags(write=True) on IQBlock.iq must raise ValueError."""
        raw = np.array([1.0 + 1j, 2.0 + 2j], dtype=np.complex64)
        block = IQBlock(
            stream=stream_info,
            iq=raw,
            start_sample=0,
            sequence_number=0,
            timestamp_ns=None,
        )

        assert block.iq.flags.writeable is False
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            block.iq.setflags(write=True)

    def test_signalevent_immutable_against_setflags(self, stream_info: StreamInfo) -> None:
        """Attempting setflags(write=True) on SignalEvent.iq and signal_iq must raise ValueError."""
        raw = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-immut",
            stream=stream_info,
            iq=raw,
            event_start_sample=0,
            event_end_sample=3,
            detected_start_sample=1,
            detected_end_sample=2,
            trigger_sample=1,
        )

        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            event.iq.setflags(write=True)

        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            event.signal_iq.setflags(write=True)

    def test_transform_results_immutable_against_setflags(self) -> None:
        """Transform arrays (envelope, power, phase, FFT, PSD) cannot be made writable."""
        raw = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j, 4.0 + 4j], dtype=np.complex64)

        env = compute_envelope(raw)
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            env.setflags(write=True)

        p = compute_power(raw)
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            p.setflags(write=True)

        phase_res = compute_phase(raw)
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            phase_res.wrapped_rad.setflags(write=True)
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            phase_res.unwrapped_rad.setflags(write=True)

        fft_res = compute_fft(raw, sample_rate_hz=1e6)
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            fft_res.spectrum.setflags(write=True)

    def test_caller_mutation_isolation(self, stream_info: StreamInfo) -> None:
        """Mutating the caller-owned input array after passing to contract does not mutate stored data."""
        raw = np.array([1.0 + 1j, 2.0 + 2j], dtype=np.complex64)
        block = IQBlock(
            stream=stream_info,
            iq=raw,
            start_sample=0,
            sequence_number=0,
            timestamp_ns=None,
        )

        # Caller alters their original array
        raw[0] = 99.0 + 99.0j

        # Stored data remains uncorrupted
        assert block.iq[0] == 1.0 + 1.0j


class TestArithmeticSafety:
    """Validate numerical stability and absence of arithmetic overflow on finite inputs."""

    def test_large_complex64_no_power_overflow(self) -> None:
        """Finite complex64 [1e20 + 0j] squared in complex64 would overflow to inf.

        Promoting to float64/complex128 prevents overflow and yields finite power 1e40.
        """
        large_iq = np.array([1e20 + 0j], dtype=np.complex64)
        # Verify that raw complex64 abs squared indeed overflows to Inf
        raw_c64_sq = np.abs(large_iq[0]) ** 2
        assert np.isinf(raw_c64_sq), "Expected naive float32 squaring of 1e20 to overflow to inf"

        # compute_power must handle it safely using double precision
        p = compute_power(large_iq)
        assert np.all(np.isfinite(p))
        assert pytest.approx(float(p[0]), rel=1e-5) == 1e40

    def test_large_complex64_envelope_finite(self) -> None:
        """Finite large complex64 values have finite envelope."""
        large_iq = np.array([1e20 + 1e20j], dtype=np.complex64)
        env = compute_envelope(large_iq)
        assert np.all(np.isfinite(env))
        expected_mag = np.sqrt(2.0) * 1e20
        assert pytest.approx(float(env[0]), rel=1e-5) == expected_mag
