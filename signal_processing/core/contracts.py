"""Core domain contracts for signal processing.

All domain contracts in this module are immutable (frozen) value objects.
They enforce strict validation on construction to guarantee domain invariants:
    - Half-open interval semantics: [start_sample, end_sample)
    - Non-negative absolute stream sample coordinates
    - One-dimensional complex64 IQ arrays, stored read-only and defensively copied
    - Explicit frequency (Hz), time (s, ns), power (dBFS), and phase (rad) units
    - No data leakage: no ground-truth information is accepted in these contracts
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from signal_processing.core.enums import SegmentState, SegmentTransition
from signal_processing.core.exceptions import ContractValidationError

# ---------------------------------------------------------------------------
# Internal Validation Helpers
# ---------------------------------------------------------------------------

def _validate_non_blank_str(value: Any, field_name: str) -> str:
    """Validate that value is a non-empty, non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            f"Field '{field_name}' must be a non-blank string, got {value!r}"
        )
    return value


def _validate_non_negative_int(value: Any, field_name: str) -> int:
    """Validate that value is an integer >= 0 (strictly excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ContractValidationError(
            f"Field '{field_name}' must be an integer >= 0, got {value!r}"
        )
    return int(value)


def _validate_finite_float(value: Any, field_name: str) -> float:
    """Validate that value is a finite float or int (strictly excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ContractValidationError(
            f"Field '{field_name}' must be a numeric float, got {value!r}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ContractValidationError(
            f"Field '{field_name}' must be finite, got {fval}"
        )
    return fval


def _make_array_immutable(arr: np.ndarray) -> np.ndarray:
    """Create an immutable numpy array view backed by Python bytes.

    This ensures that the array cannot be made writeable even by calling
    `setflags(write=True)` and cannot be mutated by caller-owned external buffers.
    """
    c_arr = np.ascontiguousarray(arr)
    frozen = np.frombuffer(c_arr.tobytes(), dtype=c_arr.dtype).reshape(c_arr.shape)
    return frozen


def _validate_and_freeze_iq_array(iq: Any, field_name: str = "iq") -> np.ndarray:
    """Validate IQ data as 1-D, non-empty, finite np.complex64 and return an immutable copy."""
    if isinstance(iq, np.ndarray):
        if iq.dtype != np.complex64:
            raise ContractValidationError(
                f"Field '{field_name}' must have dtype np.complex64, got {iq.dtype}"
            )
        arr = iq
    else:
        try:
            arr = np.asarray(iq)
        except Exception as exc:
            raise ContractValidationError(
                f"Field '{field_name}' cannot be converted to a numpy array: {exc}"
            ) from exc
        if arr.dtype != np.complex64:
            raise ContractValidationError(
                f"Field '{field_name}' must have dtype np.complex64, got {arr.dtype}"
            )

    if arr.ndim != 1:
        raise ContractValidationError(
            f"Field '{field_name}' must be 1-dimensional, got {arr.ndim} dimensions (shape {arr.shape})"
        )
    if arr.size == 0:
        raise ContractValidationError(
            f"Field '{field_name}' must not be empty"
        )
    try:
        if not np.all(np.isfinite(arr)):
            raise ContractValidationError(
                f"Field '{field_name}' contains non-finite values (NaN or Inf)"
            )
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"Field '{field_name}' failed finiteness validation: {exc}"
        ) from exc

    return _make_array_immutable(arr)


def _validate_and_freeze_real_array(
    arr: Any,
    field_name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    """Validate 1-D, non-empty, finite, real-valued numeric array and return an immutable copy."""
    try:
        out = np.asarray(arr)
    except Exception as exc:
        raise ContractValidationError(
            f"Field '{field_name}' cannot be converted to a numpy array: {exc}"
        ) from exc

    if out.ndim != 1:
        raise ContractValidationError(
            f"Field '{field_name}' must be 1-dimensional, got {out.ndim} dimensions (shape {out.shape})"
        )
    if not allow_empty and out.size == 0:
        raise ContractValidationError(
            f"Field '{field_name}' must not be empty"
        )
    if not (
        np.issubdtype(out.dtype, np.floating) or np.issubdtype(out.dtype, np.integer)
    ) or np.iscomplexobj(out):
        raise ContractValidationError(
            f"Field '{field_name}' must be a real-valued numeric array, got dtype {out.dtype}"
        )
    try:
        if out.size > 0 and not np.all(np.isfinite(out)):
            raise ContractValidationError(
                f"Field '{field_name}' contains non-finite values (NaN or Inf)"
            )
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"Field '{field_name}' failed finiteness validation: {exc}"
        ) from exc

    return _make_array_immutable(out)


def _validate_and_freeze_integer_lag_array(
    arr: Any,
    field_name: str = "lag_samples",
) -> np.ndarray:
    """Validate 1-D, non-empty, finite integer lag sample offsets and return an immutable copy."""
    try:
        raw = np.asarray(arr)
    except Exception as exc:
        raise ContractValidationError(
            f"Field '{field_name}' cannot be converted to a numpy array: {exc}"
        ) from exc

    if raw.ndim != 1:
        raise ContractValidationError(
            f"Field '{field_name}' must be 1-dimensional, got {raw.ndim} dimensions"
        )
    if raw.size == 0:
        raise ContractValidationError(
            f"Field '{field_name}' must not be empty"
        )
    if not (
        np.issubdtype(raw.dtype, np.floating) or np.issubdtype(raw.dtype, np.integer)
    ) or np.iscomplexobj(raw):
        raise ContractValidationError(
            f"Field '{field_name}' must be an integer array, got dtype {raw.dtype}"
        )
    try:
        if not np.all(np.isfinite(raw)):
            raise ContractValidationError(
                f"Field '{field_name}' contains non-finite values (NaN or Inf)"
            )
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"Field '{field_name}' failed finiteness validation: {exc}"
        ) from exc

    if np.issubdtype(raw.dtype, np.floating):
        if not np.all(np.equal(raw, np.floor(raw))):
            raise ContractValidationError(
                f"Field '{field_name}' contains fractional values; lag samples must be integers"
            )
        out = raw.astype(np.int64)
    elif np.issubdtype(raw.dtype, np.integer):
        out = raw.astype(np.int64)
    else:
        raise ContractValidationError(
            f"Field '{field_name}' must have integer dtype, got {raw.dtype}"
        )

    return _make_array_immutable(out)


def _validate_and_freeze_numeric_1d_array(
    arr: Any,
    field_name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    """Validate 1-D, finite real or complex numeric array and return an immutable copy."""
    try:
        out = np.asarray(arr)
    except Exception as exc:
        raise ContractValidationError(
            f"Field '{field_name}' cannot be converted to a numpy array: {exc}"
        ) from exc

    if out.ndim != 1:
        raise ContractValidationError(
            f"Field '{field_name}' must be 1-dimensional, got {out.ndim} dimensions"
        )
    if not allow_empty and out.size == 0:
        raise ContractValidationError(
            f"Field '{field_name}' must not be empty"
        )
    if not (
        np.issubdtype(out.dtype, np.floating)
        or np.issubdtype(out.dtype, np.complexfloating)
        or np.issubdtype(out.dtype, np.integer)
    ):
        raise ContractValidationError(
            f"Field '{field_name}' must be a numeric array, got dtype {out.dtype}"
        )
    try:
        if out.size > 0 and not np.all(np.isfinite(out)):
            raise ContractValidationError(
                f"Field '{field_name}' contains non-finite values (NaN or Inf)"
            )
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"Field '{field_name}' failed finiteness validation: {exc}"
        ) from exc

    return _make_array_immutable(out)


# ---------------------------------------------------------------------------
# Contract 1: StreamInfo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamInfo:
    """Metadata describing an IQ stream.

    StreamInfo encapsulates parameters that remain static across contiguous
    IQ blocks emitted by a common receiver/tuner frontend.

    Attributes:
        stream_id: Unique identifier of the stream instance (non-blank).
        source_id: Identifier of the source generating the stream (non-blank).
        sample_rate_hz: Sampling frequency in Hertz (finite, strictly > 0).
        center_frequency_hz: Optional tuned RF center frequency in Hertz.
            None indicates baseband or uncalibrated RF context.
        channel_id: Optional channel identifier string.
        dtype: Canonical sample data type name. Currently only 'complex64' is supported.

    Interval and coordinate semantics:
        - All frequency parameters are in Hertz (Hz).
        - Value object is strictly immutable (frozen).
    """

    stream_id: str
    source_id: str
    sample_rate_hz: float
    center_frequency_hz: float | None = None
    channel_id: str | None = None
    dtype: str = "complex64"

    def __post_init__(self) -> None:
        _validate_non_blank_str(self.stream_id, "stream_id")
        _validate_non_blank_str(self.source_id, "source_id")

        if isinstance(self.sample_rate_hz, bool) or not isinstance(
            self.sample_rate_hz, (int, float, np.floating, np.integer)
        ):
            raise ContractValidationError("Field 'sample_rate_hz' must be a numeric float")
        sr = float(self.sample_rate_hz)
        if not math.isfinite(sr) or sr <= 0.0:
            raise ContractValidationError(
                f"Field 'sample_rate_hz' must be finite and strictly positive, got {self.sample_rate_hz}"
            )
        object.__setattr__(self, "sample_rate_hz", sr)

        if self.center_frequency_hz is not None:
            if isinstance(self.center_frequency_hz, bool) or not isinstance(
                self.center_frequency_hz, (int, float, np.floating, np.integer)
            ):
                raise ContractValidationError(
                    "Field 'center_frequency_hz' must be numeric or None"
                )
            cf = float(self.center_frequency_hz)
            if not math.isfinite(cf):
                raise ContractValidationError(
                    f"Field 'center_frequency_hz' must be finite when provided, got {self.center_frequency_hz}"
                )
            object.__setattr__(self, "center_frequency_hz", cf)

        if self.channel_id is not None:
            _validate_non_blank_str(self.channel_id, "channel_id")

        if self.dtype != "complex64":
            raise ContractValidationError(
                f"Unsupported dtype '{self.dtype}'; currently only 'complex64' is supported"
            )


# ---------------------------------------------------------------------------
# Contract 2: IQBlock
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IQBlock:
    """A contiguous chunk of complex IQ samples within an absolute stream.

    IQBlock is the canonical unit of data transferred through the streaming
    runtime pipeline (ring buffer, noise estimator, detector).

    Attributes:
        stream: Stream metadata associated with this block.
        sequence_number: Monotonically increasing block index (>= 0).
        start_sample: Absolute stream index of the first sample in this block
            (0-indexed, >= 0).
        timestamp_ns: Optional absolute receiver epoch timestamp in integer
            nanoseconds corresponding to start_sample.
        iq: One-dimensional numpy array of np.complex64 samples representing
            the half-open interval [start_sample, end_sample). Stored read-only.
        dropped_samples_before: Declared count of samples dropped immediately
            before start_sample (>= 0).
        overflow: Flag indicating hardware/buffer overflow was flagged for this block.

    Derived properties:
        num_samples: Length of the IQ array.
        end_sample: Exclusive absolute end index (start_sample + num_samples).
        duration_s: Duration of the block in seconds (num_samples / sample_rate_hz).
    """

    stream: StreamInfo
    sequence_number: int
    start_sample: int
    timestamp_ns: int | None
    iq: np.ndarray
    dropped_samples_before: int = 0
    overflow: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stream, StreamInfo):
            raise ContractValidationError(
                f"Field 'stream' must be an instance of StreamInfo, got {type(self.stream)}"
            )

        _validate_non_negative_int(self.sequence_number, "sequence_number")
        _validate_non_negative_int(self.start_sample, "start_sample")
        _validate_non_negative_int(self.dropped_samples_before, "dropped_samples_before")

        if self.timestamp_ns is not None:
            if isinstance(self.timestamp_ns, bool) or not isinstance(
                self.timestamp_ns, (int, np.integer)
            ):
                raise ContractValidationError(
                    f"Field 'timestamp_ns' must be an integer nanoseconds or None, got {self.timestamp_ns!r}"
                )
            object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))

        if not isinstance(self.overflow, bool):
            raise ContractValidationError(
                f"Field 'overflow' must be a boolean, got {self.overflow!r}"
            )

        frozen_iq = _validate_and_freeze_iq_array(self.iq, "iq")
        object.__setattr__(self, "iq", frozen_iq)

    def slice_received(self, start: int, end: int) -> IQBlock:
        """Immutable sub-block view; no second IQ copy or validation of already frozen samples.

        Coordinates are local, half-open and nonempty. Loss markers belong to the
        original transport block and are not propagated into internal subdivisions.
        Timestamp is omitted; the pipeline retains its original absolute anchor.
        """
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.iq):
            raise ContractValidationError("Invalid received sub-block range")
        result = object.__new__(IQBlock)
        for name, value in (('stream', self.stream), ('sequence_number', self.sequence_number),
                            ('start_sample', self.start_sample+start), ('timestamp_ns', None),
                            ('iq', self.iq[start:end]), ('dropped_samples_before', 0), ('overflow', False)):
            object.__setattr__(result, name, value)
        return result

    @property
    def num_samples(self) -> int:
        """Total number of IQ samples in this block."""
        return len(self.iq)

    @property
    def end_sample(self) -> int:
        """Exclusive absolute stream sample index: start_sample + num_samples."""
        return self.start_sample + len(self.iq)

    @property
    def duration_s(self) -> float:
        """Duration of this block in seconds: num_samples / sample_rate_hz."""
        return len(self.iq) / self.stream.sample_rate_hz

    def sample_offset_to_time_s(self, offset: float) -> float:
        """Convert a sample offset relative to block start into elapsed time in seconds.

        Args:
            offset: Sample offset (finite number).

        Returns:
            Elapsed time in seconds: offset / sample_rate_hz.
        """
        if isinstance(offset, bool) or not isinstance(offset, (int, float, np.integer, np.floating)):
            raise ContractValidationError("Offset must be a numeric integer or float")
        foffset = float(offset)
        if not math.isfinite(foffset):
            raise ContractValidationError(f"Offset must be finite, got {offset}")
        return foffset / self.stream.sample_rate_hz


# ---------------------------------------------------------------------------
# Contract 3: NoiseEstimate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseReferenceDomain:
    """Identity of a measurement path; power and statistical uncertainty are separate.

    gain_id identifies the caller's gain convention, not a measured hardware gain.
    A receiver gain change requires a new identity and authorized recalibration.
    """

    sample_rate_hz: float
    monitored_band_hz: tuple[float, float] | None = None
    filter_sos_hash: str | None = None
    gain_id: str = "digital_iq_unity"

    def __post_init__(self) -> None:
        fs = _validate_finite_float(self.sample_rate_hz, "sample_rate_hz")
        if fs <= 0:
            raise ContractValidationError("sample_rate_hz must be positive")
        _validate_non_blank_str(self.gain_id, "gain_id")
        if self.monitored_band_hz is not None:
            band = tuple(self.monitored_band_hz)
            if len(band) != 2 or not (-fs / 2 <= band[0] < band[1] <= fs / 2):
                raise ContractValidationError("Invalid monitored band")
            object.__setattr__(self, "monitored_band_hz", band)
            _validate_non_blank_str(self.filter_sos_hash, "filter_sos_hash")
        elif self.filter_sos_hash is not None:
            raise ContractValidationError("Full-band reference cannot have a filter hash")


@dataclass(frozen=True)
class NoiseEstimate:
    """Estimated background noise level produced by a noise estimator.

    Attributes:
        stream_id: Identifier of the stream this estimate applies to (non-blank).
        block_sequence_number: Sequence number of the block for which this estimate
            was evaluated (>= 0).
        noise_power_dbfs: Estimated average noise power in dBFS.
            Must be finite if valid=True. May be NaN if valid=False.
        valid: Whether this noise estimate is reliable and usable.
        reason: Explanatory message when valid=False; must be None when valid=True.
    """

    stream_id: str
    block_sequence_number: int
    noise_power_dbfs: float
    valid: bool = True
    reason: str | None = None
    reference_domain: NoiseReferenceDomain | None = None
    training_samples: int = 0
    uncertainty_method: str = "unspecified"

    def __post_init__(self) -> None:
        _validate_non_blank_str(self.stream_id, "stream_id")
        _validate_non_negative_int(self.block_sequence_number, "block_sequence_number")

        if self.reference_domain is not None and not isinstance(self.reference_domain, NoiseReferenceDomain):
            raise ContractValidationError("reference_domain must be a NoiseReferenceDomain")
        _validate_non_negative_int(self.training_samples, "training_samples")

        if not isinstance(self.valid, bool):
            raise ContractValidationError("Field 'valid' must be a boolean")

        if self.valid:
            npow = _validate_finite_float(self.noise_power_dbfs, "noise_power_dbfs")
            object.__setattr__(self, "noise_power_dbfs", npow)
            if self.reason is not None:
                raise ContractValidationError(
                    "Field 'reason' must be None when noise estimate is valid"
                )
        else:
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ContractValidationError(
                    "Field 'reason' must be a non-blank string explaining why noise estimate is unavailable when valid=False"
                )
            if isinstance(self.noise_power_dbfs, bool) or not isinstance(
                self.noise_power_dbfs, (int, float, np.floating, np.integer)
            ):
                raise ContractValidationError(
                    "Field 'noise_power_dbfs' must be a float even when invalid (e.g. float('nan'))"
                )
            object.__setattr__(self, "noise_power_dbfs", float(self.noise_power_dbfs))


# ---------------------------------------------------------------------------
# Contract 4: DetectionMeasurement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionMeasurement:
    """Power and SNR metrics measured by a detector across an IQ block interval.

    This represents causal observations made by an online detector.
    It does NOT represent ground truth.

    Interval:
        [start_sample, end_sample) in absolute stream coordinates.

    Attributes:
        stream_id: Stream identifier (non-blank).
        block_sequence_number: Sequence number of the processed block (>= 0).
        start_sample: Absolute stream start index (>= 0).
        end_sample: Absolute stream exclusive end index (end_sample > start_sample).
        signal_power_dbfs: Measured block signal power in dBFS (finite).
        noise_power_dbfs: Reference noise floor power in dBFS (finite).
        snr_estimate_db: Physical excess/noise SNR when snr_valid. Otherwise
            finite 0.0 placeholder, NOT a measured SNR (see snr_invalid_reason).
        on_threshold_dbfs: Threshold power required to trigger detection (finite, > off_threshold_dbfs).
        off_threshold_dbfs: Threshold power required to declare signal end (finite, < on_threshold_dbfs).
        above_on_threshold: True if and only if signal_power_dbfs >= on_threshold_dbfs.
        below_off_threshold: True if and only if signal_power_dbfs < off_threshold_dbfs.
    """

    stream_id: str
    block_sequence_number: int
    start_sample: int
    end_sample: int
    signal_power_dbfs: float
    noise_power_dbfs: float
    snr_estimate_db: float
    on_threshold_dbfs: float
    off_threshold_dbfs: float
    above_on_threshold: bool
    below_off_threshold: bool
    signal_power_linear: float | None = None
    noise_power_linear: float | None = None
    on_threshold_linear: float | None = None
    off_threshold_linear: float | None = None
    threshold_support_method: str = "model_only"
    threshold_evidence_id: str | None = None
    snr_valid: bool = True
    snr_invalid_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_non_blank_str(self.stream_id, "stream_id")
        _validate_non_negative_int(self.block_sequence_number, "block_sequence_number")
        start = _validate_non_negative_int(self.start_sample, "start_sample")
        if self.threshold_support_method not in ('model_only', 'uncertified_training_quantile', 'held_out_clopper_pearson_bonferroni'):
            raise ContractValidationError("Unknown threshold support method")
        if self.threshold_support_method == 'held_out_clopper_pearson_bonferroni':
            _validate_non_blank_str(self.threshold_evidence_id, 'threshold_evidence_id')
        elif self.threshold_evidence_id is not None:
            raise ContractValidationError("Uncertified measurements cannot claim held-out evidence")

        if isinstance(self.end_sample, bool) or not isinstance(
            self.end_sample, (int, np.integer)
        ):
            raise ContractValidationError("Field 'end_sample' must be an integer")
        end = int(self.end_sample)
        if end <= start:
            raise ContractValidationError(
                f"Field 'end_sample' ({end}) must be strictly greater than 'start_sample' ({start})"
            )
        object.__setattr__(self, "end_sample", end)

        on_th = _validate_finite_float(self.on_threshold_dbfs, "on_threshold_dbfs")
        off_th = _validate_finite_float(self.off_threshold_dbfs, "off_threshold_dbfs")
        if on_th <= off_th:
            raise ContractValidationError(
                f"Detection threshold hysteresis violated: on_threshold_dbfs ({on_th}) "
                f"must be strictly greater than off_threshold_dbfs ({off_th})"
            )
        object.__setattr__(self, "on_threshold_dbfs", on_th)
        object.__setattr__(self, "off_threshold_dbfs", off_th)

        sig_pow = _validate_finite_float(self.signal_power_dbfs, "signal_power_dbfs")
        object.__setattr__(self, "signal_power_dbfs", sig_pow)

        object.__setattr__(
            self,
            "noise_power_dbfs",
            _validate_finite_float(self.noise_power_dbfs, "noise_power_dbfs"),
        )
        object.__setattr__(
            self,
            "snr_estimate_db",
            _validate_finite_float(self.snr_estimate_db, "snr_estimate_db"),
        )
        if type(self.snr_valid) is not bool:
            raise ContractValidationError('snr_valid must be boolean')
        if self.snr_valid:
            if self.snr_invalid_reason is not None:
                raise ContractValidationError('Valid SNR cannot have an invalid reason')
        elif self.snr_invalid_reason != 'nonpositive_excess' or self.snr_estimate_db != 0.0:
            raise ContractValidationError('Unavailable SNR requires nonpositive_excess and a 0.0 placeholder')

        if not isinstance(self.above_on_threshold, bool):
            raise ContractValidationError("Field 'above_on_threshold' must be a boolean")
        if not isinstance(self.below_off_threshold, bool):
            raise ContractValidationError("Field 'below_off_threshold' must be a boolean")

        physical = (self.signal_power_linear, self.noise_power_linear,
                    self.on_threshold_linear, self.off_threshold_linear)
        if any(x is not None for x in physical):
            if any(x is None for x in physical):
                raise ContractValidationError("Physical measurement fields must be supplied together")
            for name, value in zip(('signal_power_linear', 'noise_power_linear', 'on_threshold_linear', 'off_threshold_linear'), physical):
                _validate_finite_float(value, name)
            power, noise, on, off = physical
            if power < 0 or noise <= 0 or not 0 < off < on:
                raise ContractValidationError("Invalid physical powers/threshold ordering")
            if self.snr_valid != (power > noise):
                raise ContractValidationError('SNR validity contradicts physical excess')
            if self.snr_valid and not math.isclose(self.snr_estimate_db,
                    10*(math.log10(power-noise)-math.log10(noise)), rel_tol=0, abs_tol=1e-9):
                raise ContractValidationError('SNR contradicts physical excess/noise ratio')
            for linear, db in ((power, sig_pow), (noise, self.noise_power_dbfs), (on, on_th), (off, off_th)):
                # Zero power has a finite display-only sentinel. Its linear field
                # is authoritative; positive powers always have truthful logarithms.
                if linear > 0 and not math.isclose(10*math.log10(linear), db, rel_tol=0, abs_tol=1e-9):
                    raise ContractValidationError("Physical power and logarithmic metadata contradict")
            expected_above, expected_below = power >= on, power < off
        else:
            expected_above, expected_below = sig_pow >= on_th, sig_pow < off_th
        if self.above_on_threshold != expected_above:
            raise ContractValidationError(
                f"Inconsistent above_on_threshold: signal_power_dbfs ({sig_pow}) "
                f"{'>=' if expected_above else '<'} on_threshold_dbfs ({on_th}), "
                f"expected above_on_threshold={expected_above}, got {self.above_on_threshold}"
            )

        if self.below_off_threshold != expected_below:
            raise ContractValidationError(
                f"Inconsistent below_off_threshold: signal_power_dbfs ({sig_pow}) "
                f"{'<' if expected_below else '>='} off_threshold_dbfs ({off_th}), "
                f"expected below_off_threshold={expected_below}, got {self.below_off_threshold}"
            )


# ---------------------------------------------------------------------------
# Contract 5: SegmentationDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentationDecision:
    """Output decision of a segmentation state machine for a processing step.

    The state field represents the segmenter's state AFTER the current update.

    State / Transition Rules:
        - transition == NONE: state may be IDLE, ACTIVE, or POST_TRIGGER. boundary_sample is None.
        - transition == START: state MUST be ACTIVE. boundary_sample >= 0.
        - transition == END: state MUST be POST_TRIGGER. boundary_sample >= 0.

    Attributes:
        state: State of the segmenter after this update (IDLE, ACTIVE, POST_TRIGGER).
        transition: Boundary transition emitted during this step (NONE, START, END).
        block_sequence_number: Associated block sequence number (>= 0).
        boundary_sample: Absolute stream index of the transition boundary.
            Must be None when transition is NONE.
            Must be a valid sample index (>= 0) when transition is START or END.
    """

    state: SegmentState
    transition: SegmentTransition
    block_sequence_number: int
    boundary_sample: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, SegmentState):
            raise ContractValidationError(
                f"Field 'state' must be an instance of SegmentState, got {self.state!r}"
            )
        if not isinstance(self.transition, SegmentTransition):
            raise ContractValidationError(
                f"Field 'transition' must be an instance of SegmentTransition, got {self.transition!r}"
            )
        _validate_non_negative_int(self.block_sequence_number, "block_sequence_number")

        if self.transition == SegmentTransition.NONE:
            if self.boundary_sample is not None:
                raise ContractValidationError(
                    "Field 'boundary_sample' must be None when transition is SegmentTransition.NONE"
                )
        elif self.transition == SegmentTransition.START:
            if self.state != SegmentState.ACTIVE:
                raise ContractValidationError(
                    f"Segmentation transition START requires state ACTIVE, got {self.state.name}"
                )
            if self.boundary_sample is None:
                raise ContractValidationError(
                    "Field 'boundary_sample' must be provided when transition is START"
                )
            bs = _validate_non_negative_int(self.boundary_sample, "boundary_sample")
            object.__setattr__(self, "boundary_sample", bs)
        elif self.transition == SegmentTransition.END:
            if self.state != SegmentState.POST_TRIGGER:
                raise ContractValidationError(
                    f"Segmentation transition END requires state POST_TRIGGER, got {self.state.name}"
                )
            if self.boundary_sample is None:
                raise ContractValidationError(
                    "Field 'boundary_sample' must be provided when transition is END"
                )
            bs = _validate_non_negative_int(self.boundary_sample, "boundary_sample")
            object.__setattr__(self, "boundary_sample", bs)


# ---------------------------------------------------------------------------
# Contract 6: SignalEvent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalEvent:
    """A captured RF signal event extracted from the ring buffer.

    The event contains:
        [ PRE-TRIGGER MARGIN ][ DETECTED SIGNAL ][ POST-TRIGGER MARGIN ]

    Sample ordering invariant:
        0 <= event_start_sample
             <= detected_start_sample
             <= trigger_sample
             <  detected_end_sample
             <= event_end_sample

    And event_start_sample < event_end_sample.

    Interval semantics:
        - The captured event spans [event_start_sample, event_end_sample) in
          absolute stream coordinates.
        - The detected RF signal spans [detected_start_sample, detected_end_sample).
        - len(iq) == event_end_sample - event_start_sample.

    Boundary partition semantics:
        - Pre-signal samples: [event_start_sample, detected_start_sample)
        - Pre-trigger samples: [event_start_sample, trigger_sample)
        - Active signal samples: [detected_start_sample, detected_end_sample)
        - Post-signal samples: [detected_end_sample, event_end_sample)
        - Post-trigger samples: [trigger_sample, event_end_sample)
        - Non-overlapping partition:
          pre_signal_samples + signal_samples + post_signal_samples == num_samples.

    Attributes:
        event_id: Unique identifier for this extracted event (non-blank).
        stream: Stream metadata.
        iq: One-dimensional numpy array of np.complex64 samples encompassing
            pre-trigger, signal, and post-trigger regions. Stored read-only.
        event_start_sample: Absolute stream index where the captured event begins.
        event_end_sample: Absolute stream index where the captured event ends (exclusive).
        detected_start_sample: Absolute stream index where the signal was detected to begin.
        detected_end_sample: Absolute stream index where the signal was detected to end (exclusive).
        trigger_sample: Absolute stream index where the detector first triggered.
        timestamp_ns: Optional receiver epoch timestamp (nanoseconds) at event_start_sample.
        dropped_samples: Declared count of samples dropped during event capture (>= 0).
        complete: True if the entire event was captured without buffer overrun/loss.
            If dropped_samples > 0, complete MUST be False.

    Derived properties:
        num_samples: Total number of samples in the captured event.
        duration_s: Total duration in seconds.
        signal_start_offset: Local offset of detected signal start relative to event start.
        signal_end_offset: Local offset of detected signal end relative to event start.
        pre_signal_samples: Count of samples preceding detected signal start.
        pre_trigger_samples: Count of samples preceding detector trigger.
        signal_samples: Count of active signal samples (detected_end - detected_start).
        post_signal_samples: Count of samples following detected signal end.
        post_trigger_samples: Count of samples following detector trigger.
        pre_signal_duration_s: Duration of pre-signal margin in seconds.
        pre_trigger_duration_s: Duration of pre-trigger margin in seconds.
        signal_duration_s: Duration of detected signal in seconds.
        post_signal_duration_s: Duration of post-signal margin in seconds.
        post_trigger_duration_s: Duration of post-trigger margin in seconds.
        signal_iq: Read-only slice containing exclusively the detected signal samples.
    """

    event_id: str
    stream: StreamInfo
    iq: np.ndarray
    event_start_sample: int
    event_end_sample: int
    detected_start_sample: int
    detected_end_sample: int
    trigger_sample: int
    timestamp_ns: int | None = None
    dropped_samples: int = 0
    complete: bool = True

    def __post_init__(self) -> None:
        _validate_non_blank_str(self.event_id, "event_id")
        if not isinstance(self.stream, StreamInfo):
            raise ContractValidationError(
                f"Field 'stream' must be an instance of StreamInfo, got {type(self.stream)}"
            )

        # Validate integer types strictly (no bools)
        for name, val in [
            ("event_start_sample", self.event_start_sample),
            ("event_end_sample", self.event_end_sample),
            ("detected_start_sample", self.detected_start_sample),
            ("detected_end_sample", self.detected_end_sample),
            ("trigger_sample", self.trigger_sample),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
                raise ContractValidationError(f"Field '{name}' must be an integer, got {val!r}")

        ev_start = int(self.event_start_sample)
        det_start = int(self.detected_start_sample)
        trig = int(self.trigger_sample)
        det_end = int(self.detected_end_sample)
        ev_end = int(self.event_end_sample)

        if ev_start < 0:
            raise ContractValidationError(
                f"Field 'event_start_sample' must be >= 0, got {ev_start}"
            )

        if ev_start >= ev_end:
            raise ContractValidationError(
                f"Event interval must be strictly positive: require event_start ({ev_start}) < event_end ({ev_end})"
            )

        if det_start >= det_end:
            raise ContractValidationError(
                f"Detected signal interval must be non-empty: require detected_start ({det_start}) < detected_end ({det_end})"
            )

        # Enforce temporal ordering invariant: event_start <= detected_start <= trigger < detected_end <= event_end
        if not (ev_start <= det_start <= trig < det_end <= ev_end):
            raise ContractValidationError(
                f"Violated sample index ordering invariant: require "
                f"event_start ({ev_start}) <= detected_start ({det_start}) <= "
                f"trigger ({trig}) < detected_end ({det_end}) <= event_end ({ev_end})"
            )

        _validate_non_negative_int(self.dropped_samples, "dropped_samples")

        if self.timestamp_ns is not None:
            if isinstance(self.timestamp_ns, bool) or not isinstance(
                self.timestamp_ns, (int, np.integer)
            ):
                raise ContractValidationError(
                    f"Field 'timestamp_ns' must be an integer nanoseconds or None, got {self.timestamp_ns!r}"
                )
            object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))

        if not isinstance(self.complete, bool):
            raise ContractValidationError(
                f"Field 'complete' must be a boolean, got {self.complete!r}"
            )

        if self.dropped_samples > 0 and self.complete:
            raise ContractValidationError(
                f"Event with dropped_samples={self.dropped_samples} (> 0) cannot have complete=True"
            )

        frozen_iq = _validate_and_freeze_iq_array(self.iq, "iq")
        object.__setattr__(self, "iq", frozen_iq)

        expected_samples = ev_end - ev_start
        if len(frozen_iq) != expected_samples:
            raise ContractValidationError(
                f"IQ array length ({len(frozen_iq)}) does not match declared event interval "
                f"[event_start_sample={ev_start}, event_end_sample={ev_end}) "
                f"which requires {expected_samples} samples"
            )

    @property
    def num_samples(self) -> int:
        """Total number of samples across the entire event buffer."""
        return len(self.iq)

    @property
    def duration_s(self) -> float:
        """Total event duration in seconds."""
        return len(self.iq) / self.stream.sample_rate_hz

    @property
    def signal_start_offset(self) -> int:
        """Sample index offset of the detected signal relative to event_start_sample."""
        return self.detected_start_sample - self.event_start_sample

    @property
    def signal_end_offset(self) -> int:
        """Sample index offset of the detected signal end relative to event_start_sample."""
        return self.detected_end_sample - self.event_start_sample

    @property
    def pre_signal_samples(self) -> int:
        """Number of pre-signal buffer samples preceding detected signal start: detected_start - event_start."""
        return self.detected_start_sample - self.event_start_sample

    @property
    def pre_trigger_samples(self) -> int:
        """Number of pre-trigger buffer samples preceding detector trigger: trigger - event_start."""
        return self.trigger_sample - self.event_start_sample

    @property
    def signal_samples(self) -> int:
        """Number of active detected signal samples: detected_end - detected_start."""
        return self.detected_end_sample - self.detected_start_sample

    @property
    def post_signal_samples(self) -> int:
        """Number of post-signal buffer samples following detected signal end: event_end - detected_end."""
        return self.event_end_sample - self.detected_end_sample

    @property
    def post_trigger_samples(self) -> int:
        """Number of post-trigger buffer samples following detector trigger: event_end - trigger."""
        return self.event_end_sample - self.trigger_sample

    @property
    def pre_signal_duration_s(self) -> float:
        """Duration of pre-signal margin in seconds."""
        return self.pre_signal_samples / self.stream.sample_rate_hz

    @property
    def pre_trigger_duration_s(self) -> float:
        """Duration of pre-trigger margin in seconds."""
        return self.pre_trigger_samples / self.stream.sample_rate_hz

    @property
    def signal_duration_s(self) -> float:
        """Duration of detected signal in seconds."""
        return self.signal_samples / self.stream.sample_rate_hz

    @property
    def post_signal_duration_s(self) -> float:
        """Duration of post-signal margin in seconds."""
        return self.post_signal_samples / self.stream.sample_rate_hz

    @property
    def post_trigger_duration_s(self) -> float:
        """Duration of post-trigger margin in seconds."""
        return self.post_trigger_samples / self.stream.sample_rate_hz

    @property
    def signal_iq(self) -> np.ndarray:
        """Read-only view of the complex IQ samples belonging strictly to the detected signal."""
        return self.iq[self.signal_start_offset : self.signal_end_offset]


# ---------------------------------------------------------------------------
# Transform Result Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PSDResult:
    """Power Spectral Density (PSD) transform result container.

    Attributes:
        frequency_offset_hz: Real frequency bin coordinates relative to center frequency (Hz).
        power_linear: Non-negative real linear power spectral density values (>= 0).
        power_dbfs_per_hz: Logarithmic real power spectral density in dBFS/Hz.

    Requirements:
        All arrays must be 1-D, equal length, non-empty, real-valued numeric, finite, and read-only.
        power_linear must contain non-negative values (>= 0).
    """

    frequency_offset_hz: np.ndarray
    power_linear: np.ndarray
    power_dbfs_per_hz: np.ndarray

    def __post_init__(self) -> None:
        f_hz = _validate_and_freeze_real_array(self.frequency_offset_hz, "frequency_offset_hz")
        p_lin = _validate_and_freeze_real_array(self.power_linear, "power_linear")
        p_dbfs = _validate_and_freeze_real_array(self.power_dbfs_per_hz, "power_dbfs_per_hz")

        n = len(f_hz)
        if len(p_lin) != n or len(p_dbfs) != n:
            raise ContractValidationError(
                f"PSDResult array lengths must match: frequency_offset_hz={n}, "
                f"power_linear={len(p_lin)}, power_dbfs_per_hz={len(p_dbfs)}"
            )

        if np.any(p_lin < 0.0):
            raise ContractValidationError(
                "Field 'power_linear' in PSDResult must contain non-negative values (>= 0)"
            )

        object.__setattr__(self, "frequency_offset_hz", f_hz)
        object.__setattr__(self, "power_linear", p_lin)
        object.__setattr__(self, "power_dbfs_per_hz", p_dbfs)


@dataclass(frozen=True)
class PhaseResult:
    """Instantaneous phase calculation result container.

    Attributes:
        wrapped_rad: Real wrapped instantaneous phase in radians (-pi to +pi).
        unwrapped_rad: Real continuous unwrapped phase in radians.
        valid_mask: Optional boolean array marking samples with valid phase.
            Defaults to an all-True mask if None.

    Requirements:
        Arrays must be 1-D, equal length, real-valued numeric/boolean, finite, and read-only.
    """

    wrapped_rad: np.ndarray
    unwrapped_rad: np.ndarray
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        w_rad = _validate_and_freeze_real_array(self.wrapped_rad, "wrapped_rad")
        u_rad = _validate_and_freeze_real_array(self.unwrapped_rad, "unwrapped_rad")

        if len(w_rad) != len(u_rad):
            raise ContractValidationError(
                f"PhaseResult array lengths must match: wrapped_rad={len(w_rad)}, "
                f"unwrapped_rad={len(u_rad)}"
            )

        if self.valid_mask is None:
            v_mask = _make_array_immutable(np.ones(len(w_rad), dtype=bool))
        else:
            try:
                vm = np.asarray(self.valid_mask)
            except Exception as exc:
                raise ContractValidationError(
                    f"Field 'valid_mask' cannot be converted to a numpy array: {exc}"
                ) from exc
            if vm.ndim != 1 or len(vm) != len(w_rad):
                raise ContractValidationError(
                    f"Field 'valid_mask' must be 1-dimensional with length {len(w_rad)}, got shape {vm.shape}"
                )
            if vm.dtype != bool:
                if not (np.issubdtype(vm.dtype, np.integer) or np.issubdtype(vm.dtype, np.floating)):
                    raise ContractValidationError(
                        f"Field 'valid_mask' must have boolean or numeric dtype, got {vm.dtype}"
                    )
                vm = vm.astype(bool)
            v_mask = _make_array_immutable(vm)

        object.__setattr__(self, "wrapped_rad", w_rad)
        object.__setattr__(self, "unwrapped_rad", u_rad)
        object.__setattr__(self, "valid_mask", v_mask)


@dataclass(frozen=True)
class AutocorrelationResult:
    """Autocorrelation calculation result container.

    Attributes:
        lag_samples: Discrete lag indices in sample units (integer array).
        values: Autocorrelation values corresponding to each lag (real or complex).

    Requirements:
        Arrays must be 1-D, non-empty, equal length, finite, and read-only.
        lag_samples must be integer sample offsets without fractional parts.
        values may be real or complex numeric.
    """

    lag_samples: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        lags = _validate_and_freeze_integer_lag_array(self.lag_samples, "lag_samples")
        vals = _validate_and_freeze_numeric_1d_array(self.values, "values")

        if len(lags) != len(vals):
            raise ContractValidationError(
                f"AutocorrelationResult array lengths must match: lag_samples={len(lags)}, "
                f"values={len(vals)}"
            )

        object.__setattr__(self, "lag_samples", lags)
        object.__setattr__(self, "values", vals)


# ---------------------------------------------------------------------------
# Feature Contracts
# ---------------------------------------------------------------------------

class FrozenDict(dict):
    """JSON/asdict-compatible immutable mapping (including in-place union)."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("Mapping is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

@dataclass(frozen=True)
class FeatureValue:
    """A single extracted scalar feature value with validity metadata.

    Attributes:
        value: Numeric feature value (finite float if valid=True, float/NaN if valid=False).
        unit: Optional physical unit string (e.g. 'Hz', 'dBFS', 's', 'rad').
        valid: Whether this feature calculation succeeded.
        reason: Explanatory non-blank error string when valid=False. Must be None when valid=True.
    """

    value: float
    unit: str | None = None
    valid: bool = True
    reason: str | None = None
    interpretation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ContractValidationError("Field 'valid' must be a boolean")

        if isinstance(self.value, bool) or not isinstance(
            self.value, (int, float, np.floating, np.integer)
        ):
            raise ContractValidationError(
                f"Field 'value' must be a numeric float, got {self.value!r}"
            )
        fval = float(self.value)
        object.__setattr__(self, "value", fval)

        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit.strip()):
            raise ContractValidationError(
                "Field 'unit' must be a non-blank string or None"
            )

        if self.valid:
            if not math.isfinite(fval):
                raise ContractValidationError(
                    f"Field 'value' must be finite when valid=True, got {fval}"
                )
            if self.reason is not None:
                raise ContractValidationError(
                    "Field 'reason' must be None when valid=True"
                )
        else:
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ContractValidationError(
                    "Field 'reason' must be a non-blank string explaining why feature calculation failed when valid=False"
                )


@dataclass(frozen=True)
class FeatureSet:
    """A collection of features produced by a single feature extractor.

    Attributes:
        extractor: Extractor domain identifier (e.g. 'spectral', 'power', non-blank).
        extractor_version: Version identifier string of the extractor (non-blank).
        features: Immutable mapping of feature names to FeatureValue instances.
    """

    extractor: str
    extractor_version: str
    features: Mapping[str, FeatureValue]

    def __post_init__(self) -> None:
        _validate_non_blank_str(self.extractor, "extractor")
        _validate_non_blank_str(self.extractor_version, "extractor_version")

        if not isinstance(self.features, Mapping):
            raise ContractValidationError(
                f"Field 'features' must be a mapping, got {type(self.features)}"
            )
        if len(self.features) == 0:
            raise ContractValidationError("Field 'features' must not be empty")

        defensive: dict[str, FeatureValue] = {}
        for k, v in self.features.items():
            if not isinstance(k, str) or not k.strip():
                raise ContractValidationError(
                    f"Feature key must be a non-blank string, got {k!r}"
                )
            if not isinstance(v, FeatureValue):
                raise ContractValidationError(
                    f"Feature value for key '{k}' must be an instance of FeatureValue, got {type(v)}"
                )
            defensive[k] = v

        object.__setattr__(self, "features", MappingProxyType(defensive))


@dataclass(frozen=True)
class FeatureVector:
    """Consolidated, namespaced feature vector associated with an extracted event.

    Attributes:
        event_id: Identifier of the associated SignalEvent (non-blank).
        feature_schema_version: Schema version for downstream consumers (non-blank).
        features: Immutable mapping of namespaced feature names to FeatureValue objects.
    """

    event_id: str
    feature_schema_version: str
    features: Mapping[str, FeatureValue]

    def __post_init__(self) -> None:
        _validate_non_blank_str(self.event_id, "event_id")
        _validate_non_blank_str(self.feature_schema_version, "feature_schema_version")

        if not isinstance(self.features, Mapping):
            raise ContractValidationError(
                f"Field 'features' must be a mapping, got {type(self.features)}"
            )

        defensive: dict[str, FeatureValue] = {}
        for k, v in self.features.items():
            if not isinstance(k, str) or not k.strip():
                raise ContractValidationError(
                    f"Feature key must be a non-blank string, got {k!r}"
                )
            if not isinstance(v, FeatureValue):
                raise ContractValidationError(
                    f"Feature value for key '{k}' must be an instance of FeatureValue, got {type(v)}"
                )
            defensive[k] = v

        object.__setattr__(self, "features", MappingProxyType(defensive))

    @classmethod
    def from_feature_sets(
        cls,
        event_id: str,
        feature_schema_version: str,
        feature_sets: Iterable[FeatureSet],
    ) -> FeatureVector:
        """Assemble a unified FeatureVector from multiple FeatureSet instances.

        Enforces that duplicate feature names are never silently overwritten.

        Args:
            event_id: Identifier of the event.
            feature_schema_version: Schema version.
            feature_sets: Iterable of FeatureSet objects to combine.

        Returns:
            Assembled immutable FeatureVector.

        Raises:
            ContractValidationError: If duplicate feature keys are detected.
        """
        merged: dict[str, FeatureValue] = {}
        for fset in feature_sets:
            if not isinstance(fset, FeatureSet):
                raise ContractValidationError(
                    f"Expected FeatureSet instance in feature_sets, got {type(fset)}"
                )
            for k, v in fset.features.items():
                if k in merged:
                    raise ContractValidationError(
                        f"Duplicate feature name '{k}' encountered during FeatureVector assembly "
                        f"from extractor '{fset.extractor}'"
                    )
                merged[k] = v

        return cls(
            event_id=event_id,
            feature_schema_version=feature_schema_version,
            features=merged,
        )

    def to_array(
        self,
        ordered_names: Iterable[str] | None = None,
        *,
        invalid_value: float = float("nan"),
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Export feature values and boolean validity mask to deterministic 1-D numpy arrays.

        Args:
            ordered_names: Explicit feature name sequence. If None, uses keys in self.features.
            invalid_value: Floating-point value for invalid features (default: NaN).

        Returns:
            Tuple of (values_array, validity_mask_array, ordered_names_tuple).
            Both arrays are frozen (immutable).
        """
        names = tuple(ordered_names) if ordered_names is not None else tuple(self.features.keys())
        values = np.empty(len(names), dtype=np.float64)
        validity = np.empty(len(names), dtype=bool)

        for i, name in enumerate(names):
            if name not in self.features:
                raise ContractValidationError(f"Feature '{name}' not found in FeatureVector")
            fv = self.features[name]
            if fv.valid:
                values[i] = float(fv.value)
                validity[i] = True
            else:
                values[i] = float(invalid_value)
                validity[i] = False

        frozen_values = _make_array_immutable(values)
        frozen_validity = _make_array_immutable(validity)
        return frozen_values, frozen_validity, names


# ---------------------------------------------------------------------------
# Capture Quality and Extraction Policy Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaptureQuality:
    """Capture completeness, continuity, and quality metadata.

    Attributes:
        complete: Whether the event capture buffer is complete with no dropped samples.
        dropped_samples: Total dropped sample count recorded during capture (>= 0).
        is_continuous: Whether the signal buffer represents a continuous time series.
        warnings: Immutable tuple of human-readable quality warning strings.
    """

    complete: bool
    dropped_samples: int
    is_continuous: bool
    warnings: tuple[str, ...] = ()
    known_lost_samples: int = 0
    unknown_loss: bool = False
    termination_reason: str | None = None
    capture_epoch: int | None = None
    noise_valid: bool | None = None
    noise_provenance: str | None = None
    boundary_uncertainty_samples: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.complete, bool):
            raise ContractValidationError(f"complete must be a boolean, got {self.complete!r}")
        _validate_non_negative_int(self.dropped_samples, "dropped_samples")
        _validate_non_negative_int(self.known_lost_samples, "known_lost_samples")
        for name in ("capture_epoch", "boundary_uncertainty_samples"):
            if getattr(self, name) is not None:
                _validate_non_negative_int(getattr(self, name), name)
        if type(self.unknown_loss) is not bool or (self.noise_valid is not None and type(self.noise_valid) is not bool):
            raise ContractValidationError("Loss and calibration trust flags must be boolean")
        if not isinstance(self.is_continuous, bool):
            raise ContractValidationError(f"is_continuous must be a boolean, got {self.is_continuous!r}")
        if self.dropped_samples > 0 and self.complete:
            raise ContractValidationError(
                f"Event with dropped_samples={self.dropped_samples} cannot have complete=True"
            )
        if isinstance(self.warnings, (list, tuple)):
            clean_warn = tuple(str(w) for w in self.warnings)
        else:
            clean_warn = (str(self.warnings),)
        object.__setattr__(self, "warnings", clean_warn)


@dataclass(frozen=True)
class ExtractionPolicy:
    """Immutable record of resolved extraction configuration and feature schema.

    Preserves exact extraction settings to guarantee reproducible downstream modeling.
    """

    policy_id: str
    schema_version: str
    sample_rate_hz: float
    observation_samples: int
    psd_mode: str
    nfft: int
    spectral_window: str
    psd_nperseg: int | None
    psd_noverlap: int | None
    autocorrelation_max_lag_samples: int | None
    autocorrelation_normalized: bool
    phase_amplitude_threshold: float
    bandwidth_roi_hz: tuple[float, float] | None
    bandwidth_noise_floor_dbfs_per_hz: float | None
    feature_names: tuple[str, ...]
    centered_spectrum: bool = True
    bandwidth_mode: str = "raw_psd"
    bandwidth_interpolate_edges: bool = False
    bandwidth_dominant_lobe_min_prominence_db: float = 3.0
    power_floor_dbfs: float = -300.0
    psd_floor_dbfs_per_hz: float = -300.0
    bandwidth_min_snr_db: float | None = None
    amplitude_ddof: int = 1
    check_edge_clipping: bool = True
    extractor_versions: dict[str, str] = field(default_factory=dict)
    bandwidth_auto_noise_estimation: bool = True
    bandwidth_multimodal_ratio_threshold: float = 0.7
    bandwidth_occupied_power_scope: str = "observed_excess"
    bandwidth_complete_observation: bool = False
    bandwidth_support_provenance: str | None = None

    def __post_init__(self) -> None:
        _validate_non_blank_str(self.policy_id, "policy_id")
        _validate_non_blank_str(self.schema_version, "schema_version")
        _validate_finite_float(self.sample_rate_hz, "sample_rate_hz")
        _validate_non_negative_int(self.observation_samples, "observation_samples")
        _validate_non_blank_str(self.psd_mode, "psd_mode")
        _validate_non_negative_int(self.nfft, "nfft")
        _validate_non_blank_str(self.spectral_window, "spectral_window")
        if not isinstance(self.feature_names, tuple):
            object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "extractor_versions", FrozenDict(self.extractor_versions))
        if self.bandwidth_roi_hz is not None:
            object.__setattr__(self, "bandwidth_roi_hz", tuple(self.bandwidth_roi_hz))

    def to_dict(self) -> dict:
        """Canonical JSON-ready policy with no mutable aliases to this record."""
        # The schema contains immutable scalars, two scalar tuples and one
        # string-to-string mapping. Deepcopy + JSON encode/decode adds no
        # isolation here; allocate only the actual mutable containers required
        # by the public JSON-ready result. Keep field enumeration schema-driven.
        from dataclasses import fields
        result = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, Mapping):
                value = dict(value)
            result[item.name] = value
        return result

    @classmethod
    def from_dict(cls, record: Mapping) -> ExtractionPolicy:
        import hashlib
        import json
        values = dict(record)
        policy_id = values.pop("policy_id")
        values["feature_names"] = sorted(values["feature_names"])
        digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]
        if policy_id != f"policy_{values['schema_version']}_{digest}":
            raise ContractValidationError("Policy content does not match its identity")
        return cls(policy_id=policy_id, **values)


@dataclass(frozen=True)
class AnalysisResult:
    """Mandatory analysis-result envelope holding feature vector, policy, and capture quality."""

    feature_vector: FeatureVector
    policy: ExtractionPolicy
    capture_quality: CaptureQuality

    def __post_init__(self) -> None:
        if not isinstance(self.feature_vector, FeatureVector):
            raise ContractValidationError(
                f"feature_vector must be an instance of FeatureVector, got {type(self.feature_vector).__name__}"
            )
        if not isinstance(self.policy, ExtractionPolicy):
            raise ContractValidationError(
                f"policy must be an instance of ExtractionPolicy, got {type(self.policy).__name__}"
            )
        if not isinstance(self.capture_quality, CaptureQuality):
            raise ContractValidationError(
                f"capture_quality must be an instance of CaptureQuality, got {type(self.capture_quality).__name__}"
            )
