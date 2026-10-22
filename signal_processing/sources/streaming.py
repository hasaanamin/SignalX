"""Positioned source envelopes and an explicit, bounded process wire format.

IQBlock remains the DSP input. This additive envelope carries acquisition facts
which must not be inferred from host timestamps or worker cancellation clocks.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from functools import lru_cache
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from signal_processing.core import IQBlock, StreamInfo


class SourceError(RuntimeError):
    """Source operation failed; this is not finite EOF."""


class SourceReadTimeout(TimeoutError):
    """No block available within the read deadline. Retrying is permitted."""


class SourceCancelled(SourceError):
    """Read was cancelled. Restart requires an explicit new run/reset."""


class SourceDisconnected(SourceError):
    """Device/transport disconnected; do not interpret this as EOF."""


class SourceBusy(SourceError):
    """An operation conflicts with an active reader."""


class SourceClosed(SourceError):
    """The source has been closed."""


def nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative Python integer")
    return value


def read_timeout(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value < 0):
        raise ValueError("timeout must be a finite nonnegative number or None")
    return value


@dataclass(frozen=True)
class SourceMetadata:
    """Acquisition provenance, independent of capture/worker generations.

    source_epoch changes on source restart; configuration_epoch changes on a
    deliberate reconfiguration. Neither certifies RF continuity. recording time
    is preserved recorded time, receiver time is a device-supplied anchor;
    unknown means IQBlock.timestamp_ns must be None. adc_scale describes the
    conversion already applied to IQ, not an instruction to scale it again.
    """

    source_epoch: int = 0
    configuration_epoch: int = 0
    timestamp_origin: str = "unknown"
    gain_mode: str = "unknown"
    gain_db: float | None = None
    adc_scale: float = 1.0

    def __post_init__(self):
        nonnegative_int(self.source_epoch, "source_epoch")
        nonnegative_int(self.configuration_epoch, "configuration_epoch")
        if self.timestamp_origin not in ("unknown", "recording", "receiver"):
            raise ValueError("timestamp_origin must be unknown, recording or receiver")
        if self.gain_mode not in ("unknown", "manual", "slow_attack", "fast_attack"):
            raise ValueError("Unsupported gain mode")
        if self.gain_db is not None:
            if isinstance(self.gain_db, bool) or not isinstance(self.gain_db, (int, float)) or not math.isfinite(self.gain_db):
                raise ValueError("gain_db must be finite or None")
            object.__setattr__(self, "gain_db", float(self.gain_db))
        if isinstance(self.adc_scale, bool) or not isinstance(self.adc_scale, (int, float)) or not math.isfinite(self.adc_scale) or self.adc_scale <= 0:
            raise ValueError("adc_scale must be finite and positive")
        object.__setattr__(self, "adc_scale", float(self.adc_scale))


@dataclass(frozen=True)
class SourceEnvelope:
    block: IQBlock
    metadata: SourceMetadata = SourceMetadata()
    received_monotonic_ns: int = 0
    scheduled_monotonic_ns: int | None = None

    def __post_init__(self):
        if not isinstance(self.block, IQBlock) or not isinstance(self.metadata, SourceMetadata):
            raise TypeError("SourceEnvelope requires IQBlock and SourceMetadata")
        nonnegative_int(self.received_monotonic_ns, "received_monotonic_ns")
        if self.scheduled_monotonic_ns is not None:
            nonnegative_int(self.scheduled_monotonic_ns, "scheduled_monotonic_ns")
        if self.metadata.timestamp_origin == "unknown" and self.block.timestamp_ns is not None:
            raise ValueError("An IQ timestamp requires an explicit recording/receiver origin")

    @property
    def configuration_id(self):
        return _configuration_id(self.block.stream,self.metadata)


@lru_cache(maxsize=32)
def _configuration_id(stream,metadata):
    # Only frozen configuration is retained, never an envelope/IQ/clock/result.
    record = {"stream": dataclasses.asdict(stream), "metadata": dataclasses.asdict(metadata)}
    record["metadata"].pop("source_epoch")
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "source_v1_" + hashlib.sha256(encoded.encode()).hexdigest()[:20]


@runtime_checkable
class PositionedIQSource(Protocol):
    """Single-reader, cancellable source interface for replay/future hardware.

    read_envelope returns None ONLY on finite EOF. Timeouts are retryable;
    disconnect/cancellation raise their typed outcomes. cancel() must interrupt
    a pending read within the implementation's documented bounded interval.
    A hardware restart must create a new source epoch; reset is intentionally
    not part of this protocol because a device cannot rewind received RF.
    """

    @property
    def stream_info(self) -> StreamInfo: ...

    def read_envelope(self, timeout: float | None = None) -> SourceEnvelope | None: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


_MAGIC = b"SXI1"
_HEADER_LIMIT = 8192


def encode_envelope(envelope: SourceEnvelope, *, max_samples: int) -> bytes:
    """Encode explicit metadata and little-endian complex64, without pickle."""
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError("max_samples must be positive")
    if not isinstance(envelope, SourceEnvelope):
        raise TypeError("Expected SourceEnvelope")
    b = envelope.block
    if b.num_samples > max_samples:
        raise ValueError("Envelope exceeds transport sample capacity")
    record = {"stream": dataclasses.asdict(b.stream),
              "block": {k: int(getattr(b, k)) for k in ("sequence_number", "start_sample", "dropped_samples_before")},
              "timestamp_ns": b.timestamp_ns, "overflow": b.overflow,
              "metadata": dataclasses.asdict(envelope.metadata),
              "received_monotonic_ns": envelope.received_monotonic_ns,
              "scheduled_monotonic_ns": envelope.scheduled_monotonic_ns}
    header = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    if len(header) > _HEADER_LIMIT:
        raise ValueError("Envelope metadata exceeds transport capacity")
    return _MAGIC + struct.pack("<I", len(header)) + header + b.iq.astype("<c8", copy=False).tobytes()


def decode_envelope(payload: bytes, *, max_samples: int) -> SourceEnvelope:
    """Validate bounded wire data and reconstruct independently immutable IQ."""
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError("max_samples must be positive")
    if not isinstance(payload, bytes) or not 8 < len(payload) <= 8 + _HEADER_LIMIT + 8 * max_samples:
        raise ValueError("Invalid envelope byte count")
    if payload[:4] != _MAGIC:
        raise ValueError("Unsupported envelope protocol")
    n = struct.unpack("<I", payload[4:8])[0]
    remaining = len(payload) - 8 - n
    if not 0 < n <= _HEADER_LIMIT or not 0 < remaining <= 8 * max_samples or remaining % 8:
        raise ValueError("Invalid envelope lengths")
    record = json.loads(payload[8:8+n].decode("utf8"))
    expected = {"stream", "block", "timestamp_ns", "overflow", "metadata",
                "received_monotonic_ns", "scheduled_monotonic_ns"}
    if not isinstance(record, dict) or record.keys() != expected:
        raise ValueError("Invalid envelope fields")
    b = record["block"]
    if not isinstance(b, dict) or b.keys() != {"sequence_number", "start_sample", "dropped_samples_before"}:
        raise ValueError("Invalid positioned block fields")
    for key, value in b.items():
        nonnegative_int(value, key)
    block = IQBlock(StreamInfo(**record["stream"]), **b,
                    timestamp_ns=record["timestamp_ns"], overflow=record["overflow"],
                    iq=np.frombuffer(payload, dtype="<c8", offset=8+n).astype(np.complex64, copy=False))
    return SourceEnvelope(block, SourceMetadata(**record["metadata"]),
                          record["received_monotonic_ns"], record["scheduled_monotonic_ns"])
