from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

import copy
import socket
import struct
from dataclasses import dataclass
from typing import Any
import numpy as np

from .Acquisition import parse_iq


# =============================================================================
# UDP IQ STREAM PROTOCOL CONSTANTS
# =============================================================================

UDP_IQ_HEADER_FMT_LEGACY = "!IHI"
UDP_IQ_HEADER_SIZE_LEGACY = struct.calcsize(UDP_IQ_HEADER_FMT_LEGACY)  # Exactly 10 bytes

UDP_IQ_HEADER_FMT_TIMESTAMPED = "!IHIQ"
UDP_IQ_HEADER_SIZE_TIMESTAMPED = struct.calcsize(UDP_IQ_HEADER_FMT_TIMESTAMPED)  # Exactly 18 bytes

# Backward-compatibility aliases for legacy 10-byte header
UDP_IQ_HEADER_FMT = UDP_IQ_HEADER_FMT_LEGACY
UDP_IQ_HEADER_SIZE = UDP_IQ_HEADER_SIZE_LEGACY

UDP_MAX_DATAGRAM_BYTES = 65507                          # Standard IPv4 UDP payload limit
MAX_IQ_COUNT = 1_000_000                                # Upper sanity limit on sample count per packet
BYTES_PER_IQ_SAMPLE = 4                                 # 2 x float16 (2 bytes I + 2 bytes Q)


@dataclass(frozen=True)
class UDPIQPacket:
    """
    Decoded representation of an individual UDP IQ transport datagram.

    Attributes
    ----------
    signal_id : int
        Unique 32-bit emitter/signal stream identifier.
    channel_id : int
        16-bit receiver channel index (strictly >= 1).
    iq_count : int
        Number of complex IQ samples declared in the header.
    complex_samples : np.ndarray
        1D complex128 array containing the decoded IQ samples.
    i : np.ndarray
        1D float64 array of in-phase components.
    q : np.ndarray
        1D float64 array of quadrature components.
    payload_bytes : int
        Exact byte length of the float16 IQ payload.
    packet_bytes : int
        Total length in bytes of the received UDP datagram.
    receive_timestamp_ns : int | None
        Monotonic or system reception timestamp in nanoseconds when the packet was received on the host.
    source_address : tuple[str, int] | None
        Network (IP, port) tuple of the transmitting sender.
    hardware_timestamp_ns : int | None
        Genuine hardware capture timestamp in nanoseconds for sample 0 in this UDP packet,
        or None if transmitted using the legacy 10-byte header.
    """
    signal_id: int
    channel_id: int
    iq_count: int
    complex_samples: np.ndarray
    i: np.ndarray
    q: np.ndarray
    payload_bytes: int
    packet_bytes: int
    receive_timestamp_ns: int | None = None
    source_address: tuple[str, int] | None = None
    hardware_timestamp_ns: int | None = None


def parse_udp_iq_packet(
    data: bytes | bytearray | memoryview,
    *,
    receive_timestamp_ns: int | None = None,
    source_address: tuple[str, int] | None = None,
    sample_rate: float | None = None,
) -> UDPIQPacket:
    """
    Pure parsing and validation function for a single UDP IQ datagram.

    Supports both legacy (!IHI, 10 bytes) and timestamped (!IHIQ, 18 bytes) packet headers.
    Format detection is determined strictly by total datagram length and declared iq_count.

    Legacy Header format: '!IHI' (10 bytes)
        - signal_id  : uint32 (4 bytes, network big-endian)
        - channel_id : uint16 (2 bytes, network big-endian)
        - iq_count   : uint32 (4 bytes, network big-endian)
        - hardware_timestamp_ns = None

    Timestamped Header format: '!IHIQ' (18 bytes)
        - signal_id  : uint32 (4 bytes, network big-endian)
        - channel_id : uint16 (2 bytes, network big-endian)
        - iq_count   : uint32 (4 bytes, network big-endian)
        - hardware_timestamp_ns : uint64 (8 bytes, network big-endian)

    Payload format (identical for both):
        - Interleaved float16 IQ pairs [I0, Q0, I1, Q1, ...] (4 bytes per complex sample).

    Parameters
    ----------
    data : bytes | bytearray | memoryview
        Raw UDP packet bytes.
    receive_timestamp_ns : int | None
        Optional local host reception timestamp in nanoseconds.
    source_address : tuple[str, int] | None
        Optional (source_ip, source_port) tuple.
    sample_rate : float | None
        Optional declared sample rate for validation.

    Returns
    -------
    UDPIQPacket
        Structured, fully validated packet object.

    Raises
    ------
    TypeError
        If input data is not bytes-like.
    ValueError
        If packet length < 10, channel_id == 0, iq_count == 0, iq_count > MAX_IQ_COUNT,
        packet length does not match either legacy or timestamped layout, packet exceeds
        65507 bytes, or decoded IQ contains NaN/infinite values.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like, got {type(data).__name__}")

    raw_len = len(data)
    if raw_len > UDP_MAX_DATAGRAM_BYTES:
        raise ValueError(f"Packet length ({raw_len} bytes) exceeds UDP maximum {UDP_MAX_DATAGRAM_BYTES} bytes")

    if raw_len < UDP_IQ_HEADER_SIZE_LEGACY:
        raise ValueError(
            f"Packet length ({raw_len} bytes) is shorter than minimum 10-byte header ({UDP_IQ_HEADER_SIZE_LEGACY} bytes)"
        )

    buf = memoryview(data)
    # Unpack initial 10 bytes common prefix: signal_id:uint32, channel_id:uint16, iq_count:uint32
    try:
        signal_id, channel_id, iq_count = struct.unpack(UDP_IQ_HEADER_FMT_LEGACY, buf[:UDP_IQ_HEADER_SIZE_LEGACY])
    except struct.error as exc:
        raise ValueError(f"Failed to unpack UDP IQ header: {exc}") from exc

    if channel_id == 0:
        raise ValueError("Invalid channel_id 0; channel indices must be non-zero positive integers")

    if iq_count == 0:
        raise ValueError("Invalid iq_count 0; packet must contain at least 1 IQ sample")

    if iq_count > MAX_IQ_COUNT:
        raise ValueError(f"iq_count {iq_count} exceeds maximum sanity limit of {MAX_IQ_COUNT} samples")

    expected_payload_bytes = iq_count * BYTES_PER_IQ_SAMPLE
    expected_legacy_total = UDP_IQ_HEADER_SIZE_LEGACY + expected_payload_bytes
    expected_timestamped_total = UDP_IQ_HEADER_SIZE_TIMESTAMPED + expected_payload_bytes

    if raw_len == expected_legacy_total:
        hardware_timestamp_ns = None
        header_size = UDP_IQ_HEADER_SIZE_LEGACY
    elif raw_len == expected_timestamped_total:
        try:
            _, _, _, hw_ts = struct.unpack(UDP_IQ_HEADER_FMT_TIMESTAMPED, buf[:UDP_IQ_HEADER_SIZE_TIMESTAMPED])
            hardware_timestamp_ns = int(hw_ts)
        except struct.error as exc:
            raise ValueError(f"Failed to unpack timestamped UDP IQ header: {exc}") from exc
        header_size = UDP_IQ_HEADER_SIZE_TIMESTAMPED
    else:
        if raw_len < expected_legacy_total:
            raise ValueError(
                f"Truncated UDP IQ packet: datagram has {raw_len} bytes, expected at least {expected_legacy_total} bytes (legacy) or "
                f"{expected_timestamped_total} bytes (timestamped) for iq_count={iq_count}"
            )
        raise ValueError(
            f"Trailing / unaligned bytes in UDP IQ packet: datagram has {raw_len} bytes, expected {expected_legacy_total} (legacy) or "
            f"{expected_timestamped_total} (timestamped) for iq_count={iq_count}"
        )

    payload_view = buf[header_size:]
    actual_payload_bytes = len(payload_view)

    # Decode float16 IQ pairs according to sender NumPy memory representation
    try:
        raw_scalars = np.frombuffer(payload_view, dtype=np.float16)
        iq_pairs = raw_scalars.reshape(-1, 2)
    except Exception as exc:
        raise ValueError(f"Failed to reshape payload into [N, 2] float16 IQ pairs: {exc}") from exc

    i_samples = iq_pairs[:, 0].astype(np.float64)
    q_samples = iq_pairs[:, 1].astype(np.float64)

    if not np.all(np.isfinite(i_samples)) or not np.all(np.isfinite(q_samples)):
        raise ValueError("Decoded UDP IQ payload contains NaN or infinite values")

    complex_samples = i_samples + 1j * q_samples

    return UDPIQPacket(
        signal_id=int(signal_id),
        channel_id=int(channel_id),
        iq_count=int(iq_count),
        complex_samples=complex_samples,
        i=i_samples,
        q=q_samples,
        payload_bytes=actual_payload_bytes,
        packet_bytes=raw_len,
        receive_timestamp_ns=receive_timestamp_ns,
        source_address=source_address,
        hardware_timestamp_ns=hardware_timestamp_ns,
    )


def make_udp_iq_packet(
    signal_id: int,
    channel_id: int,
    complex_samples: np.ndarray,
    hardware_timestamp_ns: int | None = None,
) -> bytes:
    """
    Constructs a wire-exact UDP IQ datagram matching reference hardware transmitter.

    Parameters
    ----------
    signal_id : int
        32-bit unsigned signal identifier.
    channel_id : int
        16-bit unsigned channel identifier (0..65535).
    complex_samples : np.ndarray
        Array of complex IQ samples to pack as interleaved float16.
    hardware_timestamp_ns : int | None, optional
        Genuine hardware capture timestamp in nanoseconds for sample 0.
        If None, formats legacy 10-byte '!IHI' header.
        If provided (integer >= 0), formats 18-byte '!IHIQ' header.

    Returns
    -------
    bytes
        Complete UDP IQ datagram ready for socket transmission.
    """
    signal_id = int(signal_id)
    if signal_id < 0 or signal_id > 0xFFFFFFFF:
        raise ValueError(f"signal_id must be in range 0..4294967295, got {signal_id}")

    channel_id = int(channel_id)
    if channel_id < 0 or channel_id > 65535:
        raise ValueError(f"channel_id must be in range 0..65535, got {channel_id}")

    arr = np.asarray(complex_samples, dtype=np.complex64).reshape(-1)
    iq_count = int(arr.size)
    if iq_count > MAX_IQ_COUNT:
        raise ValueError(f"iq_count {iq_count} exceeds maximum allowed sanity limit of {MAX_IQ_COUNT}")

    # Pack header
    if hardware_timestamp_ns is None:
        header = struct.pack(UDP_IQ_HEADER_FMT_LEGACY, signal_id, channel_id, iq_count)
    else:
        hw_ts = int(hardware_timestamp_ns)
        if hw_ts < 0 or hw_ts > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"hardware_timestamp_ns must be an unsigned 64-bit integer, got {hardware_timestamp_ns}")
        header = struct.pack(UDP_IQ_HEADER_FMT_TIMESTAMPED, signal_id, channel_id, iq_count, hw_ts)

    # Pack interleaved float16 payload
    if iq_count > 0:
        # np.complex64 is 2 x float32; casting view directly to float16
        iq_pairs = np.empty((iq_count, 2), dtype=np.float16)
        iq_pairs[:, 0] = arr.real.astype(np.float16)
        iq_pairs[:, 1] = arr.imag.astype(np.float16)
        payload = iq_pairs.tobytes()
    else:
        payload = b""

    return header + payload


def send_iq(
    sock: Any,
    dest_address: tuple[str, int],
    signal_id: int,
    channel_id: int,
    complex_samples: np.ndarray,
    hardware_timestamp_ns: int | None = None,
) -> int:
    """
    Transmits an IQ datagram over a UDP socket to destination address.

    Parameters
    ----------
    sock : socket.socket
        UDP socket to transmit through.
    dest_address : tuple[str, int]
        Destination (host, port) tuple.
    signal_id : int
        32-bit signal identifier.
    channel_id : int
        16-bit channel identifier.
    complex_samples : np.ndarray
        Complex samples to transmit.
    hardware_timestamp_ns : int | None, optional
        Hardware capture timestamp in nanoseconds for sample 0.
        If None, transmits legacy '!IHI' datagram.
        If provided (integer >= 0), transmits timestamped '!IHIQ' datagram.

    Returns
    -------
    int
        Number of bytes sent.
    """
    datagram = make_udp_iq_packet(
        signal_id=signal_id,
        channel_id=channel_id,
        complex_samples=complex_samples,
        hardware_timestamp_ns=hardware_timestamp_ns,
    )
    return sock.sendto(datagram, dest_address)


def udp_packet_to_canonical_iq(
    packet: UDPIQPacket,
    sample_rate: float | None = None,
    destination_port: int | None = None,
    adc_resolution: int | None = None,
) -> dict[str, Any]:
    """
    Converts a decoded UDPIQPacket into the canonical Acquisition and Processing
    IQ dictionary format, reusing existing parsing/canonicalization structures.

    Provenance Rules:
    -----------------
    1. Sample rate provenance:
       The wire UDP header does NOT contain sample rate. The sample rate MUST be
       supplied via configuration. 'sample_rate_source' is explicitly recorded.
    2. Timestamp semantics:
       - If genuine hardware capture timestamp exists (hardware_timestamp_ns is not None):
         record['timestamp'] = packet.hardware_timestamp_ns
         record['timestamp_unit'] = 'ns'
         metadata['timestamp_source'] = 'hardware'
       - If absent (legacy packet):
         record['timestamp'] = None
         record['timestamp_unit'] = 'ns'
         metadata['timestamp_source'] = 'none'
       - UDP network reception timestamp is preserved in metadata['transport']['receive_timestamp_ns']
         and is NEVER used as capture/hardware timestamp.
       - 'sample_timestamps' is preserved as None (NOT fabricated from network arrival).
    3. ADC Resolution semantics:
       Float16 is a transport encoding, NOT originating ADC hardware resolution.
       'adc_resolution' is None unless explicitly supplied by genuine source metadata.
       The transport format is recorded in metadata['transport']['sample_format'] = 'float16'.

    Parameters
    ----------
    packet : UDPIQPacket
        Decoded UDP packet.
    sample_rate : float | None
        Configured stream sample rate in Hz.
    destination_port : int | None
        Optional local UDP listening port for transport metadata.
    adc_resolution : int | None
        Explicit ADC resolution in bits if provided by genuine source metadata.

    Returns
    -------
    dict[str, Any]
        Canonical Acquisition and Processing IQ record.
    """
    fs = float(sample_rate) if sample_rate is not None else None
    ts = (1.0 / fs) if (fs is not None and fs > 0) else None

    if packet.hardware_timestamp_ns is not None:
        record_timestamp = packet.hardware_timestamp_ns
        timestamp_source = "hardware"
    else:
        record_timestamp = None
        timestamp_source = "none"

    transport_timestamp_source = "udp_receive" if packet.receive_timestamp_ns is not None else "none"

    transport_meta = {
        "protocol": "UDP",
        "signal_id": packet.signal_id,
        "channel_id": packet.channel_id,
        "source_ip": packet.source_address[0] if packet.source_address else None,
        "source_port": packet.source_address[1] if packet.source_address else None,
        "destination_port": destination_port,
        "packet_bytes": packet.packet_bytes,
        "payload_bytes": packet.payload_bytes,
        "receive_timestamp_ns": packet.receive_timestamp_ns,
        "hardware_timestamp_ns": packet.hardware_timestamp_ns,
        "sample_format": "float16",
        "timestamp_source": transport_timestamp_source,
    }

    metadata = {
        "transport": transport_meta,
        "sample_rate_source": "stream_configuration" if fs is not None else "unspecified",
        "timestamp_source": timestamp_source,
        "processing": [],
    }

    return {
        "i": packet.i.copy(),
        "q": packet.q.copy(),
        "complex_samples": packet.complex_samples.copy(),
        "sample_rate": fs,
        "sampling_period": ts,
        "sample_count": packet.iq_count,
        "timestamp": record_timestamp,
        "timestamp_unit": "ns",
        "sample_timestamps": None,  # NOT fabricated from network arrival
        "adc_resolution": adc_resolution,  # Transport float16 does not imply ADC resolution
        "adc_full_scale": None,
        "sample_units": "native",
        "quality_flags": set(),
        "metadata": metadata,
    }


__all__ = [
    "UDP_IQ_HEADER_FMT",
    "UDP_IQ_HEADER_SIZE",
    "UDP_IQ_HEADER_FMT_LEGACY",
    "UDP_IQ_HEADER_SIZE_LEGACY",
    "UDP_IQ_HEADER_FMT_TIMESTAMPED",
    "UDP_IQ_HEADER_SIZE_TIMESTAMPED",
    "UDP_MAX_DATAGRAM_BYTES",
    "MAX_IQ_COUNT",
    "BYTES_PER_IQ_SAMPLE",
    "UDPIQPacket",
    "parse_udp_iq_packet",
    "udp_packet_to_canonical_iq",
    "make_udp_iq_packet",
    "send_iq",
]


def main() -> None:
    print("=" * 60)
    print("  UDPIQPacket: Demonstration")
    print("=" * 60)
    # Synthesize test packet with hardware timestamp
    signal_id = 1001
    channel_id = 1
    iq_count = 8
    hw_ts_ns = 1700000000123456789
    i_vals = np.linspace(-0.8, 0.8, iq_count, dtype=np.float16)
    q_vals = np.linspace(0.8, -0.8, iq_count, dtype=np.float16)
    complex_samples = i_vals + 1j * q_vals

    raw_datagram = make_udp_iq_packet(
        signal_id=signal_id,
        channel_id=channel_id,
        complex_samples=complex_samples,
        hardware_timestamp_ns=hw_ts_ns,
    )

    packet = parse_udp_iq_packet(raw_datagram, receive_timestamp_ns=1700000000500000, sample_rate=1e6)
    print(f"Decoded Packet: signal_id={packet.signal_id}, ch={packet.channel_id}, count={packet.iq_count}")
    print(f"Hardware Timestamp: {packet.hardware_timestamp_ns} ns")
    print(f"Receive Timestamp: {packet.receive_timestamp_ns} ns")

    canonical_record = udp_packet_to_canonical_iq(packet, sample_rate=1e6)
    print(f"Canonical Record Timestamp: {canonical_record['timestamp']} {canonical_record['timestamp_unit']}")
    print(f"Timestamp Source: {canonical_record['metadata']['timestamp_source']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
