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

import collections
import copy
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

from .UDP_Acquisition import (
    UDP_IQ_HEADER_FMT,
    UDP_IQ_HEADER_SIZE,
    UDP_IQ_HEADER_FMT_LEGACY,
    UDP_IQ_HEADER_SIZE_LEGACY,
    UDP_IQ_HEADER_FMT_TIMESTAMPED,
    UDP_IQ_HEADER_SIZE_TIMESTAMPED,
    UDP_MAX_DATAGRAM_BYTES,
    UDPIQPacket,
    parse_udp_iq_packet,
    udp_packet_to_canonical_iq,
    make_udp_iq_packet,
    send_iq,
)


@dataclass
class UDPReceiverConfig:
    """
    Configuration parameters for the reusable UDP IQ Receiver.

    Attributes
    ----------
    bind_ip : str
        Local network interface IP address to bind to (e.g. '0.0.0.0' or '127.0.0.1').
    port : int
        Local UDP port to listen for incoming IQ datagrams.
    sample_rate_hz : float
        Configured stream sample rate in Hz (sample rate is NOT in the UDP header).
    socket_receive_buffer_bytes : int
        OS socket receive buffer size in bytes (defaults to 8 MB).
    timeout_s : float
        Socket select/receive polling timeout in seconds.
    queue_capacity : int
        Maximum bounded queue depth for assembled frames before oldest frames are dropped.
    """
    bind_ip: str = "0.0.0.0"
    port: int = 5000
    sample_rate_hz: float = 1.0e6
    socket_receive_buffer_bytes: int = 8 * 1024 * 1024
    timeout_s: float = 0.5
    queue_capacity: int = 100


@dataclass
class IQFrame:
    """
    Assembled multi-channel or single-channel frame of canonical IQ records.

    Attributes
    ----------
    signal_id : int
        32-bit emitter/signal stream identifier.
    channels : dict[int, dict]
        Mapping from channel_id (int) to canonical Acquisition and Processing IQ record.
    reception_metadata : dict
        Transport reception information including timestamp, source address, packet counts.
    observed_channel_structure : dict
        Observed vs validated channel counts, learned signal classification, validation status.
    validation_state : dict
        Structural validation flags (e.g. frame_structure_valid, structure_mismatch, transport_integrity).
    quality_flags : set[str]
        Set of quality / diagnostic flags associated with this frame.
    """
    signal_id: int
    channels: dict[int, dict] = field(default_factory=dict)
    reception_metadata: dict[str, Any] = field(default_factory=dict)
    observed_channel_structure: dict[str, Any] = field(default_factory=dict)
    validation_state: dict[str, Any] = field(default_factory=dict)
    quality_flags: set[str] = field(default_factory=set)

    @property
    def hardware_timestamp_ns(self) -> int | None:
        """Frame-level hardware capture timestamp if present across packet channels."""
        return self.reception_metadata.get("hardware_timestamp_ns")


class UDPFrameAssembler:
    """
    Stateful frame and channel assembler that replicates the reference receiver's
    channel block accumulation, rollover detection, and structure-learning behavior.

    Frame Rollover Logic:
    ---------------------
    A frame rollover is inferred when:
    1. The signal_id changes from the current pending signal.
    2. An incoming packet contains a channel_id that was ALREADY received in the pending frame.
    3. An incoming packet has a channel_id <= the last observed channel_id in the current frame.

    Structure Validation & Learning:
    ---------------------------------
    - Observes channel sets across consecutive frames over a sliding history window.
    - After N consecutive frames with identical channel counts/sets (default N=5),
      the channel structure is marked as validated (`structure_validated = True`).
    - Classifies stream as 'single_channel' (N=1) or 'multi_channel' (N>1).
    - If later frames deviate from this validated structure, a `frame_structure_mismatch`
      is flagged.
    - Transport integrity is reported as 'unknown' or 'inferred_anomaly', avoiding
      unsubstantiated claims of guaranteed packet loss or reordering.
    """

    def __init__(self, validation_window: int = 5) -> None:
        try:
            val_win = int(validation_window)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"validation_window must be an integer >= 1, got {validation_window!r}") from exc
        if val_win < 1:
            raise ValueError(f"validation_window must be >= 1, got {val_win}")
        self.validation_window = val_win
        self.reset()

    def reset(self) -> None:
        """Resets all pending frame state and structure observation history."""
        self._pending_channels: dict[int, dict] = {}
        self._pending_packets: list[UDPIQPacket] = []
        self._pending_signal_id: int | None = None
        self._last_channel_id: int | None = None

        # Frame structure learning state
        self._history_channel_counts: collections.deque[int] = collections.deque(maxlen=self.validation_window)
        self._history_channel_sets: collections.deque[tuple[int, ...]] = collections.deque(maxlen=self.validation_window)
        self._validated_channel_count: int | None = None
        self._validated_channel_ids: tuple[int, ...] | None = None
        self._validated_signal_type: str = "unvalidated"
        self._structure_validated: bool = False

        # Diagnostic counters
        self._frames_observed: int = 0
        self._structure_mismatches: int = 0
        self._duplicate_channel_observations: int = 0
        self._unexpected_channel_orders: int = 0
        self._suspected_loss_events: int = 0

    def process_packet(
        self,
        packet: UDPIQPacket,
        sample_rate: float | None = None,
        destination_port: int | None = None,
    ) -> list[IQFrame]:
        """
        Ingests a decoded UDPIQPacket, checks for frame rollover, updates state,
        and returns any newly completed IQFrame(s).

        Parameters
        ----------
        packet : UDPIQPacket
            Decoded UDP datagram packet.
        sample_rate : float | None
            Configured stream sample rate in Hz.
        destination_port : int | None
            Local UDP listening port.

        Returns
        -------
        list[IQFrame]
            List of completed frames (empty if frame is still accumulating).
        """
        emitted_frames: list[IQFrame] = []

        # Check for frame rollover conditions
        is_rollover = False
        duplicate_in_frame = False
        order_anomaly = False

        if self._pending_signal_id is not None:
            if packet.signal_id != self._pending_signal_id:
                is_rollover = True
            elif packet.channel_id in self._pending_channels:
                is_rollover = True
                first_ch = self._pending_packets[0].channel_id if self._pending_packets else None
                if packet.channel_id == first_ch:
                    # Normal rollover marking start of next frame cycle
                    duplicate_in_frame = False
                else:
                    # Repeated channel within the same frame cycle (not start channel)
                    duplicate_in_frame = True
                    self._duplicate_channel_observations += 1
            elif self._last_channel_id is not None and packet.channel_id < self._last_channel_id:
                # Out-of-order channel arrived (e.g. 1, 3, 2, 4) -> triggers rollover with order anomaly
                is_rollover = True
                order_anomaly = True
                self._unexpected_channel_orders += 1


        if is_rollover and self._pending_channels:
            finalized_frame = self._finalize_pending_frame(
                duplicate_flag=duplicate_in_frame,
                order_anomaly=order_anomaly,
            )
            if finalized_frame is not None:
                emitted_frames.append(finalized_frame)

        # Accumulate current packet into pending frame
        canonical_record = udp_packet_to_canonical_iq(
            packet,
            sample_rate=sample_rate,
            destination_port=destination_port,
        )

        self._pending_signal_id = packet.signal_id
        self._pending_channels[packet.channel_id] = canonical_record
        self._pending_packets.append(packet)
        self._last_channel_id = packet.channel_id

        return emitted_frames

    def flush(self) -> IQFrame | None:
        """
        Flushes and finalizes any remaining pending channels into a completed frame.
        """
        if not self._pending_channels:
            return None
        return self._finalize_pending_frame()

    def _finalize_pending_frame(
        self,
        duplicate_flag: bool = False,
        order_anomaly: bool = False,
    ) -> IQFrame | None:
        """Internal helper to assemble and validate a pending channel set into an IQFrame."""
        if not self._pending_channels or self._pending_signal_id is None:
            return None

        self._frames_observed += 1
        observed_channel_ids = tuple(sorted(self._pending_channels.keys()))
        observed_channel_count = len(observed_channel_ids)

        # Update structure observation history
        self._history_channel_counts.append(observed_channel_count)
        self._history_channel_sets.append(observed_channel_ids)

        # Structure validation learning
        if len(self._history_channel_counts) >= self.validation_window:
            # Check if all observations in window are identical
            first_count = self._history_channel_counts[0]
            first_set = self._history_channel_sets[0]
            if all(c == first_count for c in self._history_channel_counts) and all(
                s == first_set for s in self._history_channel_sets
            ):
                self._structure_validated = True
                self._validated_channel_count = first_count
                self._validated_channel_ids = first_set
                self._validated_signal_type = "single_channel" if first_count == 1 else "multi_channel"

        # Check for structure mismatch against validated structure
        structure_mismatch = False
        frame_structure_valid = True
        transport_integrity = "unknown"

        if self._structure_validated and self._validated_channel_count is not None:
            if observed_channel_count != self._validated_channel_count or observed_channel_ids != self._validated_channel_ids:
                structure_mismatch = True
                frame_structure_valid = False
                transport_integrity = "inferred_anomaly"
                self._structure_mismatches += 1
                self._suspected_loss_events += 1

        quality_flags: set[str] = set()
        if duplicate_flag:
            quality_flags.add("duplicate_channel_observed")
        if order_anomaly:
            quality_flags.add("unexpected_channel_order")
        if structure_mismatch:
            quality_flags.add("frame_structure_mismatch")

        recv_timestamps = [p.receive_timestamp_ns for p in self._pending_packets if p.receive_timestamp_ns is not None]
        hw_timestamps = [p.hardware_timestamp_ns for p in self._pending_packets if p.hardware_timestamp_ns is not None]
        source_addrs = [p.source_address for p in self._pending_packets if p.source_address is not None]

        frame_hw_ts = None
        hw_ts_consistent = None
        if hw_timestamps:
            if len(hw_timestamps) == len(self._pending_packets) and all(ts == hw_timestamps[0] for ts in hw_timestamps):
                frame_hw_ts = hw_timestamps[0]
                hw_ts_consistent = True
            else:
                frame_hw_ts = None
                hw_ts_consistent = False

        reception_meta = {
            "packets_in_frame": len(self._pending_packets),
            "first_receive_timestamp_ns": recv_timestamps[0] if recv_timestamps else None,
            "last_receive_timestamp_ns": recv_timestamps[-1] if recv_timestamps else None,
            "hardware_timestamp_ns": frame_hw_ts,
            "hardware_timestamps_consistent": hw_ts_consistent,
            "source_address": source_addrs[0] if source_addrs else None,
        }

        observed_structure = {
            "observed_channel_ids": list(observed_channel_ids),
            "observed_channel_count": observed_channel_count,
            "validated_channel_count": self._validated_channel_count,
            "validated_channel_ids": list(self._validated_channel_ids) if self._validated_channel_ids else None,
            "validated_signal_type": self._validated_signal_type,
            "structure_validated": self._structure_validated,
            "history_window_size": len(self._history_channel_counts),
        }

        val_state = {
            "frame_structure_valid": frame_structure_valid,
            "structure_mismatch": structure_mismatch,
            "duplicate_channel_observed": duplicate_flag,
            "unexpected_channel_order": order_anomaly,
            "transport_integrity": transport_integrity,  # Honest: "unknown" when normal, "inferred_anomaly" on mismatch
        }

        frame = IQFrame(
            signal_id=self._pending_signal_id,
            channels=copy.deepcopy(self._pending_channels),
            reception_metadata=reception_meta,
            observed_channel_structure=observed_structure,
            validation_state=val_state,
            quality_flags=quality_flags,
        )

        # Clear pending accumulator
        self._pending_channels.clear()
        self._pending_packets.clear()
        self._pending_signal_id = None
        self._last_channel_id = None

        return frame

    def get_diagnostics(self) -> dict[str, Any]:
        """Returns diagnostic statistics for frame assembly."""
        return {
            "frames_observed": self._frames_observed,
            "structure_validated": self._structure_validated,
            "validated_channel_count": self._validated_channel_count,
            "validated_signal_type": self._validated_signal_type,
            "structure_mismatches": self._structure_mismatches,
            "duplicate_channel_observations": self._duplicate_channel_observations,
            "unexpected_channel_orders": self._unexpected_channel_orders,
            "suspected_loss_events": self._suspected_loss_events,
        }


class UDPReceiver:
    """
    Reusable, non-GUI, thread-safe UDP receiver for incoming IQ stream datagrams.

    Features:
    ---------
    - Binds UDP socket with configurable receive buffer.
    - Background worker thread with clean start/stop.
    - Strict packet decoding and validation.
    - Stateful frame assembly and structural learning.
    - Bounded FIFO queue (drops oldest frames on saturation to prevent stalling UDP socket).
    - Callback registration and synchronous polling API.
    - Comprehensive transport and validation statistics.
    """

    def __init__(
        self,
        config: UDPReceiverConfig | None = None,
        assembler: UDPFrameAssembler | None = None,
    ) -> None:
        self.config = config or UDPReceiverConfig()
        self.assembler = assembler or UDPFrameAssembler()

        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Bounded frame queue
        self._frame_queue: collections.deque[IQFrame] = collections.deque(maxlen=self.config.queue_capacity)
        self._frame_available = threading.Condition(self._lock)
        self._callbacks: list[Callable[[IQFrame], None]] = []

        # Statistics
        self._datagrams_received: int = 0
        self._valid_datagrams: int = 0
        self._invalid_datagrams: int = 0
        self._bytes_received: int = 0
        self._frames_emitted: int = 0
        self._dropped_oldest_frames: int = 0
        self._last_validation_error: str | None = None
        self._last_sender: tuple[str, int] | None = None
        self._current_signal_id: int | None = None

    def start(self) -> None:
        """Binds the UDP socket and starts the background worker thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Set receive buffer size
            try:
                self._socket.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_RCVBUF,
                    self.config.socket_receive_buffer_bytes,
                )
            except OSError:
                pass

            self._socket.settimeout(self.config.timeout_s)
            self._socket.bind((self.config.bind_ip, self.config.port))

            self._thread = threading.Thread(
                target=self._worker_loop,
                name=f"UDPReceiver-{self.config.port}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stops the background worker thread and closes the socket."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None

    def is_running(self) -> bool:
        """Returns True if the receiver worker thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def on_frame(self, callback: Callable[[IQFrame], None]) -> None:
        """Registers a callback invoked on each newly assembled IQFrame."""
        with self._lock:
            self._callbacks.append(callback)

    def get_frame(self, timeout: float | None = None) -> IQFrame | None:
        """
        Retrieves the next assembled IQFrame from the bounded queue.

        Parameters
        ----------
        timeout : float | None
            Maximum time in seconds to wait for a frame. None waits indefinitely.

        Returns
        -------
        IQFrame | None
            The next assembled frame, or None if timeout expired.
        """
        with self._frame_available:
            if not self._frame_queue:
                if timeout == 0:
                    return None
                success = self._frame_available.wait(timeout=timeout)
                if not success or not self._frame_queue:
                    return None
            return self._frame_queue.popleft()

    def get_stats(self) -> dict[str, Any]:
        """Returns a snapshot of transport, validation, and queue metrics."""
        with self._lock:
            diag = self.assembler.get_diagnostics()
            return {
                "datagrams_received": self._datagrams_received,
                "valid_datagrams": self._valid_datagrams,
                "invalid_datagrams": self._invalid_datagrams,
                "bytes_received": self._bytes_received,
                "frames_emitted": self._frames_emitted,
                "queue_depth": len(self._frame_queue),
                "dropped_oldest_frames": self._dropped_oldest_frames,
                "last_validation_error": self._last_validation_error,
                "last_sender": self._last_sender,
                "current_signal_id": self._current_signal_id,
                **diag,
            }

    def _worker_loop(self) -> None:
        """Continuous background UDP reception and frame assembly loop."""
        while not self._stop_event.is_set():
            if self._socket is None:
                break
            try:
                data, addr = self._socket.recvfrom(UDP_MAX_DATAGRAM_BYTES)
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                continue

            recv_time_ns = time.time_ns()
            raw_len = len(data)

            with self._lock:
                self._datagrams_received += 1
                self._bytes_received += raw_len
                self._last_sender = addr

                # Decode and validate packet
                try:
                    packet = parse_udp_iq_packet(
                        data,
                        receive_timestamp_ns=recv_time_ns,
                        source_address=addr,
                        sample_rate=self.config.sample_rate_hz,
                    )
                    self._valid_datagrams += 1
                    self._current_signal_id = packet.signal_id
                except Exception as exc:
                    self._invalid_datagrams += 1
                    self._last_validation_error = str(exc)
                    continue

                # Pass to frame assembler
                frames = self.assembler.process_packet(
                    packet,
                    sample_rate=self.config.sample_rate_hz,
                    destination_port=self.config.port,
                )

                for frame in frames:
                    self._frames_emitted += 1
                    # Bounded queue drop-oldest management
                    if len(self._frame_queue) >= self._frame_queue.maxlen:
                        self._dropped_oldest_frames += 1

                    self._frame_queue.append(frame)
                    self._frame_available.notify()

                    # Trigger registered callbacks
                    for cb in list(self._callbacks):
                        try:
                            cb(frame)
                        except Exception:
                            pass


__all__ = [
    "UDPReceiverConfig",
    "IQFrame",
    "UDPFrameAssembler",
    "UDPReceiver",
    "send_iq",
]


def main() -> None:
    print("=" * 60)
    print("  UDPReceiver & Frame Assembler: Demonstration")
    print("=" * 60)
    assembler = UDPFrameAssembler(validation_window=3)
    sample_rate = 1.0e6

    # Feed 6 consecutive 2-channel frames to demonstrate structure learning
    signal_id = 42
    for frame_idx in range(6):
        for ch in [1, 2]:
            raw_i = np.ones(16, dtype=np.float16) * ch * 0.1
            raw_q = np.zeros(16, dtype=np.float16)
            payload = np.column_stack([raw_i, raw_q]).tobytes()
            header = struct.pack(UDP_IQ_HEADER_FMT, signal_id, ch, 16)
            pkt = parse_udp_iq_packet(header + payload, receive_timestamp_ns=time.time_ns())
            emitted = assembler.process_packet(pkt, sample_rate=sample_rate)
            for f in emitted:
                print(f"Emitted Frame: signal_id={f.signal_id}, channels={list(f.channels.keys())}")
                print(f"  Structure: {f.observed_channel_structure['validated_signal_type']}, "
                      f"valid={f.validation_state['frame_structure_valid']}")

    # Flush any remaining frame
    last_frame = assembler.flush()
    if last_frame:
        print(f"Final Flushed Frame: signal_id={last_frame.signal_id}, channels={list(last_frame.channels.keys())}")
        print(f"  Validation State: {last_frame.validation_state}")

    print("Diagnostics:", assembler.get_diagnostics())
    print("=" * 60)


if __name__ == "__main__":
    main()
