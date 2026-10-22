"""Receive-only PlutoSDR acquisition using pyadi configuration and libiio buffers.

Requires the libiio 0.x Python API and native driver. No receive buffer exists
until the first read, so the process runner's paused startup cannot lose RF
while the DSP compiles. A buffer is reused continuously, never recreated per
read. DMA overflow and transport errors terminate acquisition; neither is EOF.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

from signal_processing.core import IQBlock, StreamInfo

from .streaming import (
    SourceBusy,
    SourceCancelled,
    SourceClosed,
    SourceDisconnected,
    SourceEnvelope,
    SourceError,
    SourceMetadata,
    SourceReadTimeout,
    read_timeout,
)

_DMA_STATUS = 0x80000088
_OVERFLOW = 4


@dataclass(frozen=True)
class PlutoSnapshot:
    requested_samples: int | None
    returned_samples: int
    finished: bool
    cancelled: bool
    hardware_overflow_events: int
    transport_errors: int
    unknown_loss: bool
    sample_rate_hz: int
    kernel_buffers: int
    block_size: int
    acquisition_started_ns: int | None
    last_received_ns: int | None
    maximum_refill_ms: float
    uri: str
    center_frequency_hz: int
    rf_bandwidth_hz: int
    gain_db: float
    source_epoch: int
    driver_versions: dict
    # Standard Pluto IQ has no per-sample hardware sequence counter. The DMA
    # sticky overflow flag detects overruns but cannot count the missing IQ.
    continuity_evidence: str = 'sticky_dma_overflow_flag_no_sample_counter'


@dataclass(frozen=True)
class PlutoSourceFactory:
    """Spawn-safe configuration; use one operating mode per receiver process."""
    uri: str
    stream: StreamInfo
    block_size: int = 16384
    total_blocks: int | None = None
    rf_bandwidth_hz: int = 1_500_000
    gain_db: float = 30.0
    kernel_buffers: int = 4

    def __post_init__(self):
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ValueError('Supply a Pluto IIO URI')
        if not isinstance(self.stream, StreamInfo) or self.stream.sample_rate_hz != 1_920_000:
            raise ValueError('Pluto native acquisition requires a 1.92 MS/s stream')
        frequency = self.stream.center_frequency_hz
        if frequency is None or not 325_000_000 <= frequency <= 3_800_000_000 or frequency != int(frequency):
            raise ValueError('Supply an integer centre frequency in the standard Pluto 325 MHz–3.8 GHz range')
        if type(self.block_size) is not int or not 1024 <= self.block_size <= 65536:
            raise ValueError('Pluto block_size must be in [1024, 65536]')
        if self.total_blocks is not None and (type(self.total_blocks) is not int or self.total_blocks < 1):
            raise ValueError('total_blocks must be a positive integer or None')
        if type(self.kernel_buffers) is not int or not 2 <= self.kernel_buffers <= 8:
            raise ValueError('kernel_buffers must be in [2, 8]')
        if type(self.rf_bandwidth_hz) is not int or not 200_000 <= self.rf_bandwidth_hz <= 1_920_000:
            raise ValueError('RF bandwidth must be an integer in [200000, 1920000] Hz')
        if isinstance(self.gain_db, bool) or not isinstance(self.gain_db, (int, float)) or not math.isfinite(self.gain_db) or not -3 <= self.gain_db <= 71:
            raise ValueError('Manual receive gain must be in [-3, 71] dB')

    def __call__(self):
        return PlutoIQSource(self)


class PlutoIQSource:
    """Single reader, bounded storage, manual gain and normalized complex64 IQ.

    A finite run ends only after total_blocks successful refills. A driver
    timeout is fatal: retrying a partially consumed USB/network buffer could
    conceal an unknown hole. cancel() interrupts a pending libiio refill.
    No transmit path, digital loopback, or firmware settings are enabled.
    """
    def __init__(self, config: PlutoSourceFactory):
        self._config = config
        self._gate = threading.Lock()
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._closed = False
        self._buffer = self._sdr = self._device = None
        self._channels = []
        self._sequence = self._returned = self._overflows = self._errors = 0
        self._unknown = self._finished = False
        self._failure = None
        self._started = self._last = None
        self._maximum_refill_ms = 0.
        try:
            try:
                import adi
                import iio
            except (ImportError, OSError) as exc:
                raise SourceError('Pluto requires pyadi-iio, pylibiio 0.x and the native libiio driver; install the pluto extra and ADI drivers') from exc
            if not hasattr(iio, 'Buffer'):
                raise SourceError('Pluto requires the libiio 0.x Buffer API')
            self._iio = iio
            self._sdr = sdr = adi.Pluto(uri=config.uri)
            sdr.ctx.set_timeout(1000)
            sdr.rx_enabled_channels = [0]
            # pyadi configures AD936x FIR decimation for rates below 2.083 MS/s.
            sdr.sample_rate = 1_920_000
            sdr.rx_lo = int(config.stream.center_frequency_hz)
            sdr.rx_rf_bandwidth = config.rf_bandwidth_hz
            sdr.gain_control_mode_chan0 = 'manual'
            sdr.rx_hardwaregain_chan0 = config.gain_db
            if int(sdr.sample_rate) != 1_920_000:
                raise SourceError(f'Pluto sample-rate readback is {sdr.sample_rate}, expected 1920000')
            if int(sdr.rx_lo) != int(config.stream.center_frequency_hz):
                raise SourceError('Pluto LO readback differs from the requested centre frequency')
            if sdr.gain_control_mode_chan0 != 'manual':
                raise SourceError('Pluto did not enter manual gain mode')
            gain = float(sdr.rx_hardwaregain_chan0)
            self._bandwidth = int(sdr.rx_rf_bandwidth)
            identity = json.dumps([1_920_000, int(sdr.rx_lo), self._bandwidth, gain])
            configuration = int.from_bytes(hashlib.sha256(identity.encode('ascii')).digest()[:8], 'little')
            self._metadata = SourceMetadata(source_epoch=uuid.uuid4().int,
                configuration_epoch=configuration, gain_mode='manual', gain_db=gain, adc_scale=1/2048.)
            self._versions = dict(pyadi_iio=str(getattr(adi, '__version__', 'unknown')),
                pylibiio=str(getattr(iio, 'version', 'unknown')),
                context=str(getattr(sdr.ctx, 'version', 'unknown')))
            self._device = device = sdr.ctx.find_device('cf-ad9361-lpc')
            if device is None:
                raise SourceError('Pluto receive DMA device cf-ad9361-lpc is unavailable')
            for channel in device.channels:
                if channel.scan_element:
                    channel.enabled = False
            self._channels = [device.find_channel('voltage0'), device.find_channel('voltage1')]
            for index, channel in enumerate(self._channels):
                if channel is None:
                    raise SourceError('Pluto receive I/Q channels are unavailable')
                fmt = channel.data_format
                if (channel.index != index or fmt.length != 16 or fmt.bits != 12
                        or not fmt.is_signed or fmt.is_be or fmt.repeat != 1
                        or fmt.shift not in range(5)):
                    raise SourceError('Unsupported Pluto I/Q scan format; expected interleaved little-endian signed 12-in-16')
                channel.enabled = True
            self._shifts = tuple(c.data_format.shift for c in self._channels)
            if device.sample_size != 4:
                raise SourceError('Pluto scan must contain exactly one complex I/Q pair')
            device.set_kernel_buffers_count(config.kernel_buffers)
            # Ensure required monitoring is supported before reporting ready.
            device.reg_read(_DMA_STATUS)
        except BaseException:
            self.close()
            raise

    @property
    def stream_info(self):
        return self._config.stream

    def snapshot(self):
        with self._lock:
            config = self._config
            return PlutoSnapshot(None if config.total_blocks is None else config.total_blocks*config.block_size,
                self._returned, self._finished, self._cancelled.is_set(), self._overflows,
                self._errors, self._unknown, 1_920_000, config.kernel_buffers, config.block_size,
                self._started, self._last, self._maximum_refill_ms, config.uri,
                int(config.stream.center_frequency_hz), self._bandwidth, self._metadata.gain_db,
                self._metadata.source_epoch, dict(self._versions))

    def cancel(self):
        self._cancelled.set()
        with self._lock:
            if self._buffer is not None:
                self._buffer.cancel()

    def close(self):
        self.cancel()
        # Cancellation happens before waiting for the sole reader. Never free
        # its C buffer while a GIL-released refill is still using it.
        with self._gate:
            with self._lock:
                if self._closed:
                    return
                self._buffer = None
                for channel in self._channels:
                    if channel is not None:
                        channel.enabled = False
                self._channels = []
                self._device = self._sdr = None
                self._closed = True

    def read_envelope(self, timeout=None):
        read_timeout(timeout)
        if not self._gate.acquire(False):
            raise SourceBusy('Pluto supports one reader')
        try:
            if self._closed:
                raise SourceClosed('Pluto is closed')
            if self._cancelled.is_set():
                raise SourceCancelled('Pluto acquisition was cancelled')
            if self._failure is not None:
                raise SourceError(self._failure)
            if self._config.total_blocks is not None and self._sequence == self._config.total_blocks:
                self._finished = True
                return None
            if timeout == 0:
                raise SourceReadTimeout('Pluto requires a positive refill deadline')
            # A bounded timeout also lets ProcessSourceRunner observe shutdown
            # between reads, without requiring a driver object in the parent.
            self._sdr.ctx.set_timeout(max(1, math.ceil(min(.05, timeout if timeout is not None else .05)*1000)))
            if self._buffer is None:
                with self._lock:
                    if self._cancelled.is_set():
                        raise SourceCancelled('Pluto acquisition was cancelled')
                    self._device.reg_write(_DMA_STATUS, _OVERFLOW)
                    self._started = time.perf_counter_ns()
                    self._buffer = self._iio.Buffer(self._device, self._config.block_size, False)
            before = time.perf_counter_ns()
            self._buffer.refill()
            received = time.perf_counter_ns()
            raw = self._buffer.read()
            if self._device.reg_read(_DMA_STATUS) & _OVERFLOW:
                self._overflows += 1
                self._unknown = True
                self._failure = 'Pluto DMA receive overflow: missing sample count is unknown'
                raise SourceError(self._failure)
            if len(raw) != self._config.block_size*4:
                self._unknown = True
                self._failure = f'Short Pluto receive buffer: {len(raw)} bytes'
                raise SourceError(self._failure)
            words = np.frombuffer(raw, dtype='<u2').reshape(-1, 2)
            iq = np.empty(self._config.block_size, np.complex64)
            for index, destination in enumerate((iq.real, iq.imag)):
                # Explicit 12-bit sign extension handles either sign-extended
                # or zero-padded DMA storage, with no complex128 intermediate.
                values = ((words[:, index] >> self._shifts[index]) & 4095).astype(np.int16)
                np.copyto(destination, ((values ^ 2048)-2048), casting='unsafe')
                destination *= np.float32(1/2048.)
            block = IQBlock(self.stream_info, self._sequence, self._returned, timestamp_ns=None, iq=iq)
            with self._lock:
                self._sequence += 1
                self._returned += len(iq)
                self._last = received
                self._maximum_refill_ms = max(self._maximum_refill_ms, (received-before)/1e6)
            # Pluto supplies neither a host monotonic acquisition clock nor an
            # RF timestamp. Do not manufacture either from a successful read.
            return SourceEnvelope(block, self._metadata, received)
        except (SourceCancelled, SourceClosed, SourceReadTimeout):
            raise
        except OSError as exc:
            if self._cancelled.is_set() or exc.errno == errno.ECANCELED:
                raise SourceCancelled('Pluto refill cancelled') from exc
            self._errors += 1
            self._unknown = True
            self._failure = f'Pluto transport failed (including timeout): {exc}'
            raise SourceDisconnected(self._failure) from exc
        except Exception as exc:
            # Any unexpected failure after a buffer might have advanced is
            # terminal too; a caller cannot retry into apparently contiguous IQ.
            self._unknown = True
            self._failure = self._failure or f'Pluto receive failed: {type(exc).__name__}: {exc}'
            raise
        finally:
            self._gate.release()
