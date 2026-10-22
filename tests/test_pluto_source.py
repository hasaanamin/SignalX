"""Driver-boundary tests; simulated IIO is never hardware qualification evidence."""
import errno
import sys
import threading
import time
import types
from dataclasses import dataclass, replace

import numpy as np
import pytest

from signal_processing.core import StreamInfo
from signal_processing.sources.pluto_source import PlutoIQSource, PlutoSourceFactory
from signal_processing.sources.streaming import (
    SourceBusy,
    SourceCancelled,
    SourceClosed,
    SourceDisconnected,
    SourceError,
    SourceReadTimeout,
)


def _config(**kwargs):
    return PlutoSourceFactory('ip:simulated-only', StreamInfo('pluto-test', 'test', 1_920_000., 2.14e9), **kwargs)


class _Channel:
    def __init__(self, index, shift=0):
        self.index = index
        self.scan_element = True
        self.enabled = False
        self.data_format = types.SimpleNamespace(length=16, bits=12, is_signed=True,
            is_be=False, repeat=1, shift=shift)


class _Device:
    def __init__(self, raw, overflow_at=None, failure=None, shift=0, blocking=False, pace=False):
        self.channels = [_Channel(0, shift), _Channel(1, shift), _Channel(2)]
        self.sample_size = 4
        self.raw = raw
        self.status = 4  # Stale status must be cleared before starting a run.
        self.overflow_at = overflow_at
        self.failure = failure
        self.blocking, self.pace = blocking, pace
        self.buffers = self.refills = 0
        self.cancelled = threading.Event()
        self.entered = threading.Event()
        self.origin = None
        self.count = None

    def find_channel(self, name):
        return self.channels[int(name[-1])]

    def set_kernel_buffers_count(self, count):
        self.count = count

    def reg_read(self, address):
        assert address == 0x80000088
        return self.status

    def reg_write(self, address, value):
        assert address == 0x80000088 and value == 4
        self.status &= ~value


class _Buffer:
    def __init__(self, device, count, cyclic):
        assert not cyclic and device.count is not None
        self.device, self.count = device, count
        device.buffers += 1
        device.origin = time.perf_counter()

    def refill(self):
        device = self.device
        device.entered.set()
        if device.blocking:
            assert device.cancelled.wait(5)
            raise OSError(errno.ECANCELED, 'simulated cancellation')
        if device.failure:
            raise OSError(device.failure, 'simulated driver failure')
        device.refills += 1
        if device.pace:
            delay = device.origin + device.refills*self.count/1_920_000 - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        if device.refills == device.overflow_at:
            device.status |= 4

    def read(self):
        start = (self.device.refills-1)*self.count*4
        return bytearray(self.device.raw[start:start+self.count*4])

    def cancel(self):
        self.device.cancelled.set()


class _Pluto:
    def __init__(self, device, rate=1_920_000):
        self.ctx = self
        self.device = device
        self._rate = rate
        self.timeouts = []

    def set_timeout(self, value):
        self.timeouts.append(value)

    def find_device(self, name):
        assert name == 'cf-ad9361-lpc'
        return self.device

    @property
    def sample_rate(self):
        return self._rate

    @sample_rate.setter
    def sample_rate(self, value):
        assert value == 1_920_000


def _install_driver(device, rate=1_920_000):
    sdr = _Pluto(device, rate)
    sys.modules['adi'] = types.SimpleNamespace(Pluto=lambda uri: sdr)
    sys.modules['iio'] = types.SimpleNamespace(Buffer=_Buffer)
    return sdr


@pytest.fixture
def driver(monkeypatch):
    # monkeypatch tracks module restoration, including previously absent SDKs.
    def install(device, rate=1_920_000):
        sdr = _Pluto(device, rate)
        monkeypatch.setitem(sys.modules, 'adi', types.SimpleNamespace(Pluto=lambda uri: sdr))
        monkeypatch.setitem(sys.modules, 'iio', types.SimpleNamespace(Buffer=_Buffer))
        return sdr
    return install


@pytest.mark.parametrize('shift', [0, 4])
def test_exact_signed_iq_normalization_reused_buffer_and_finite_eof(driver, shift):
    values = np.tile(np.array([-2048, -2047, -1, 0, 1, 2047], np.int16), 1024)[:4096]
    raw = (values.astype(np.uint16) << shift).tobytes()
    device = _Device(raw, shift=shift)
    sdr = driver(device)
    source = PlutoIQSource(_config(block_size=1024, total_blocks=2))
    assert device.buffers == device.refills == 0  # Ready without RF acquisition.
    assert sdr.rx_lo == 2_140_000_000 and sdr.gain_control_mode_chan0 == 'manual'
    assert sdr.rx_hardwaregain_chan0 == 30 and device.count == 4
    assert not device.channels[2].enabled
    for sequence in range(2):
        envelope = source.read_envelope(.05)
        block = envelope.block
        expected = values.reshape(-1, 2)[sequence*1024:(sequence+1)*1024]
        np.testing.assert_array_equal(block.iq.real, expected[:, 0]/2048)
        np.testing.assert_array_equal(block.iq.imag, expected[:, 1]/2048)
        assert block.iq.dtype == np.complex64 and not block.iq.flags.writeable
        assert block.start_sample == sequence*1024 and block.sequence_number == sequence
        assert block.timestamp_ns is None and envelope.scheduled_monotonic_ns is None
        assert envelope.received_monotonic_ns > 0 and envelope.metadata.adc_scale == 1/2048
        with pytest.raises(ValueError):
            block.iq.setflags(write=True)
    assert source.read_envelope(.05) is None
    state = source.snapshot()
    assert state.finished and state.returned_samples == state.requested_samples == 2048
    assert not state.unknown_loss and state.hardware_overflow_events == 0
    assert device.buffers == 1 and device.refills == 2
    source.close()
    source.close()
    assert all(not channel.enabled for channel in device.channels)
    with pytest.raises(SourceClosed):
        source.read_envelope()


@pytest.mark.parametrize('failure', ['overflow', 'short', errno.ETIMEDOUT, errno.ENODEV])
def test_hardware_loss_and_transport_failure_are_sticky_not_eof(driver, failure):
    device = _Device(bytes(4096 if failure != 'short' else 4094),
        overflow_at=1 if failure == 'overflow' else None,
        failure=failure if isinstance(failure, int) else None)
    driver(device)
    source = PlutoIQSource(_config(block_size=1024, total_blocks=1))
    expected = SourceDisconnected if isinstance(failure, int) else SourceError
    with pytest.raises(expected):
        source.read_envelope(.05)
    state = source.snapshot()
    assert not state.finished and state.unknown_loss and state.returned_samples == 0
    assert state.hardware_overflow_events == (failure == 'overflow')
    assert state.transport_errors == isinstance(failure, int)
    with pytest.raises(SourceError):
        source.read_envelope(.05)
    assert device.refills <= 1
    source.close()


def test_cancel_interrupts_active_refill_without_freeing_reader_buffer(driver):
    device = _Device(bytes(4096), blocking=True)
    driver(device)
    source = PlutoIQSource(_config(block_size=1024))
    outcomes = []
    def read():
        try:
            source.read_envelope(.05)
        except Exception as exc:
            outcomes.append(exc)
    thread = threading.Thread(target=read)
    thread.start()
    assert device.entered.wait(5)
    with pytest.raises(SourceBusy):
        source.read_envelope()
    source.close()
    thread.join(5)
    assert not thread.is_alive() and len(outcomes) == 1
    assert isinstance(outcomes[0], SourceCancelled)
    assert source.snapshot().cancelled and not source.snapshot().unknown_loss


def test_sample_rate_readback_rejection_and_nonblocking_read_do_not_acquire(driver):
    device = _Device(bytes(4096))
    driver(device, rate=2_000_000)
    with pytest.raises(SourceError, match='readback'):
        PlutoIQSource(_config(block_size=1024))
    assert device.buffers == 0
    driver(device)
    source = PlutoIQSource(_config(block_size=1024))
    with pytest.raises(SourceReadTimeout):
        source.read_envelope(0)
    assert device.buffers == 0
    source.cancel()
    with pytest.raises(SourceCancelled):
        source.read_envelope(.05)
    source.close()


@pytest.mark.parametrize('attribute,value', [('length', 32), ('is_be', True),
    ('is_signed', False), ('bits', 16), ('shift', 5), ('repeat', 2)])
def test_unsupported_scan_format_fails_before_buffer_creation(driver, attribute, value):
    device = _Device(bytes(4096))
    setattr(device.channels[1].data_format, attribute, value)
    driver(device)
    with pytest.raises(SourceError, match='scan format'):
        PlutoIQSource(_config(block_size=1024))
    assert device.buffers == 0 and not any(c.enabled for c in device.channels)


def test_restart_changes_source_epoch_and_bandwidth_changes_configuration(driver):
    sources = []
    for bandwidth in (1_500_000, 1_500_000, 1_400_000):
        driver(_Device(bytes(4096)))
        source = PlutoIQSource(_config(block_size=1024, rf_bandwidth_hz=bandwidth))
        sources.append(source.read_envelope(.05))
        source.close()
    assert len({x.metadata.source_epoch for x in sources}) == 3
    assert sources[0].configuration_id == sources[1].configuration_id
    assert sources[1].configuration_id != sources[2].configuration_id


@pytest.mark.parametrize('arguments', [
    ['adaptive'], ['adaptive', '--initial-noise-power', 'nan'],
    ['adaptive', '--initial-noise-power', '.0001', '--require-mib'],
    ['lte', '--seconds', '0'], ['lte', '--block-size', '0'],
    ['lte', '--max-latency-ms', 'nan']])
def test_live_cli_rejects_invalid_arguments_before_hardware_is_opened(arguments):
    from signal_processing.runtime.pluto_cli import main
    with pytest.raises(SystemExit) as exc:
        main([*arguments, '--uri', 'ip:simulated-only', '--center-frequency', '2140000000'])
    assert exc.value.code == 2


@pytest.mark.parametrize('field,value', [('block_size', True), ('block_size', 0),
    ('total_blocks', 0), ('kernel_buffers', 100), ('gain_db', float('nan')),
    ('rf_bandwidth_hz', 2_000_000), ('uri', ''),
    ('stream', StreamInfo('bad', 'test', 2_000_000., 2.14e9)),
    ('stream', StreamInfo('bad', 'test', 1_920_000.))])
def test_invalid_hardware_configuration_is_rejected_without_sdk(field, value):
    with pytest.raises(ValueError):
        replace(_config(), **{field: value})


@dataclass(frozen=True)
class _SimulatedPlutoFactory:
    config: PlutoSourceFactory
    raw: bytes
    overflow_at: int | None = None

    @property
    def stream(self):
        return self.config.stream

    @property
    def block_size(self):
        return self.config.block_size

    def __call__(self):
        _install_driver(_Device(self.raw, overflow_at=self.overflow_at, pace=True))
        return PlutoIQSource(self.config)


@pytest.mark.parametrize('mode', ['adaptive', 'lte'])
@pytest.mark.parametrize('failure', ['none', 'sink', 'overflow'])
def test_real_spawn_live_runner_processes_all_iq_or_reports_failure(mode, failure):
    from signal_processing.runtime.pluto_cli import run
    if mode == 'lte':
        from tests.lte_pbch_fixture import waveform
        samples = waveform(42, ports=2, frames=12, rate=1_920_000., snr_db=20.)
        # Keep the independent encoded waveform within the 12-bit ADC range.
        samples *= .8/max(np.abs(samples.real).max(), np.abs(samples.imag).max())
    else:
        samples = np.full(65536, .003+.002j, np.complex64)
        samples[1024:1280] += .1
    count = (len(samples)+16383)//16384
    samples = np.pad(samples, (0, count*16384-len(samples)))
    words = np.empty((len(samples), 2), '<i2')
    words[:, 0] = np.rint(samples.real*2048).astype(np.int16)
    words[:, 1] = np.rint(samples.imag*2048).astype(np.int16)
    quantized = (words[:, 0].astype(np.float32)+1j*words[:, 1].astype(np.float32))/2048
    factory = _SimulatedPlutoFactory(_config(total_blocks=count), words.tobytes(),
        overflow_at=2 if failure == 'overflow' else None)
    delivered = []
    def sink(value):
        if failure == 'sink':
            raise RuntimeError('real live sink failure')
        if mode == 'adaptive':
            event = value.event
            assert event.iq.tobytes() == quantized[event.event_start_sample:event.event_end_sample].tobytes()
            assert value.features is None
        delivered.append(value)
    result = run(mode, factory, initial_noise_power=1e-4, result_sink=sink,
        max_latency_ms=.000001, performance_qos=False, priority_class='normal')
    assert result['acquisition']['state'] == 'CLOSED'
    assert not result['acquisition']['reader_alive'] and result['acquisition']['outstanding_deliveries'] == 0
    assert not result['publication']['writer_alive']
    assert not result['hardware_qualified'] and not result['passed']
    if failure == 'none':
        assert result['acquisition_passed'], result
        assert result['processed_samples'] == len(samples) and delivered
        assert not result['latency_passed']  # Real work cannot meet the impossible deadline.
        if mode == 'lte':
            assert result['decoded_mibs'] >= 1
            assert all(mib.pci == 42 for batch in delivered for mib in batch.mibs)
    else:
        assert not result['acquisition_passed'] and result['errors']
        if failure == 'overflow':
            assert result['source']['unknown_loss'] and result['source']['hardware_overflow_events'] == 1
        else:
            assert any('sink failure' in error for error in result['errors'])
