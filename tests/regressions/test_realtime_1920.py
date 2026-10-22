"""Independent SSS arithmetic and chronological adaptive-prefix parity."""

import math
import sys

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import POSITIVE_SNR_IID, POSITIVE_SNR_IID_1920


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize("repeats", [2, 4, 8])
def test_integrated_sss_matches_independent_complex_sum(scale, repeats):
    from signal_processing.lte._native import _integrated_sss, initialize_sync
    from signal_processing.lte.sequences import SYNC_BINS, pss_sequence, sss_hypotheses

    initialize_sync("numba")
    rng = np.random.default_rng(23481)
    spectra = (rng.normal(size=(2, repeats, 128)) + 1j * rng.normal(size=(2, repeats, 128))) * scale
    pss, bank = pss_sequence(2), sss_hypotheses(2)
    accumulated = np.zeros(336, complex)
    energy = 0.0
    for repeat in range(repeats):
        channel = spectra[0, repeat, SYNC_BINS] * pss.conj()
        equalized = []
        for j in range(62):
            base = (j // 31) * 31
            values = channel[max(base, j - 5) : min(base + 31, j + 6)]
            smooth = complex(
                math.fsum(v.real for v in values), math.fsum(v.imag for v in values)
            ) / len(values)
            equalized.append(spectra[1, repeat, SYNC_BINS[j]] * smooth.conjugate())
        energy += math.fsum(abs(v) ** 2 for v in equalized)
        for hypothesis in range(336):
            terms = bank[hypothesis ^ (repeat % 2)] * equalized
            accumulated[hypothesis] += complex(
                math.fsum(v.real for v in terms), math.fsum(v.imag for v in terms)
            )
    expected = np.abs(accumulated) ** 2 / (repeats * 62 * energy)
    np.testing.assert_allclose(
        _integrated_sss(spectra, SYNC_BINS, pss, bank), expected, rtol=2e-12, atol=1e-15
    )
    assert not np.any(_integrated_sss(spectra * 0, SYNC_BINS, pss, bank))


@pytest.mark.parametrize("origin", [0, 10**12])
@pytest.mark.parametrize("partition", [1024, 4096, 16384])
@pytest.mark.parametrize('capped',[False,True])
def test_adaptive_prefix_preserves_exact_capture_and_all_noise_state(origin, partition,capped):
    stream = StreamInfo("prefix", "test", 1_920_000.0)
    rng = np.random.default_rng(23482)
    iq = ((rng.normal(size=40003) + 1j * rng.normal(size=40003)) * 0.007071).astype(np.complex64)
    # Events both inside and across transport/numerical-preparation boundaries.
    for start, end in (
        (0, 256),
        (2816, 3050),
        (3900, 4300),
        (6800, 7143),
        (16300, 16603),
        (32000, 32512),
    ):
        iq[start:end] += 0.1
    if capped:
        iq[4096:20000] += .2+.1j
        iq[25000:39000] += .2+.1j
    observations = []
    for fast in (False, True):
        pipe = POSITIVE_SNR_IID_1920.create_pipeline(stream, 1e-4, enable_feature_analysis=False)
        pipe._fixed_quiet_batch = fast
        snapshots, captures = [], []
        try:
            for seq, start in enumerate(range(0, len(iq), partition)):
                captures.extend(
                    pipe.process_block(
                        IQBlock(stream, seq, origin + start, None, iq[start : start + partition])
                    )
                )
                noise = pipe.noise_estimator
                snapshots.append(
                    (
                        noise.noise_power_linear,
                        noise.staleness_samples,
                        noise.total_accepted_samples,
                        tuple(noise._history_powers),
                        noise._adaptation_residual.tobytes(),
                        noise._frozen,
                        pipe._noise_admission_sample,
                        pipe._pending_noise_buf.tobytes(),
                        pipe.consumed_end_sample,
                        repr(sorted(pipe._segmenter.__dict__.items())),
                    )
                )
            captures.extend(pipe.flush())
            events = []
            for capture in captures:
                event = capture.event
                assert (
                    event.iq.tobytes()
                    == iq[
                        event.event_start_sample - origin : event.event_end_sample - origin
                    ].tobytes()
                )
                events.append(
                    (
                        event.event_start_sample,
                        event.event_end_sample,
                        event.trigger_sample,
                        capture.termination_reason,
                        event.iq.tobytes(),
                    )
                )
            observations.append((snapshots, events))
        finally:
            pipe.shutdown()
    assert observations[0] == observations[1]
    assert len(observations[0][1]) >= (2 if capped else 5)


def test_native_adaptive_profile_uses_real_clock_and_distinct_identity():
    stream = StreamInfo("native", "test", 1_920_000.0)
    pipe = POSITIVE_SNR_IID_1920.create_pipeline(stream, 1e-4, enable_feature_analysis=False)
    try:
        assert pipe.noise_estimator._sample_rate_hz == stream.sample_rate_hz
        assert pipe.detection_profile_id == POSITIVE_SNR_IID_1920.profile_id
        assert pipe.detection_profile_id != POSITIVE_SNR_IID.profile_id
        assert pipe._fixed_quiet_batch
    finally:
        pipe.shutdown()


def test_native_sss_is_warmed_before_input_and_keeps_real_mib_decoding():
    from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
    from signal_processing.lte._native import _integrated_sss
    from tests.lte_pbch_fixture import waveform

    stream = StreamInfo("warm-native", "encoded", 1_920_000.0)
    iq = waveform(42, ports=2, rate=1_920_000.0, frames=5, snr_db=10.0)
    receiver = LTEBroadcastReceiver(
        stream, LTESyncConfig.native(low_snr=True), sync_backend="numba"
    )
    signatures = tuple(_integrated_sss.signatures)
    mibs = []
    for seq, start in enumerate(range(0, len(iq), 16384)):
        mibs.extend(
            receiver.process_block(
                IQBlock(stream, seq, start, None, iq[start : start + 16384])
            ).mibs
        )
    mibs.extend(receiver.flush().mibs)
    assert len(mibs) == 5 and all(mib.pci == 42 for mib in mibs)
    assert tuple(_integrated_sss.signatures) == signatures


@pytest.mark.parametrize("fail_sink", [False, True])
@pytest.mark.parametrize('priority',['normal','high'])
@pytest.mark.parametrize('dsp_scheduling',['priority','mmcss'])
@pytest.mark.parametrize('dsp_wait',['spin','block'])
def test_public_adaptive_runner_accounts_for_real_capture_delivery(tmp_path, fail_sink,priority,dsp_scheduling,dsp_wait):
    from signal_processing.runtime.adaptive_cli import run
    if priority == 'high' and sys.platform != 'win32':
        pytest.skip('Windows process priority')
    import psutil
    owner = psutil.Process()
    original_priority = owner.nice()
    original_affinity = owner.cpu_affinity()

    iq = np.full(65536, 0.003, np.complex64)
    iq[1024:1280] += 0.1
    path = tmp_path / "adaptive.cf32"
    iq.tofile(path)
    delivered = []

    def sink(capture):
        if fail_sink:
            raise RuntimeError("real sink failure")
        event = capture.event
        assert event.iq.tobytes() == iq[event.event_start_sample : event.event_end_sample].tobytes()
        delivered.append(capture)

    result = run(path, initial_noise_power=1e-4, capture_sink=sink,priority_class=priority,
                 dsp_scheduling=dsp_scheduling,dsp_wait=dsp_wait)
    assert result['dsp_scheduling'] == dsp_scheduling
    assert result['dsp_wait'] == dsp_wait
    assert result['smt_reservation']['stopped'] and result['smt_reservation']['error'] is None
    assert result['realtime_scheduling']['requested'] == (dsp_scheduling == 'mmcss')
    assert owner.nice() == original_priority and owner.cpu_affinity() == original_affinity
    assert result["runner"]["state"] == "CLOSED"
    assert result["runner"]["outstanding_deliveries"] == 0
    if fail_sink:
        assert not result["scheduled_lossless_passed"]
        assert any("real sink failure" in reason for reason in result["errors"])
        assert result["runner"]["failed_samples"] > 0
    else:
        assert result["scheduled_lossless_passed"] and delivered
        assert result["processed_samples"] == len(iq)


@pytest.mark.parametrize(
    "rate,power,size", [(2e6, 1e-4, 16384), (1.92e6, float("nan"), 16384), (1.92e6, 1e-4, 0)]
)
def test_adaptive_replay_rejects_invalid_configuration_before_input(rate, power, size):
    from signal_processing.runtime.adaptive_cli import run

    with pytest.raises(ValueError):
        run("not-opened.cf32", sample_rate_hz=rate, initial_noise_power=power, block_size=size)


def test_adaptive_pool_limit_is_checked_before_allocation():
    from signal_processing.runtime.adaptive_cli import run

    for capacity in (0, True, 100000):
        with pytest.raises(ValueError, match="pool"):
            run("not-opened.cf32", initial_noise_power=1e-4, queue_capacity=capacity)


@pytest.mark.parametrize('endian',['little','big'])
@pytest.mark.parametrize('block_size',[997,200000])
@pytest.mark.parametrize('backend',['mmap','buffered'])
def test_recording_storage_is_bounded_across_regions_reset_and_close(tmp_path,endian,block_size,backend):
    from signal_processing.sources import RecordedFileIQSource
    values = (np.arange(300003)+.5j).astype(np.complex64)
    path = tmp_path/'bounded.cf32'
    values.astype('<c8' if endian == 'little' else '>c8').tofile(path)
    stream = StreamInfo('bounded','file',1_920_000.)
    source = RecordedFileIQSource(path,stream,block_size=block_size,start_sample=10**12,
                                  endianness=endian,read_backend=backend)
    first = None
    try:
        received = 0
        while (block := source.read_block()) is not None:
            if first is None:
                first = block.iq
            assert block.start_sample == 10**12+received
            np.testing.assert_array_equal(block.iq,values[received:received+block.num_samples])
            if backend == 'mmap':
                assert source._mmap.nbytes <= max(1024*1024,block_size*8)
            else:
                assert source._mmap is None and source._buffer_bytes == max(1024*1024,block_size*8)
            received += block.num_samples
        assert received == len(values)
        np.testing.assert_array_equal(first,values[:len(first)])
        assert not first.flags.writeable
        source.reset()
        np.testing.assert_array_equal(source.read_block().iq,first)
    finally:
        source.close()
    assert source.closed and source._mmap is None and source._file.closed
    path.unlink()
