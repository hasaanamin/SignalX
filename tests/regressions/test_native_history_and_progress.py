"""Owned LTE history and actual spawned capture protocol equivalence."""
import multiprocessing

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo, StreamConsistencyError
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
from signal_processing.lte._history import IQHistory
from signal_processing.runtime.capture_stream import CaptureStreamProcess
from signal_processing.runtime.shared_transport import SharedIQTransport
from tests.lte_pbch_fixture import waveform, fixture


def test_exact_closed_path_bound_matches_all_64_state_reference():
    from signal_processing.lte.bch import decode_tail_biting, _trellis
    from signal_processing.lte._native import _branches, _closed_candidates
    rng = np.random.default_rng(6619)
    widths = set()
    for k in range(240):
        soft = rng.normal(size=120) if k % 3 else rng.integers(-2, 3, 120).astype(float)
        soft *= (1e-200, 1., 1e200)[k % 3]
        expected = decode_tail_biting(soft, backend='reference')
        actual = decode_tail_biting(soft, backend='numba', threads=1)
        assert np.array_equal(actual, expected)
        normalized = (soft/np.max(np.abs(soft))).reshape(40, 3)
        pred, signs = _trellis()
        widths.add(len(_closed_candidates(_branches(normalized, signs), pred)))
    # Strong/noiseless and tie-heavy observations exercise both reduced and
    # full-reference fallback paths, without assuming a particular RNG score.
    for soft in (np.ones(120), np.tile([1., 0., 0.], 40), np.zeros(120)):
        assert np.array_equal(decode_tail_biting(soft, backend='numba'),
                              decode_tail_biting(soft, backend='reference'))
    assert min(widths) <= 32


@pytest.mark.parametrize('capacity', [1, 17, 64, 1001])
def test_mirrored_history_matches_direct_bytes_over_many_wraps(capacity):
    rng = np.random.default_rng(5517)
    history = IQHistory(capacity)
    storage = history._storage
    oracle = np.empty(0, np.complex128)
    for _ in range(400):
        discard = int(rng.integers(len(oracle)+1))
        history.discard_before(history.start+discard)
        oracle = oracle[discard:]
        n = int(rng.integers(capacity-len(oracle)+1))
        x = rng.normal(size=n)+1j*rng.normal(size=n)
        history.append(x)
        oracle = np.concatenate((oracle, x))
        result = history.view()
        assert result.tobytes() == oracle.tobytes()
        assert result.flags.c_contiguous and not result.flags.writeable
        assert history._storage is storage and len(storage) == 2*capacity
        before = history.start, history.end, result.tobytes()
        with pytest.raises(ValueError):
            history.append(np.ones(capacity+1, complex))
        assert before == (history.start, history.end, history.view().tobytes())
    history.reset()
    assert history._storage is storage and len(history.view()) == 0


@pytest.mark.parametrize('low', [False, True])
def test_native_profile_real_iq_never_calls_resampling(monkeypatch, low):
    import signal_processing.lte.channel as channel
    iq = waveform(42, ports=2, rate=1_920_000., frames=8, cfo=12345, snr_db=10)
    def forbidden(*args, **kwargs):
        raise AssertionError('Native 1.92 MS/s must not resample')
    monkeypatch.setattr(channel, 'upfirdn', forbidden)
    stream = StreamInfo('native', 'independent-encoded', 1_920_000., 2.14e9)
    receiver = LTEBroadcastReceiver(stream, LTESyncConfig.native(low_snr=low))
    mibs = []
    for seq, start in enumerate(range(0, len(iq), 16381)):
        mibs.extend(receiver.process_block(IQBlock(stream, seq, 10**12+start, None, iq[start:start+16381])).mibs)
    mibs.extend(receiver.flush().mibs)
    assert len(mibs) == 8
    for m in mibs:
        frame = round(float(m.input_frame_start-10**12-431)/19200)
        # This fixture advances its real SFN at each BCH block.
        from tests.lte_pbch_fixture import frame_fixture
        assert m.mib.payload_bits == tuple(map(int, frame_fixture(4, 2, frame)[0]))
        assert abs(float(m.input_frame_start-10**12-431-frame*19200)) <= 2
    state = receiver.snapshot()
    assert state['retained_resampler_samples'] == 0
    before = state.copy()
    receiver.reset()
    first = IQBlock(stream, 0, 1000, None, iq[:137])
    receiver.process_block(first)
    state = receiver.snapshot()
    wrong = StreamInfo('native', 'independent-encoded', 2_000_000., 2.14e9)
    with pytest.raises(StreamConsistencyError):
        receiver.process_block(IQBlock(wrong, 1, 1137, None, iq[137:1000]))
    assert receiver.snapshot() == state
    assert before['retained_broadcast_samples'] <= (62400 if low else 24000)


def test_frequency_kernel_cache_is_bounded_config_only_and_exact():
    receiver = LTEBroadcastReceiver(StreamInfo('kernels', 'test', 1_920_000.), LTESyncConfig.native(low_snr=True))
    t = np.arange(128)/1_920_000.
    for center in np.linspace(-30000, 30000, 100):
        for fine in (False, True):
            grid, matrix = receiver._frequency_templates(float(center), fine)
            width, count = (500, 11) if fine else (7500, 31)
            expected = np.linspace(max(-30000, center-width), min(30000, center+width), count)
            assert np.array_equal(grid, expected)
            assert np.array_equal(matrix, np.exp(-2j*np.pi*expected[:, None]*t))
            assert not grid.flags.writeable and not matrix.flags.writeable
            assert len(receiver._frequency_cache) <= 32
    assert LTESyncConfig.native().profile_id != LTESyncConfig().profile_id
    with pytest.raises(ValueError):
        LTESyncConfig.native(low_snr=1)


def capture_rows(protocol, adaptive, completion_backend='pipe'):
    stream = StreamInfo('progress', 'actual-spawn', 1e6)
    source = SharedIQTransport(multiprocessing.get_context('spawn'), 16, 64)
    options = dict(window_size=17, hop_size=8, ring_capacity=4096, start_persistence=1,
        end_persistence=1, pre_trigger_samples=32, pre_signal_margin_samples=0,
        post_signal_samples=0, min_post_trigger_samples=0, max_event_samples=1024,
        fixed_quiet_batch=True)
    noise = (dict(window_size=17, initial_noise_power_linear=1e-4, adaptation_window_size=32)
             if adaptive else dict(window_size=17, calibrated_noise_power_linear=1e-4))
    pipe = CaptureStreamProcess(source, stream, noise_mode='adaptive' if adaptive else 'fixed',
        pipeline_options=options, noise_options=noise, progress_protocol=protocol, completion_capacity=2,
        completion_backend=completion_backend)
    x = np.full(256, .003, np.complex64)
    x[80:113] = .3
    for k in range(4):
        source.put((0., k, 10**12+k*64+(32 if k >= 2 else 0), x[k*64:k*64+64].tobytes(), 32 if k == 2 else 0))
    source.put(None)
    rows, captures = [], []
    try:
        while not pipe.done:
            record = pipe.get(10)
            try:
                assert record['error'] is None
                rows.append((record['sequence'], record['start_sample'], record['num_samples'], record['snapshot']))
                for sc in record['captures']:
                    e = sc.event
                    captures.append((e.event_id, e.event_start_sample, e.event_end_sample,
                        e.iq.tobytes(), sc.stream_epoch, sc.known_lost_samples, sc.termination_reason))
                    local = e.event_start_sample-10**12
                    assert e.iq.tobytes() == x[local:local+e.num_samples].tobytes()
            finally:
                pipe.task_done()
        assert pipe.drained and pipe.unfinished_completions == 0
        assert source.unfinished_tasks == 0
    finally:
        pipe.close()
        source.close()
    assert captures and rows[-1][0] == -1
    return rows, captures


@pytest.mark.parametrize('adaptive', [False, True])
def test_compact_and_pickle_real_capture_progress_match(adaptive):
    assert capture_rows('compact', adaptive) == capture_rows('pickle', adaptive)


def test_compact_falls_back_without_truncating_legal_large_python_horizon():
    stream = StreamInfo('large-progress', 'spawn', 1e6)
    source = SharedIQTransport(multiprocessing.get_context('spawn'), 2, 64)
    pipe = CaptureStreamProcess(source, stream, noise_options=dict(window_size=16, calibrated_noise_power_linear=1e-4),
        pipeline_options=dict(window_size=16, hop_size=8, ring_capacity=4096))
    try:
        start = 2**63-32
        source.put((0., 0, start, np.full(64, .003, np.complex64).tobytes(), 0))
        source.put(None)
        result = pipe.get(10)
        assert result['error'] is None and result['num_samples'] == 64
        assert result['snapshot']['received_end_sample'] == start+64
        pipe.task_done()
        pipe.get(10)
        pipe.task_done()
        assert pipe.drained
    finally:
        pipe.close()
        source.close()
