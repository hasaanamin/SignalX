"""Large transport blocks preserve public behavior and exact native BCH math."""
import numpy as np
import pytest
import dataclasses
import math

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import GaussianNoiseEstimator, AdaptiveRobustNoiseEstimator
from signal_processing.runtime import StreamingCapturePipeline
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
from signal_processing.lte.bch import decode_tail_biting
from tests.lte_pbch_fixture import waveform


@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('origin', [0, 10**12])
@pytest.mark.parametrize('n,h', [(17, 8), (64, 32)])
def test_large_block_chunking_matches_reference_exact_captures_and_noise(adaptive, origin, n, h):
    stream = StreamInfo('large-block', 'test', 2e6)
    rng = np.random.default_rng(2233)
    x = ((rng.normal(size=20029)+1j*rng.normal(size=20029))*.007).astype(np.complex64)
    for a, b in ((3111, 4777), (8101, 10199), (15000, 19000)):
        x[a:b] += .1
    outcomes = []
    for fast in (False, True):
        noise = (AdaptiveRobustNoiseEstimator(window_size=n, initial_noise_power_linear=1e-4,
                     time_constant_s=.01, adaptation_window_size=127) if adaptive else
                 GaussianNoiseEstimator(window_size=n, calibrated_noise_power_linear=1e-4))
        pipe = StreamingCapturePipeline(stream, window_size=n, hop_size=h, ring_capacity=4096,
            noise_estimator=noise, pre_trigger_samples=128, pre_signal_margin_samples=0,
            max_event_samples=4096, enable_feature_analysis=False, fixed_quiet_batch=fast)
        events = []
        preparation_sizes = []
        prepare = pipe.detector._prepare_full_band
        def observed(block):
            preparation_sizes.append(block.num_samples)
            return prepare(block)
        pipe.detector._prepare_full_band = observed
        try:
            events += pipe.process_block(IQBlock(stream, 0, origin, None, x[:16384]))
            events += pipe.process_block(IQBlock(stream, 1, origin+16384, None, x[16384:]))
            events += pipe.flush()
            summary = []
            for sc in events:
                event = sc.event
                np.testing.assert_array_equal(event.iq,
                    x[event.event_start_sample-origin:event.event_end_sample-origin])
                summary.append((event.event_start_sample, event.event_end_sample,
                                event.detected_start_sample, event.detected_end_sample, event.iq.tobytes()))
            estimate = noise.get_estimate(stream.stream_id, 1)
            outcomes.append((summary, estimate.noise_power_dbfs, pipe._noise_admission_sample))
            if fast:
                assert len(preparation_sizes) >= 4 and max(preparation_sizes) <= 4096
            else:
                assert not preparation_sizes
        finally:
            pipe.shutdown()
    assert outcomes[0] == outcomes[1] and outcomes[0][0]


@pytest.mark.parametrize('threads', [1, 2, 4])
def test_native_tail_biting_exact_reference_random_and_ties(threads):
    from signal_processing.lte._native import diagnostics
    info = diagnostics('numba', threads)
    assert info['active_kernel_threads'] == threads
    rng = np.random.default_rng(3199)
    for k in range(45):
        values = (rng.integers(-2, 3, 120).astype(float) if k % 3 else rng.normal(size=120))
        values *= 10.**(k % 25-16)
        expected = decode_tail_biting(values, backend='reference')
        actual = decode_tail_biting(values, backend='numba', threads=threads)
        np.testing.assert_array_equal(actual, expected)
    assert decode_tail_biting(np.zeros(120), backend='numba', threads=threads) is None


@pytest.mark.parametrize('threads', [True, 1.5, 0, 5])
def test_native_configuration_strict_even_after_cached_valid_initialization(threads):
    from signal_processing.lte._native import initialize
    initialize('numba', 1)
    with pytest.raises(ValueError):
        initialize('numba', threads)
    with pytest.raises(ValueError):
        decode_tail_biting(np.zeros(120), threads=threads)


def test_real_iq_native_and_reference_receiver_results_match_exactly():
    stream = StreamInfo('native', 'independent', 2e6, 2.14e9)
    x = waveform(42, ports=2, snr_db=-5, cfo=12345, frames=4)
    outputs = []
    for backend, threads in (('reference', 1), ('numba', 1), ('numba', 2)):
        receiver = LTEBroadcastReceiver(stream, LTESyncConfig.low_snr(),
                                       bch_backend=backend, numba_threads=threads)
        info = receiver.execution
        info['backend'] = 'invented'
        assert receiver.execution['backend'] == backend
        found = []
        for seq, a in enumerate(range(0, len(x), 16384)):
            found.extend(receiver.process_block(IQBlock(stream, seq, 10**12+a, None, x[a:a+16384])).mibs)
        found.extend(receiver.flush().mibs)
        outputs.append(found)
    assert outputs[0] == outputs[1] == outputs[2] and len(outputs[0]) == 4


def test_later_chunk_failure_preserves_committed_outputs_and_horizons():
    from signal_processing.detection.exceptions import PolicyCompatibilityError
    stream = StreamInfo('chunk-fault', 'test', 2e6)
    x = np.full(16384, .003, np.complex64)
    x[512:1700] = .2
    x[5000:6300] = .2
    outcomes = []
    for fast in (False, True):
        pipe = StreamingCapturePipeline(stream, window_size=64, hop_size=32, ring_capacity=4096,
            noise_estimator=GaussianNoiseEstimator(window_size=64, calibrated_noise_power_linear=1e-4),
            pre_trigger_samples=128, pre_signal_margin_samples=0, max_event_samples=4096,
            enable_feature_analysis=False, fixed_quiet_batch=fast)
        method = '_commit_full_band' if fast else 'process_windows'
        original = getattr(pipe.detector, method)
        def reject(*args, **kwargs):
            block = args[1] if fast else args[0]
            if block.start_sample >= 8192:
                raise PolicyCompatibilityError('Injected failure after two committed chunks')
            return original(*args, **kwargs)
        setattr(pipe.detector, method, reject)
        try:
            with pytest.raises(PolicyCompatibilityError) as caught:
                pipe.process_block(IQBlock(stream, 0, 0, None, x))
            completed = caught.value.completed_captures
            assert len(completed) == 2
            assert pipe.state.value == 'FAULTED'
            assert pipe.consumed_end_sample == 8192 and pipe.received_end_sample == 16384
            summaries = []
            for sc in completed:
                event = sc.event
                np.testing.assert_array_equal(event.iq, x[event.event_start_sample:event.event_end_sample])
                summaries.append((event.event_start_sample, event.event_end_sample, event.iq.tobytes()))
            outcomes.append(summaries)
            assert pipe.flush() == pipe.flush() == []
        finally:
            pipe.shutdown()
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize('repeats', [1, 4, 8])
def test_bounded_blind_search_matches_full_fft_and_direct_received_windows(repeats):
    stream = StreamInfo('scores', 'encoded-reference', 2e6, 2.14e9)
    config = dataclasses.replace(LTESyncConfig.low_snr(), integration_half_frames=repeats, tracking=False)
    receiver = LTEBroadcastReceiver(stream, config)
    x = waveform(42, ports=2, snr_db=-5, cfo=12345, frames=8)
    original = receiver._bounded_scores
    compared = []
    def checked(iq, start, stop):
        actual = original(iq, start, stop)
        reference = receiver._reference_scores(iq, start, stop)
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)
        for offset in (0, (stop-start)//2, stop-start-1):
            powers = np.zeros(len(receiver._templates))
            energy = 0.
            for repeat in range(repeats):
                index = start+offset-receiver._base-repeat*9600
                window = iq[index:index+128]
                energy += math.fsum(float(z.real)**2+float(z.imag)**2 for z in window)
                powers += np.abs(np.array([np.vdot(template, window) for template in receiver._templates]))**2
            expected = powers/(62*energy) if energy else powers*0
            np.testing.assert_allclose(actual[:, offset], expected, rtol=1e-12, atol=1e-14)
        compared.append(stop-start)
        return actual
    receiver._bounded_scores = checked
    for seq, a in enumerate(range(0, len(x), 16384)):
        receiver.process_block(IQBlock(stream, seq, 10**12+a, None, x[a:a+16384]))
    receiver.flush()
    assert compared and max(compared) <= config.search_chunk_samples


def test_optional_native_absence_uses_reference_and_explicit_request_fails(monkeypatch):
    from signal_processing.lte import _native
    expected = decode_tail_biting(np.arange(120, dtype=float), backend='reference')
    _native._initialize_cached.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(_native, 'numba', None)
            assert _native.initialize('auto') == 'reference'
            np.testing.assert_array_equal(decode_tail_biting(np.arange(120, dtype=float)), expected)
            with pytest.raises(ImportError):
                _native.initialize('numba')
    finally:
        _native._initialize_cached.cache_clear()


def test_transport_policy_cache_uses_all_content_and_not_event_values():
    from signal_processing.runtime.process_analysis import _result, _result_record, _policy_from_json
    from signal_processing.runtime import FeatureAnalysisWorker
    from tests.regressions.test_process_analysis import capture
    worker = FeatureAnalysisWorker()
    try:
        sc = capture()
        original = worker.submit(sc).analysis_result
        record = _result_record(original)
        first, second = _result(record), _result(record)
        assert first.policy is second.policy  # Only immutable configuration is shared.
        assert first.feature_vector is not second.feature_vector
        assert first.feature_vector.features is not second.feature_vector.features
        changed = list(record)
        changed[3] = dict(record[3])
        changed[3]['bandwidth_auto_noise_estimation'] = not record[3]['bandwidth_auto_noise_estimation']
        with pytest.raises(ValueError):
            _result(tuple(changed))
        assert _policy_from_json.cache_info().currsize <= 32
    finally:
        worker.shutdown()


def test_native_concurrent_callers_keep_private_state_and_restore_masks():
    import concurrent.futures
    import threading
    import numba
    from signal_processing.lte._native import initialize
    initialize('numba', 2)
    rng = np.random.default_rng(772)
    values = [rng.normal(size=120), rng.normal(size=120)]
    expected = [decode_tail_biting(x, backend='reference') for x in values]
    barrier = threading.Barrier(2, timeout=3)
    main_mask = numba.get_num_threads()
    def calculate(index):
        previous = numba.get_num_threads()
        barrier.wait()
        for _ in range(12):
            np.testing.assert_array_equal(decode_tail_biting(values[index], backend='numba', threads=2), expected[index])
        assert numba.get_num_threads() == previous
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(calculate, range(2)))
    assert numba.get_num_threads() == main_mask
