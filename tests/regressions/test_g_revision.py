"""G repair public regressions; real IQ, results, and bounded event coordination."""
import dataclasses
import json
import math
import threading
import time

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo, NoiseEstimate, DetectionMeasurement
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.detection import EnergyDetector, GaussianNoiseEstimator, EmpiricalNoiseCalibrator
from signal_processing.runtime import StreamingCapturePipeline, FeatureAnalysisWorker
from signal_processing.features import FeatureAggregator, build_default_feature_registry

S = StreamInfo('g-public', 'synthetic', 1e6)


def block(x, start=0, seq=0, timestamp=None, **kw):
    return IQBlock(S, seq, start, timestamp, np.asarray(x, np.complex64), **kw)


def pipeline(**kw):
    args = dict(stream_info=S, ring_capacity=4096, window_size=16, hop_size=8,
        start_persistence=1, end_persistence=1, pre_trigger_samples=0,
        pre_signal_margin_samples=0, post_signal_samples=0, min_post_trigger_samples=0,
        enable_feature_analysis=False,
        noise_estimator=GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4))
    args.update(kw)
    return StreamingCapturePipeline(**args)


def sidecar(name=None):
    p = pipeline()
    x = np.full(128, .003, np.complex64); x[:32] = .5
    out = p.process_block(block(x)) + p.flush()
    p.shutdown()
    sc = out[0]
    if name:
        sc = dataclasses.replace(sc, event_id=name, event=dataclasses.replace(sc.event, event_id=name))
    return sc


def aggregate(ctx):
    return FeatureAggregator(build_default_feature_registry()).extract_analysis_result(ctx)


@pytest.mark.parametrize('scale', [1e-16, 1e-8, 1e-4, 1., 1e12, 1e20])
@pytest.mark.parametrize('ratio', [0., .5, 1.1, 4.])
@pytest.mark.parametrize('floor', [1e-15, 1e-4])
def test_g01_physical_snr(scale, ratio, floor):
    mean = 1e-4*scale**2
    d = EnergyDetector(window_size=16, hop_size=8, conversion_floor_linear=floor)
    m = d.process(block(np.full(16, math.sqrt(mean*ratio))), NoiseEstimate(S.stream_id, 0, 10*math.log10(mean)))
    excess = m.signal_power_linear-m.noise_power_linear
    assert m.snr_valid == (excess > 0)
    if excess > 0:
        assert m.snr_estimate_db == pytest.approx(10*(math.log10(excess)-math.log10(mean)), abs=1e-10)
    else:
        assert m.snr_estimate_db == 0 and m.snr_invalid_reason == 'nonpositive_excess'
    assert DetectionMeasurement(**json.loads(json.dumps(dataclasses.asdict(m), allow_nan=False))) == m


def powers_oracle(x, n=16, h=8):
    return [math.fsum(float(v.real)**2+float(v.imag)**2 for v in x[i:i+n])/n for i in range(0, len(x)-n+1, h)]


def partition_spans(length, size):
    rng = np.random.default_rng(1129)
    start = seq = 0
    while start < length:
        end = min(length, start + (size if size else int(rng.integers(1, 113))))
        yield seq, start, end
        seq += 1; start = end


@pytest.mark.parametrize('size', [1, 8, 16, 64, 256, 2048, 0])
@pytest.mark.parametrize('origin', [0, 10**12])
def test_g02_g03_exact_dynamic_range(size, origin):
    rng = np.random.default_rng(2233)
    x = ((rng.normal(size=2048)+1j*rng.normal(size=2048))*1e-8).astype(np.complex64); x[0] = 1
    oracle = powers_oracle(x)
    d = EnergyDetector(window_size=16, hop_size=8)
    c = EmpiricalNoiseCalibrator(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1)
    c.begin_calibration(); ms = []
    for seq, a, b in partition_spans(len(x), size):
        ms += d.process_windows(block(x[a:b], origin+a, seq), NoiseEstimate(S.stream_id, seq, -160))
        c.update_calibration(x[a:b])
    np.testing.assert_allclose([m.signal_power_linear for m in ms], oracle, rtol=1e-10, atol=0)
    np.testing.assert_allclose(c._cal_window_powers, oracle, rtol=1e-10, atol=0)
    assert [m.start_sample for m in ms] == list(range(origin, origin+len(x)-15, 8))
    assert [m.above_on_threshold for m in ms] == [v >= ms[0].on_threshold_linear for v in oracle]
    r = c.finalize_calibration()
    assert r['empirical_t_on_linear'] == pytest.approx(np.quantile(oracle, .99), rel=1e-10, abs=0)
    assert r['empirical_t_off_linear'] == pytest.approx(np.quantile(oracle, .9), rel=1e-10, abs=0)
    assert len(d._residual_iq) < 16 and len(c._cal_residual_iq) < 16


@pytest.mark.parametrize('phase', ['analysis', 'sink', 'exception'])
def test_g04_sync_deadline_tracks_analysis_and_delivery(phase):
    entered, release = threading.Event(), threading.Event()
    outputs, callbacks = [], []
    def pause():
        entered.set(); assert release.wait(3)
    def analyse(ctx):
        if phase != 'sink': pause()
        if phase == 'exception': raise ValueError('intentional backend failure')
        return aggregate(ctx)
    def sink(r):
        if phase == 'sink': pause()
        callbacks.append((r.event_id, w.state.value))
    w = FeatureAnalysisWorker(analysis_function=analyse, result_sink=sink)
    sc = sidecar()
    t = threading.Thread(target=lambda: outputs.append(w.submit(sc)))
    t.start(); assert entered.wait(2)
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError): w.shutdown(timeout=.02)
        assert time.monotonic()-start < .5
        assert w.state.value == 'CLOSING' and w.unfinished_work == 1
        with pytest.raises(RuntimeError): w.submit(sc)
    finally:
        release.set(); t.join(2); w.shutdown(timeout=1)
    assert not t.is_alive() and w.unfinished_work == 0 and w.state.value == 'CLOSED'
    assert len(outputs) == len(callbacks) == 1
    assert outputs[0].feature_status == ('failed' if phase == 'exception' else 'ok')
    assert callbacks[0][1] == 'CLOSING'


@pytest.mark.parametrize('phase', ['analysis', 'sink'])
def test_g04_reentrant_shutdown_is_deferred(phase):
    states = []
    def close():
        w.shutdown(timeout=.01); states.append(w.state.value)
    def analyse(ctx):
        if phase == 'analysis': close()
        return aggregate(ctx)
    def sink(r):
        if phase == 'sink': close()
    w = FeatureAnalysisWorker(analysis_function=analyse, result_sink=sink)
    assert w.submit(sidecar()).feature_status == 'ok'
    assert states == ['CLOSING'] and w.state.value == 'CLOSED' and w.unfinished_work == 0


def test_g04_concurrent_admission_bounded_reset_does_not_erase_active_work():
    entered = threading.Barrier(3); release = threading.Event(); out = []
    def analyse(ctx):
        entered.wait(timeout=2); assert release.wait(3); return aggregate(ctx)
    w = FeatureAnalysisWorker(max_queue_size=2, analysis_function=analyse)
    captures = [sidecar(str(i)) for i in range(3)]
    threads = [threading.Thread(target=lambda sc=sc: out.append(w.submit(sc))) for sc in captures[:2]]
    for t in threads: t.start()
    entered.wait(timeout=2)
    try:
        with pytest.raises(RuntimeError, match='capacity'): w.submit(captures[2])
        w.reset()
        assert w.unfinished_work == 2
        with pytest.raises(TimeoutError): w.shutdown(timeout=.01)
    finally:
        release.set()
        for t in threads: t.join(2)
        w.shutdown(timeout=1)
    assert len(out) == 2 and w.get_result(block=False) is None
    assert w.unfinished_work == 0 and w.discarded_old_generation_count == 2


@pytest.mark.parametrize('origin', [0, 1, 1000, 10**12])
@pytest.mark.parametrize('trained', [0, 8])
def test_g05_frozen_preflight_is_atomic_and_retryable(origin, trained):
    est = GaussianNoiseEstimator(window_size=16, warmup_samples=16)
    p = pipeline(noise_estimator=est)
    if trained: p.process_block(block(np.full(trained, .003), origin))
    est.freeze(True)
    b = block(np.full(64, .003), origin+trained, int(bool(trained)), 1_000_000_000)
    snapshot = (p.detector._input_end, p.ring_buffer.available_range(), p.ring_buffer.stream_generation,
        p._previous_block, p._time_anchor, p.received_end_sample, p._noise_admission_sample,
        est.samples_trained, est._warmup_power_sum, p.detector._phase_accumulator)
    for _ in range(2):
        with pytest.raises(StreamConsistencyError, match='Frozen'): p.process_block(b)
        assert snapshot == (p.detector._input_end, p.ring_buffer.available_range(), p.ring_buffer.stream_generation,
            p._previous_block, p._time_anchor, p.received_end_sample, p._noise_admission_sample,
            est.samples_trained, est._warmup_power_sum, p.detector._phase_accumulator)
        assert p.state.value == 'OPEN'
    est.freeze(False)
    p.process_block(b)
    assert est.samples_trained == 16 and p._previous_block is b
    assert p.detector._input_end == b.end_sample
    p.flush(); p.shutdown()


@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('mismatch', ['identity', 'quality', 'samples', 'rate', 'transform', 'schema', 'none'])
def test_g06_backend_envelope_validation_and_recovery(async_mode, mismatch):
    from signal_processing.core import ExtractionPolicy
    first = True
    def analyse(ctx):
        nonlocal first
        r = aggregate(ctx)
        if not first: return r
        first = False
        if mismatch == 'identity': return dataclasses.replace(r, feature_vector=dataclasses.replace(r.feature_vector, event_id='different-event'))
        if mismatch == 'quality': return dataclasses.replace(r, capture_quality=dataclasses.replace(r.capture_quality, capture_epoch=999))
        if mismatch == 'schema': return dataclasses.replace(r, feature_vector=dataclasses.replace(r.feature_vector, feature_schema_version='custom-mismatch'))
        if mismatch in ('samples', 'rate', 'transform'):
            import hashlib
            d = r.policy.to_dict(); d.pop('policy_id')
            key = dict(samples='observation_samples', rate='sample_rate_hz', transform='nfft')[mismatch]
            d[key] += 1
            pid = f"policy_{d['schema_version']}_" + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
            return dataclasses.replace(r, policy=ExtractionPolicy(policy_id=pid, **d))
        return r
    callbacks = []
    w = FeatureAnalysisWorker(synchronous=not async_mode, analysis_function=analyse, result_sink=callbacks.append)
    sc = sidecar('submitted-B')
    w.submit(sc); w.submit(sidecar('subsequent-C')); w.shutdown(timeout=3)
    results = [w.get_result(block=False), w.get_result(block=False)]
    assert len(callbacks) == 2 and w.unfinished_work == 0
    assert results[0].event_id == sc.event_id
    assert results[0].feature_status == ('ok' if mismatch == 'none' else 'failed')
    if mismatch != 'none': assert results[0].features is None and results[0].analysis_result is None
    assert results[1].feature_status == 'ok' and results[1].features.event_id == 'subsequent-C'


@pytest.mark.parametrize('origin', [0, 10**12])
@pytest.mark.parametrize('warmup', [False, True])
@pytest.mark.parametrize('jump', [-1_000_000_000, -1, 0, 1, 1_000_000_000])
def test_g07_timestamp_preflight_and_original_transport(origin, warmup, jump):
    est = GaussianNoiseEstimator(window_size=16, warmup_samples=16) if warmup else GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4)
    p = pipeline(noise_estimator=est)
    first = block(np.full(64, .003), origin, timestamp=2_000_000_000)
    p.process_block(first)
    assert p._previous_block is first
    x = np.full(128, .003); x[:32] = .5
    second = block(x, origin+64, 1, 2_000_064_000+jump)
    if abs(jump) > 1:
        before = (p.consumed_end_sample, p.received_end_sample, p._time_anchor, p.detector._input_end)
        with pytest.raises(StreamConsistencyError, match='timestamp'): p.process_block(second)
        assert before == (p.consumed_end_sample, p.received_end_sample, p._time_anchor, p.detector._input_end)
        assert p.state.value == 'OPEN'
        second = dataclasses.replace(second, timestamp_ns=2_000_064_000)
    events = p.process_block(second)
    assert len(events) == 1 and events[0].event.timestamp_ns == 2_000_056_000
    p.shutdown()


@pytest.mark.parametrize('delivery', ['callbacks', 'polling'])
def test_g08_rejection_returns_before_any_sink_gate(delivery):
    entered, release_analysis, release_sink = threading.Event(), threading.Event(), threading.Event()
    returned = threading.Event(); outputs = []; callbacks = []
    def analyse(ctx):
        entered.set(); assert release_analysis.wait(3); return aggregate(ctx)
    def sink(r):
        callbacks.append((r.event_id, r.feature_status, threading.get_ident()))
        assert release_sink.wait(3)
    w = FeatureAnalysisWorker(synchronous=False, max_queue_size=1, max_results=2,
        result_delivery=delivery, analysis_function=analyse, result_sink=sink)
    assert w.submit(sidecar('one')).feature_status == 'pending'; assert entered.wait(2)
    assert w.submit(sidecar('two')).feature_status == 'pending'
    rejected = sidecar('three')
    def submit():
        outputs.append(w.submit(rejected)); returned.set()
    t = threading.Thread(target=submit); t.start()
    try:
        assert returned.wait(1), 'async acquisition blocked on rejection sink'
        assert outputs[0].feature_status == 'dropped_overload' and outputs[0].event_id == 'three'
        assert not callbacks
        for _ in range(10): assert w.submit(rejected).feature_status == 'dropped_overload'
        assert len(w.processed_results) <= 2
    finally:
        release_analysis.set(); release_sink.set(); t.join(2); w.shutdown(timeout=3)
    assert w.completed_count == 2 and w.dropped_overload_count == 11
    assert [c[0] for c in callbacks] == ['one', 'two']
    assert all(c[2] != t.ident for c in callbacks) and w.unfinished_work == 0
    if delivery == 'polling':
        assert [w.get_result(block=False).event_id for _ in range(2)] == ['one', 'two']
        assert w.get_result(block=False) is None and w._polling_reserved == 0


@pytest.mark.parametrize('size', [1, 63, 256, 0, 8192])
def test_g02_g03_filtered_and_long_stream_oracle(size):
    from scipy.signal import sosfilt
    rng = np.random.default_rng(302)
    x = (rng.normal(size=8192)+1j*rng.normal(size=8192)).astype(np.complex64)*np.float32(.01)
    x[::257] *= 1e8
    d = EnergyDetector(window_size=64, hop_size=32, mode='configured_band', monitored_band_hz=(90e3, 110e3))
    # Independent mixer/filter followed by stable direct-window physical sums.
    t = np.arange(len(x))/1e6
    mixed = x*np.exp(-2j*np.pi*100e3*t).astype(np.complex64)
    filtered = sosfilt(d._sos, mixed).astype(np.complex64)
    ms = []
    for seq, a, b in partition_spans(len(x), size):
        ms += d.process_windows(block(x[a:b], a, seq), NoiseEstimate(S.stream_id, seq, -40))
    np.testing.assert_allclose([m.signal_power_linear for m in ms], powers_oracle(filtered, 64, 32), rtol=1e-10, atol=0)
    assert len(d._residual_iq) < 64
    # The separate calibrator path is verified directly against its actual
    # prepared filtered samples by the frozen moderate-range numerical script.


@pytest.mark.parametrize('size', [8, 64])
def test_g05_filtered_frozen_retry_matches_unfrozen(size):
    def run(freeze):
        est = GaussianNoiseEstimator(window_size=16, warmup_samples=24, warmup_settling_samples=16)
        p = pipeline(noise_estimator=est, detector_mode='configured_band', monitored_band_hz=(90e3, 110e3))
        x = (np.exp(2j*np.pi*.1*np.arange(128))*.003).astype(np.complex64)
        for seq, start in enumerate(range(0, 128, size)):
            b = block(x[start:start+size], 10**12+start, seq)
            if freeze and not est.is_calibrated:
                est.freeze(True)
                before = None if p.detector._sos_zi is None else p.detector._sos_zi.copy()
                with pytest.raises(StreamConsistencyError, match='Frozen'): p.process_block(b)
                np.testing.assert_array_equal(before, p.detector._sos_zi)
                est.freeze(False)
            p.process_block(b)
        p.flush(); p.shutdown()
        return est.noise_power_linear, est.samples_trained, p.detector._input_end, p.ring_buffer.read_range(10**12, 10**12+128)
    a, b = run(False), run(True)
    assert a[:3] == b[:3]
    np.testing.assert_array_equal(a[3], b[3])


def test_g07_unknown_late_anchor_loss_and_reset():
    p = pipeline()
    p.process_block(block(np.full(64, .003), 1000))
    p.process_block(block(np.full(64, .003), 1064, 1, 1_000_064_000))
    assert p._time_anchor == (1064, 1_000_064_000)
    # Known missing extent does not excuse an unexplained receiver clock jump.
    with pytest.raises(StreamConsistencyError, match='timestamp'):
        p.process_block(block(np.full(64, .003), 1160, 2, 2_000_160_000, dropped_samples_before=32))
    p.process_block(block(np.full(64, .003), 1160, 2, 1_000_160_000, dropped_samples_before=32))
    # Unknown overflow explicitly discards the old clock relationship.
    p.process_block(block(np.full(64, .003), 1250, 3, None, overflow=True))
    assert p._time_anchor is None
    p.process_block(block(np.full(64, .003), 1314, 4, 9_000_000_000))
    assert p._time_anchor == (1314, 9_000_000_000)
    p.reset()
    p.process_block(block(np.full(64, .003), 10**12, 0, 50))
    assert p._time_anchor == (10**12, 50)
    p.shutdown()


def test_g07_rational_timestamp_rounding_does_not_accumulate():
    from signal_processing.core.timebase import sample_offset_ns
    from fractions import Fraction
    for fs in (3e6, 44_100., 1_000_000.25):
        for index in (1, -17, 10**12, 10**18):
            assert sample_offset_ns(index, fs) == round(Fraction(index*10**9, 1)/Fraction(fs))


def test_validated_ring_commit_matches_general_public_write():
    from signal_processing.buffering import CircularIQBuffer
    a, b = CircularIQBuffer(64), CircularIQBuffer(64)
    rng = np.random.default_rng(851)
    start = 10**12
    for seq in range(300):
        n = int(rng.integers(1, 150))
        iq = (rng.normal(size=n)+1j*rng.normal(size=n)).astype(np.complex64)
        raw = block(iq, start, seq)
        a.write(start, iq); b.write_received(raw)
        assert a.available_range() == b.available_range()
        np.testing.assert_array_equal(a.read_range(*a.available_range()), b.read_range(*b.available_range()))
        start += n
    with pytest.raises(TypeError): b.write_received(iq)


def test_g08_benchmark_immediate_rejections_are_accounted():
    from benchmarks.sustained_evaluation import bounded_acquisition
    r = bounded_acquisition(.6, sample_rate_hz=40_000, features=True, slow_sink_seconds=.12)
    assert r['recovered'] and r['sample_accounting_balanced']
    d = r['delivery']
    assert d['raw_captures'] == d['terminal_results']
    assert d['terminal_results'] == d['accepted_callback_results']+d['immediate_unaccepted_results']
    assert d['unresolved_identities'] == d['duplicate_results'] == 0
    if r['overload_drops']:
        assert d['immediate_unaccepted_results'] == r['overload_drops'] and not r['analysis_delivery_passed']
