"""Public live F-series reproductions. No saved observations are used as answers."""
import dataclasses
import math
import threading
import json

import numpy as np
import pytest

from signal_processing.core import IQBlock, NoiseEstimate, StreamInfo
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.detection import EnergyDetector, GaussianNoiseEstimator, AdaptiveRobustNoiseEstimator, EmpiricalNoiseCalibrator
from signal_processing.detection.exceptions import PolicyCompatibilityError
from signal_processing.runtime import StreamingCapturePipeline

S = StreamInfo('f-series', 'synthetic', 1e6)

def block(x, start=0, seq=0, **kw):
    return IQBlock(S, seq, start, None, np.asarray(x, np.complex64), **kw)

def pipe(**kw):
    args = dict(stream_info=S, window_size=16, hop_size=8, ring_capacity=4096,
                start_persistence=1, end_persistence=1, pre_trigger_samples=0,
                pre_signal_margin_samples=0, post_signal_samples=0, min_post_trigger_samples=0,
                skip_incomplete_quality=False,
                noise_estimator=GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4))
    args.update(kw)
    return StreamingCapturePipeline(**args)

def policy(mode='empirical_matched'):
    r = np.random.default_rng(117)
    x = ((r.normal(size=20000)+1j*r.normal(size=20000))*.007).astype(np.complex64)
    q = EmpiricalNoiseCalibrator(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1).create_policy(x, settling_samples=0)
    return dataclasses.replace(q, mode=mode)

def supported(mode='empirical_matched'):
    q = policy(mode)
    # Genuine independent Gaussian window powers, conservative fixed thresholds.
    q = dataclasses.replace(q, t_on_linear=4*q.mean_power_linear, t_off_linear=3*q.mean_power_linear,
                            t_on_dbfs=10*math.log10(4*q.mean_power_linear), t_off_dbfs=10*math.log10(3*q.mean_power_linear))
    powers = np.random.default_rng(852).gamma(16, q.mean_power_linear/16, 10000)
    return q.validate_held_out(powers, reference_domain=q.reference_domain,
            independence_assumption='independent_held_out_windows', evidence_id='seed852-independent-Gamma16'), powers

@pytest.mark.parametrize('overflow', [False, True])
def test_f01_running_gap_job_survives(overflow):
    p = pipe(synchronous_analysis=False)
    entered, release = threading.Event(), threading.Event()
    real = p.worker._process_one
    def paused(sc):
        entered.set()
        assert release.wait(5)
        return real(sc)
    p.worker._process_one = paused
    submit = p.worker.submit
    def wait_started(sc):
        result = submit(sc)
        assert entered.wait(3)
        return result
    p.worker.submit = wait_started
    try:
        p.process_block(block(np.full(64, .5)))
        raw = p.process_block(block(np.zeros(32), 96, 1, dropped_samples_before=32, overflow=overflow))
        assert len(raw) == 1 and entered.wait(3)
        release.set()
        p.shutdown(timeout=5)
        result = p.worker.get_result(block=False)
        assert result is not None, 'Ordinary gap silently discarded accepted analysis'
        assert result.feature_status == 'degraded' and result.event.event_id == raw[0].event.event_id
        assert result.stream_epoch == raw[0].stream_epoch and result.unknown_loss == overflow
        np.testing.assert_array_equal(result.event.iq, np.full(64, .5, np.complex64))
    finally:
        release.set()
        p.shutdown(timeout=5)

@pytest.mark.parametrize('features,synchronous', [(False, True), (True, True), (True, False)])
def test_f02_fault_retains_completed_capture(features, synchronous):
    q = policy()
    e = AdaptiveRobustNoiseEstimator(window_size=16, initial_noise_power_linear=q.mean_power_linear,
                                     adaptation_window_size=16, time_constant_s=1e-6)
    p = pipe(noise_estimator=e, threshold_policy=q, pfa_on=.01, pfa_off=.1,
             enable_feature_analysis=features, synchronous_analysis=synchronous)
    x = np.full(128, .003, np.complex64); x[:32] = .5
    try:
        with pytest.raises(PolicyCompatibilityError) as err:
            p.process_block(block(x))
        assert p.state.value == 'FAULTED' and p.consumed_end_sample == 64 and p.received_end_sample == 128
        recovered = getattr(err.value, 'completed_captures', ())
        assert len(recovered) == 1, 'Committed [0,48) capture not recoverable from failure'
        np.testing.assert_array_equal(recovered[0].event.iq, x[:48])
        assert not p.flush() and not p.flush()
    finally:
        p.shutdown(timeout=5)

def test_f03_evidence_rejects_changed_thresholds():
    q, powers = supported()
    assert q.supported_pfa and q.validation_exceedances_on == 0
    assert np.count_nonzero(powers >= q.mean_power_linear*.5) > 9800
    with pytest.raises(ValueError):
        dataclasses.replace(q, t_on_linear=q.mean_power_linear*.5, t_off_linear=q.mean_power_linear*.4,
                            t_on_dbfs=10*math.log10(q.mean_power_linear*.5), t_off_dbfs=10*math.log10(q.mean_power_linear*.4))

@pytest.mark.parametrize('mode', ['analytic_gamma', 'empirical_matched'])
def test_f04_strict_applies_validated_thresholds(mode):
    q, _ = supported(mode)
    d = EnergyDetector(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1, threshold_policy=q, require_supported_pfa=True)
    ref = NoiseEstimate(S.stream_id, 0, 10*math.log10(q.mean_power_linear), reference_domain=q.reference_domain)
    m = d.process(block(np.ones(16)*.003), ref)
    assert 10**(m.on_threshold_dbfs/10) == pytest.approx(q.t_on_linear, rel=1e-12)
    from signal_processing.core import DetectionMeasurement
    assert DetectionMeasurement(**dataclasses.asdict(m)) == m
    d.reset()
    with pytest.raises(PolicyCompatibilityError):
        d.process(block(np.ones(16)*.003), dataclasses.replace(ref, noise_power_dbfs=ref.noise_power_dbfs-10))

def test_f05_direct_gap_is_rejected_before_mutation():
    d = EnergyDetector(window_size=16, hop_size=8)
    ref = NoiseEstimate(S.stream_id, 0, -40)
    assert not d.process_windows(block(np.ones(8)), ref)
    before = d._residual_iq.copy()
    with pytest.raises(StreamConsistencyError):
        d.process_windows(block(np.ones(8), 100, 1, dropped_samples_before=92), ref)
    np.testing.assert_array_equal(before, d._residual_iq)
    d.reset()
    assert len(d.process_windows(block(np.ones(16), 100, 1), ref)) == 1

@pytest.mark.parametrize('scale', [1., 1e-8, 1e-15, 1e8])
def test_f07_physical_scale_invariance(scale):
    e = GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4*scale**2)
    ref = e.get_estimate(S.stream_id, 0)
    assert 10**(ref.noise_power_dbfs/10) == pytest.approx(1e-4*scale**2, rel=1e-12, abs=0)
    d = EnergyDetector(window_size=16, hop_size=8)
    m = d.process(block(np.full(16, .02*scale)), ref)
    assert m.above_on_threshold and not m.below_off_threshold
    assert 10**((m.on_threshold_dbfs-ref.noise_power_dbfs)/10) == pytest.approx(2.205, rel=.001)

def test_f06_required_sink_failure_fails_acceptance(monkeypatch):
    from benchmarks import sustained_evaluation as b
    real = b._pipeline
    def build(*args, **kwargs):
        p = real(*args, **kwargs)
        def fail(_):
            raise RuntimeError('required sink unavailable')
        p.worker.add_result_sink(fail)
        return p
    monkeypatch.setattr(b, '_pipeline', build)
    r = b.bounded_acquisition(1.0, sample_rate_hz=10000, block_size=1024, features=True)
    assert r['worker_sink_errors'] > 0
    assert r['recovered'] and not r['passed']

@pytest.mark.parametrize('overflow', [False, True])
def test_f01_complete_and_queued_jobs_survive_multiple_gaps(overflow):
    p = pipe(synchronous_analysis=False)
    entered, release = threading.Event(), threading.Event()
    real = p.worker._process_one
    def pause(sc):
        entered.set(); assert release.wait(5)
        return real(sc)
    p.worker._process_one = pause
    x = np.full(128, .003, np.complex64); x[:32] = .5
    try:
        raw = p.process_block(block(x))
        assert entered.wait(3)
        raw += p.process_block(block(x, 160, 1, dropped_samples_before=32, overflow=overflow))
        raw += p.process_block(block(x, 320, 2, dropped_samples_before=32, overflow=overflow))
        assert len(raw) == 3 and p.worker._queue.qsize() == 2
        release.set(); p.shutdown()
        results = []
        while (r := p.worker.get_result(block=False)) is not None:
            results.append(r)
        assert [(r.stream_epoch, r.event_id) for r in results] == [(r.stream_epoch, r.event_id) for r in raw]
        assert all(r.feature_status == 'ok' for r in results)
        assert all(len(r.features.features) == 56 for r in results)
        assert p.worker.discarded_old_generation_count == 0
        for r, capture in zip(results, raw):
            np.testing.assert_array_equal(r.event.iq, capture.event.iq)
    finally:
        release.set(); p.shutdown()

@pytest.mark.parametrize('count', [0, 1, 3])
def test_f02_fault_after_several_completions_and_bounded_ownership(count):
    # The public worker shutdown makes the NEXT dispatch fail genuinely.
    p = pipe()
    delivered = []
    def sink(sc):
        delivered.append(sc)
        if len(delivered) == count:
            p.worker.shutdown()
    p.worker.add_result_sink(sink)
    if count == 0:
        p.worker.shutdown()
    burst = np.full(128, .003, np.complex64); burst[:32] = .5
    x = np.tile(burst, count+1)
    try:
        with pytest.raises(RuntimeError) as err:
            p.process_block(block(x))
        outputs = err.value.completed_captures
        assert len(outputs) == count+1 and outputs[-1].feature_status == 'failed'
        assert len({s.event_id for s in outputs}) == count+1
        assert p._call_outputs is None and p.state.value == 'FAULTED'
        assert not p.flush() and not p.flush()
        for sc in outputs:
            np.testing.assert_array_equal(sc.event.iq, x[sc.event.event_start_sample:sc.event.event_end_sample])
        p.reset()
        assert err.value.completed_captures == outputs  # Exception owns immutable snapshots, reset cannot erase them.
    finally:
        p.shutdown()

def test_f02_output_limit_is_explicit_and_fault_reason_is_not_hardware_loss():
    p = pipe(enable_feature_analysis=False, max_completed_per_call=2)
    burst = np.full(128, .003, np.complex64); burst[:32] = .5
    with pytest.raises(RuntimeError, match='Completed-capture limit') as err:
        p.process_block(block(np.tile(burst, 10)))
    assert len(err.value.completed_captures) == 2 and p._call_outputs is None
    assert p.consumed_end_sample < p.received_end_sample
    assert not p.flush()
    p.shutdown()
    q = policy(); e = AdaptiveRobustNoiseEstimator(window_size=16, initial_noise_power_linear=q.mean_power_linear,
                                                  adaptation_window_size=16, time_constant_s=1e-6)
    p = pipe(noise_estimator=e, threshold_policy=q, pfa_on=.01, pfa_off=.1)
    p.process_block(block(np.full(32, .5)))
    p.worker.shutdown()
    # A failed next input prerequisite while an event remains active.
    e.recalibrate(reference_linear=q.mean_power_linear*2)
    with pytest.raises(PolicyCompatibilityError):
        p.process_block(block(np.full(32, .5), 32, 1))
    with pytest.raises(RuntimeError) as err:
        p.flush()
    sc = err.value.completed_captures[0]
    assert sc.termination_reason == 'processing_failure' and not sc.unknown_loss and sc.known_lost_samples == 0
    assert not p.flush()

@pytest.mark.parametrize('edit', [dict(window_size=32), dict(hop_size=4), dict(settling_samples=1),
    dict(mean_power_linear=1e-5), dict(validation_windows=True), dict(validation_windows=10000.5),
    dict(validation_exceedances_on=1, validation_exceedances_off=0), dict(pfa_upper_on=float('nan')),
    dict(confidence_level=True), dict(evidence_id=123), dict(evidence_id=' '), dict(reference_domain={}),
    dict(statistic_version='unknown'), dict(policy_version='5.0'), dict(evidence_id='relabelled-acquisition')])
def test_f03_metadata_and_stale_evidence_rejected(edit):
    q, _ = supported()
    with pytest.raises(ValueError):
        dataclasses.replace(q, **edit)

def test_f03_roundtrip_clear_and_genuine_revalidation():
    from signal_processing.detection.noise_estimation import ThresholdPolicy
    q, powers = supported()
    assert ThresholdPolicy.from_dict(json.loads(json.dumps(q.to_dict()))) == q
    record = q.to_dict(); record['hop_size'] = 4
    with pytest.raises(ValueError):
        ThresholdPolicy.from_dict(record)
    fresh = dataclasses.replace(q.without_validation(), hop_size=4)
    assert not fresh.supported_pfa and fresh.validation_configuration_hash is None
    replay = fresh.validate_held_out(powers, reference_domain=fresh.reference_domain,
                 independence_assumption='independent_held_out_windows', evidence_id='new-applicable-same-statistic-H4')
    assert replay.supported_pfa and replay.validation_configuration_hash != q.validation_configuration_hash
    assert replay.pfa_upper_on == pytest.approx(1-.025**(1/10000), rel=1e-11)

@pytest.mark.parametrize('mode', ['analytic_gamma', 'empirical_matched'])
@pytest.mark.parametrize('scale', [.1, 1, 10])
@pytest.mark.parametrize('streaming', [False, True])
def test_f04_strict_replay_and_scaling_are_atomic(mode, scale, streaming):
    q, _ = supported(mode)
    d = EnergyDetector(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1, threshold_policy=q, require_supported_pfa=True)
    ref = NoiseEstimate(S.stream_id, 0, 10*math.log10(q.mean_power_linear*scale), reference_domain=q.reference_domain)
    f = d.process_windows if streaming else d.process
    if scale != 1:
        with pytest.raises(PolicyCompatibilityError):
            f(block(np.full(16, .003)), ref)
        assert d._input_end is None and len(d._residual_iq) == 0
    else:
        result = f(block(np.full(16, .003)), ref)
        m = result[0] if streaming else result
        assert m.on_threshold_linear == q.t_on_linear and m.off_threshold_linear == q.t_off_linear

def test_f04_requires_policy_and_actual_n():
    with pytest.raises(PolicyCompatibilityError):
        EnergyDetector(require_supported_pfa=True)
    q, _ = supported('analytic_gamma')
    d = EnergyDetector(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1, threshold_policy=q, require_supported_pfa=True)
    ref = NoiseEstimate(S.stream_id, 0, 10*math.log10(q.mean_power_linear), reference_domain=q.reference_domain)
    with pytest.raises(PolicyCompatibilityError):
        d.process(block([.003]), ref)
    assert d._input_end is None

@pytest.mark.parametrize('origin', [0, 1, 1000, 10**12])
@pytest.mark.parametrize('configured', [False, True])
def test_f05_partition_continuity_warmup_and_mixed_api(origin, configured):
    kw = dict(window_size=17, hop_size=8)
    if configured:
        kw.update(mode='configured_band', monitored_band_hz=(90000, 110000))
    x = (np.exp(.63j*np.arange(128))*.02).astype(np.complex64)
    ref = NoiseEstimate(S.stream_id, 0, -60)
    baseline = EnergyDetector(**kw).process_windows(block(x, origin), ref)
    d = EnergyDetector(**kw); results = []
    start = 0
    for n in (1, 3, 7, 17, 29, 71):
        results += d.process_windows(block(x[start:start+n], origin+start, start), ref)
        start += n
    assert [(m.start_sample, m.end_sample) for m in results] == [(m.start_sample, m.end_sample) for m in baseline]
    np.testing.assert_allclose([m.signal_power_linear for m in results], [m.signal_power_linear for m in baseline], rtol=1e-11)
    with pytest.raises(StreamConsistencyError):
        d.process(block(x[:17], origin+128, 128), ref)
    d.reset()
    d.prepare_detection_iq(block(x[:16], origin))
    assert d.process_windows(block(x[16:], origin+16, 1), ref)
    with pytest.raises(ValueError, match='must not be empty'):
        block([], origin+128, 1)  # Public IQBlock rejects empty input before detector state is reachable.
    assert d._input_end == origin+128

@pytest.mark.parametrize('start,flags', [(7, {}), (0, {}), (100, {}), (8, dict(overflow=True)),
                                      (100, dict(dropped_samples_before=92))])
def test_f05_rejection_does_not_mutate_filter(start, flags):
    d = EnergyDetector(window_size=16, hop_size=8, mode='configured_band', monitored_band_hz=(90000, 110000))
    ref = NoiseEstimate(S.stream_id, 0, -40)
    d.process_windows(block(np.ones(8)), ref)
    before = d._sos_zi.copy(), d._phase_accumulator, d._input_end
    with pytest.raises(StreamConsistencyError):
        d.process_windows(block(np.ones(8), start, 1, **flags), ref)
    np.testing.assert_array_equal(before[0], d._sos_zi)
    assert before[1:] == (d._phase_accumulator, d._input_end)

def test_f06_public_analysis_failure_and_success_paths():
    from benchmarks.sustained_evaluation import bounded_acquisition
    def fail(context):
        raise RuntimeError('intentional analysis backend failure')
    failed = bounded_acquisition(.6, sample_rate_hz=10000, block_size=1024, features=True, analysis_function=fail)
    assert failed['worker_failed'] > 0 and not failed['analysis_delivery_passed'] and not failed['passed']
    assert failed['recovered'] and failed['sample_accounting_balanced']
    good = bounded_acquisition(.6, sample_rate_hz=10000, block_size=1024, features=True)
    assert good['analysis_delivery_passed'] and good['capture_accounting_passed'] and good['recovered']
    assert good['passed'] == good['acquisition_passed']

@pytest.mark.parametrize('floor', [1e-3, 1e-15, 1e-100])
@pytest.mark.parametrize('amplitude', [0., .003, .02])
def test_f07_zero_and_display_floor_never_affect_physical_decisions(floor, amplitude):
    d = EnergyDetector(window_size=16, hop_size=8, conversion_floor_linear=floor)
    ref = GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-20, conversion_floor_linear=floor).get_estimate(S.stream_id, 0)
    m = d.process(block(np.full(16, amplitude*1e-8)), ref)
    assert m.above_on_threshold == (amplitude == .02)
    assert m.below_off_threshold == (amplitude != .02)
    assert math.isfinite(m.signal_power_dbfs) and math.isfinite(m.snr_estimate_db)
    assert m.noise_power_linear == pytest.approx(1e-20, rel=1e-12, abs=0)

def test_f07_zero_training_is_invalid_low_training_is_true_and_capture_survives():
    zero = GaussianNoiseEstimator(window_size=16, warmup_samples=16)
    zero.update_samples(np.zeros(16, np.complex64))
    assert not zero.get_estimate(S.stream_id, 0).valid
    e = GaussianNoiseEstimator(window_size=16, warmup_samples=16)
    e.update_samples(np.full(16, 1e-10, np.complex64))
    assert 10**(e.get_estimate(S.stream_id, 0).noise_power_dbfs/10) == pytest.approx(1e-20, rel=1e-7, abs=0)
    p = pipe(noise_estimator=e)
    p.process_block(block(np.full(64, 2e-10)))
    result = p.flush()
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].event.iq, np.full(64, 2e-10, np.complex64))
    p.shutdown()

def test_async_polling_reserves_bounded_terminal_delivery():
    from signal_processing.runtime import FeatureAnalysisWorker
    p = pipe(enable_feature_analysis=False)
    p.process_block(block(np.full(32, .5))); sc = p.flush()[0]; p.shutdown()
    w = FeatureAnalysisWorker(synchronous=False, max_results=1, skip_incomplete_quality=False)
    first = w.submit(sc)
    rejected = w.submit(dataclasses.replace(sc, event_id='second'))
    assert first.feature_status == 'pending' and rejected.feature_status == 'dropped_overload'
    w.shutdown()
    result = w.get_result(block=False)
    assert result.event_id == sc.event_id and result.feature_status == 'degraded'
    assert w.dropped_result_overload_count == 0 and w._polling_reserved == 0

@pytest.mark.parametrize('scale', [1e-8, 1e-15, 1e8])
def test_f07_empirical_and_adaptive_transport_scale(scale):
    q = policy().without_validation()
    factor = scale**2
    q = dataclasses.replace(q, mean_power_linear=q.mean_power_linear*factor,
          t_on_linear=q.t_on_linear*factor, t_off_linear=q.t_off_linear*factor,
          t_on_dbfs=q.t_on_dbfs+20*math.log10(scale), t_off_dbfs=q.t_off_dbfs+20*math.log10(scale))
    e = AdaptiveRobustNoiseEstimator(window_size=16, initial_noise_power_linear=q.mean_power_linear,
                                     reference_domain=q.reference_domain)
    ref = e.get_estimate(S.stream_id, 0)
    d = EnergyDetector(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1, threshold_policy=q)
    m = d.process(block(np.full(16, .02*scale)), ref)
    assert m.above_on_threshold and m.noise_power_linear == pytest.approx(q.mean_power_linear, rel=1e-12, abs=0)

def test_immutable_subblock_and_raw_preparation_rejection():
    b = block(np.ones(32), 100)
    v = b.slice_received(3, 19)
    assert v.start_sample == 103 and v.end_sample == 119 and np.shares_memory(v.iq, b.iq)
    with pytest.raises(ValueError):
        v.iq.setflags(write=True)
    d = EnergyDetector(window_size=16, hop_size=8)
    d.prepare_detection_iq(np.ones(8, np.complex64))
    with pytest.raises(StreamConsistencyError):
        d.process_windows(block(np.ones(16), 8), NoiseEstimate(S.stream_id, 0, -40))

def test_f06_enforcement_and_explicit_cancellation():
    from benchmarks.run_streaming_benchmarks import acceptance_exit_code
    from benchmarks.sustained_evaluation import bounded_acquisition
    assert acceptance_exit_code({'passed': True}, True) == 0
    assert acceptance_exit_code({'passed': False}, True) == 1
    assert acceptance_exit_code({'passed': False}, False) == 0
    def configure(p):
        p.worker.add_result_sink(lambda _: p.worker.reset())
    r = bounded_acquisition(.6, sample_rate_hz=10000, block_size=1024, features=True, configure_pipeline=configure)
    assert r['recovered'] and r['cancellation_operations'] > 0
    assert not r['analysis_delivery_passed'] and not r['passed']

def test_f07_configured_scale_and_zero_reference_rejection():
    results = []
    for scale in (1., 1e-8):
        d = EnergyDetector(window_size=64, hop_size=32, mode='configured_band', monitored_band_hz=(90000, 110000))
        x = (scale*.5*np.exp(.2j*np.pi*np.arange(1024))).astype(np.complex64)
        ref = NoiseEstimate(S.stream_id, 0, 10*math.log10(1e-4*scale**2))
        results.append(d.process_windows(block(x), ref))
    assert [m.above_on_threshold for m in results[0]] == [m.above_on_threshold for m in results[1]]
    assert all(m.above_on_threshold for m in results[1][-8:])
    d = EnergyDetector(window_size=16, hop_size=8)
    from signal_processing.detection.exceptions import UncalibratedNoiseError
    with pytest.raises(UncalibratedNoiseError):
        d.process(block(np.ones(16)), NoiseEstimate(S.stream_id, 0, -4000))
    assert d._input_end is None

def test_internal_measurements_satisfy_full_public_constructor():
    from signal_processing.core import DetectionMeasurement
    rng = np.random.default_rng(1122)
    for scale in (0., 1e-15, 1., 1e15):
        x = ((rng.normal(size=1024)+1j*rng.normal(size=1024))*.02*scale).astype(np.complex64)
        d = EnergyDetector(window_size=17, hop_size=8)
        ref = NoiseEstimate(S.stream_id, 0, -40+20*math.log10(max(scale, 1e-15)))
        for m in d.process_windows(block(x, 10**12), ref):
            assert DetectionMeasurement(**dataclasses.asdict(m)) == m

def test_supported_settling_is_applicable_to_actual_detector_history():
    q, powers = supported()
    q = dataclasses.replace(q.without_validation(), settling_samples=8)
    q = q.validate_held_out(powers, reference_domain=q.reference_domain,
        independence_assumption='independent_held_out_windows', evidence_id='settled-8')
    d = EnergyDetector(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1, threshold_policy=q, require_supported_pfa=True)
    ref = NoiseEstimate(S.stream_id, 0, 10*math.log10(q.mean_power_linear), reference_domain=q.reference_domain)
    with pytest.raises(PolicyCompatibilityError, match='settling'):
        d.process_windows(block(np.full(16, .003)), ref)
    assert d._input_end is None
    d.prepare_detection_iq(block(np.full(8, .003)))
    m = d.process_windows(block(np.full(16, .003), 8, 1), ref)[0]
    assert m.threshold_evidence_id == q.evidence_id

def test_f06_early_consumer_failure_keeps_full_schedule_accounted():
    from benchmarks.sustained_evaluation import bounded_acquisition
    r = bounded_acquisition(.6, sample_rate_hz=10000, block_size=1024, features=True,
                            configure_pipeline=lambda p: p.worker.shutdown())
    assert r['scheduled_samples'] == 6000 and r['errors']
    assert r['sample_accounting_balanced'] and r['unproduced_samples'] > 0
    assert r['consumed_samples'] + r['dropped_samples'] == 6000
    assert r['recovered'] and not r['passed']
