"""Bounded preparation and overlapping real process-feature service."""
import dataclasses
import math
import threading

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo, NoiseEstimate
from signal_processing.detection import EnergyDetector, AdaptiveRobustNoiseEstimator
from signal_processing.runtime import StreamingCapturePipeline, FeatureAnalysisWorker
from tests.regressions.test_process_analysis import capture


@pytest.mark.parametrize('n,h', [(16,8), (17,8), (64,32)])
@pytest.mark.parametrize('origin', [0, 10**12])
def test_prepared_active_powers_match_independent_exact_windows(n,h,origin):
    stream = StreamInfo('powers', 'test', 1e6)
    rng = np.random.default_rng(2233)
    x = ((rng.normal(size=2048)+1j*rng.normal(size=2048))*1e-8).astype(np.complex64)
    x[0] = 1
    detector = EnergyDetector(window_size=n,hop_size=h)
    noise = NoiseEstimate(stream.stream_id,0,-160)
    block = IQBlock(stream,0,origin,None,x)
    prepared = detector._prepare_full_band(block)
    assert detector._input_end is None
    results = []
    for a in range(0,len(x),h):
        results += detector._commit_full_band(prepared,block.slice_received(a,min(a+h,len(x))),noise)
        assert detector._input_end == origin+min(a+h,len(x))
        assert len(detector._residual_iq) < n
    for measurement in results:
        a,b = measurement.start_sample-origin, measurement.end_sample-origin
        oracle = math.fsum(float(z.real)**2+float(z.imag)**2 for z in x[a:b])/n
        assert measurement.signal_power_linear == pytest.approx(oracle,rel=1e-14,abs=0)


@pytest.mark.parametrize('partition', [17, 1024, 3000])
@pytest.mark.parametrize('statistic_size', [1, 64, 256, 1024])
def test_adaptive_preparation_preserves_causality_and_pretrigger(partition, statistic_size):
    stream = StreamInfo('adaptive-batch','test',1e6)
    rng = np.random.default_rng(82)
    x = ((rng.normal(size=8192)+1j*rng.normal(size=8192))*.007).astype(np.complex64)
    x[3111:3677] += .1
    observations = []
    for fast in (False,True):
        estimator = AdaptiveRobustNoiseEstimator(window_size=64,initial_noise_power_linear=1e-4,
            time_constant_s=.01, adaptation_window_size=statistic_size)
        p = StreamingCapturePipeline(stream,window_size=64,hop_size=32,ring_capacity=4096,
            noise_estimator=estimator,pre_trigger_samples=128,pre_signal_margin_samples=0,
            max_event_samples=4096,enable_feature_analysis=False,fixed_quiet_batch=fast)
        events=[]
        for seq,a in enumerate(range(0,len(x),partition)):
            events += p.process_block(IQBlock(stream,seq,1000+a,None,x[a:a+partition]))
        events += p.flush()
        summaries=[]
        for sc in events:
            ev=sc.event
            np.testing.assert_array_equal(ev.iq,x[ev.event_start_sample-1000:ev.event_end_sample-1000])
            summaries.append((ev.event_start_sample,ev.event_end_sample,ev.iq.tobytes()))
        observations.append((summaries,estimator._noise_power_linear,estimator._total_accepted_samples,
                             estimator.staleness_samples,p._noise_admission_sample,
                             estimator._adaptation_residual.tobytes(),tuple(estimator._history_powers)))
        p.shutdown()
    assert observations[0] == observations[1]
    assert observations[0][0]


@pytest.mark.parametrize('partition', [17, 64, 1024])
def test_idle_hysteresis_collapse_keeps_candidate_and_cap_rearm(partition):
    from signal_processing.detection import GaussianNoiseEstimator
    stream = StreamInfo('idle-rearm', 'test', 1e6)
    noise = GaussianNoiseEstimator(window_size=64, calibrated_noise_power_linear=1e-4)
    detector = EnergyDetector(window_size=64, hop_size=32)
    on, off, *_ = detector._linear_thresholds(noise.get_estimate(stream.stream_id, 0))
    x = np.full(4096, np.sqrt((on+off)/2), np.complex64)
    x[256:768] = .2
    x[1536:2048] = .003  # releases cap rearm, unlike the middle plateau
    x[2304:2800] = .2
    results = []
    for fast in (False, True):
        p = StreamingCapturePipeline(stream,window_size=64,hop_size=32,ring_capacity=4096,
            noise_estimator=GaussianNoiseEstimator(window_size=64,calibrated_noise_power_linear=1e-4),
            start_persistence=2,end_persistence=2,max_event_samples=256,pre_trigger_samples=64,
            pre_signal_margin_samples=0,post_signal_samples=0,min_post_trigger_samples=0,
            enable_feature_analysis=False,fixed_quiet_batch=fast)
        captures=[]
        states=[]
        try:
            for seq,a in enumerate(range(0,len(x),partition)):
                captures.extend(p.process_block(IQBlock(stream,seq,10**12+a,None,x[a:a+partition])))
                states.append((p._segmenter.state,p._segmenter.awaiting_off_rearm,p._segmenter._on_count))
            captures.extend(p.flush())
            results.append((states,[(sc.event.event_start_sample,sc.event.event_end_sample,
                sc.event.detected_start_sample,sc.event.detected_end_sample,sc.event.iq.tobytes()) for sc in captures]))
        finally:p.shutdown()
    assert results[0]==results[1]
    assert len(results[0][1])==2


def test_pool_order_exact_real_results_and_bounded_blocked_delivery():
    sc = capture()
    reference = FeatureAnalysisWorker()
    expected = reference.submit(sc).analysis_result
    reference.shutdown()
    entered,release = threading.Event(),threading.Event()
    delivered=[]
    def sink(result):
        delivered.append(result)
        if len(delivered)==1:
            entered.set()
            assert release.wait(10)
    worker=FeatureAnalysisWorker(synchronous=False,analysis_backend='process_pool',
        result_delivery='callbacks',result_sink=sink,max_queue_size=4)
    try:
        worker.submit(sc)
        assert entered.wait(10)
        for i in range(4):
            ev=dataclasses.replace(sc.event,event_id=f'pool-{i}')
            assert worker.submit(dataclasses.replace(sc,event=ev,event_id=ev.event_id)).feature_status=='pending'
        assert worker.unfinished_work==5
        assert worker.submit(sc).feature_status=='dropped_overload'
        with pytest.raises(TimeoutError):
            worker.shutdown(timeout=.02)
        assert worker.state.value=='CLOSING'
    finally:
        release.set()
        worker.shutdown(timeout=10)
    assert [v.event_id for v in delivered]==[sc.event_id]+[f'pool-{i}' for i in range(4)]
    assert delivered[0].analysis_result==expected
    assert all(v.feature_status=='ok' and len(v.features.features)==56 for v in delivered)
    assert worker.unfinished_work==0 and worker.state.value=='CLOSED'


def test_capacity_reservation_polling_wakes_only_after_required_drain():
    worker=FeatureAnalysisWorker(synchronous=False,result_delivery='polling',max_results=1)
    sc=capture()
    ready=threading.Event()
    try:
        assert worker.submit(sc).feature_status=='pending'
        assert not worker.wait_for_capacity(.02)
        waiter=threading.Thread(target=lambda: ready.set() if worker.wait_for_capacity(3) else None)
        waiter.start()
        assert worker.get_result(timeout=3).feature_status=='ok'
        assert ready.wait(3)
        waiter.join(3)
        assert not waiter.is_alive()
    finally:worker.shutdown()


def test_all_pool_servers_dead_give_submitted_failures_and_drain():
    import os,signal
    sc=capture()
    worker=FeatureAnalysisWorker(synchronous=False,analysis_backend='process_pool')
    try:
        for pid in worker._process_backend.pids:os.kill(pid,signal.SIGTERM)
        for i in range(3):
            ev=dataclasses.replace(sc.event,event_id=f'dead-{i}')
            assert worker.submit(dataclasses.replace(sc,event=ev,event_id=ev.event_id)).feature_status=='pending'
        results=[worker.get_result(timeout=8) for _ in range(3)]
        assert [r.event_id for r in results]==['dead-0','dead-1','dead-2']
        assert all(r.feature_status=='failed' and r.features is None for r in results)
    finally:worker.shutdown(timeout=8)
    assert worker.unfinished_work==0


@pytest.mark.parametrize('order',['submission','completion'])
def test_real_paused_server_obeys_explicit_result_order(order):
    import psutil
    original=capture()
    event=dataclasses.replace(original.event,event_id='independent-second',iq=original.event.iq*.5)
    second=dataclasses.replace(original,event=event,event_id=event.event_id)
    delivered=[]
    ready=threading.Event()
    def sink(result):
        delivered.append(result)
        ready.set()
    worker=FeatureAnalysisWorker(synchronous=False,analysis_backend='process_pool',process_workers=2,
        result_order=order,result_delivery='callbacks',result_sink=sink,max_queue_size=4)
    paused=psutil.Process(worker.backend_pids[0])
    try:
        paused.suspend()
        assert worker.submit(original).feature_status=='pending'
        assert worker.submit(second).feature_status=='pending'
        if order=='completion':
            assert ready.wait(5)
            assert delivered[0].event_id==second.event_id and delivered[0].feature_status=='ok'
            assert worker.unfinished_work>=1
        else:
            assert not ready.wait(.05)
        with pytest.raises(TimeoutError):worker.shutdown(timeout=.02)
        assert worker.state.value=='CLOSING'
    finally:
        paused.resume()
        worker.shutdown(timeout=8)
    expected=[original.event_id,second.event_id] if order=='submission' else [second.event_id,original.event_id]
    assert [r.event_id for r in delivered]==expected
    assert all(r.feature_status=='ok' and r.stream_epoch==42 for r in delivered)
    reference=FeatureAnalysisWorker()
    try:
        for sc in (original,second):
            expected_result=reference.submit(sc).analysis_result
            assert next(r for r in delivered if r.event_id==sc.event_id).analysis_result==expected_result
    finally:reference.shutdown()
    assert worker.unfinished_work==0


def test_prepared_midblock_fault_preserves_public_completed_outputs(monkeypatch):
    import os,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    monkeypatch.setenv('SIGNAL_PROCESSING_SRC',str(root))
    sys.path.insert(0,str(root/'checks'))
    import f_revision_probes as probes
    original=probes.pipeline
    reference=probes.partial_block_emissions()
    monkeypatch.setattr(probes,'pipeline',lambda **kw:original(fixed_quiet_batch=True,**kw))
    actual=probes.partial_block_emissions()
    assert actual==reference
    assert actual['error']=='PolicyCompatibilityError'
    assert actual['consumed']==64 and actual['received']==128
    assert len(actual['returned'])==1 and actual['returned'][0]['end']==48
    assert not actual['recovered']


def test_process_payload_limit_is_terminal_and_later_real_job_recovers():
    from signal_processing.detection import GaussianNoiseEstimator
    stream=StreamInfo('large-process-event','test',1e6)
    p=StreamingCapturePipeline(stream,window_size=16,hop_size=8,ring_capacity=8192,
        max_event_samples=8192,start_persistence=1,end_persistence=1,
        pre_trigger_samples=0,pre_signal_margin_samples=0,post_signal_samples=0,min_post_trigger_samples=0,
        noise_estimator=GaussianNoiseEstimator(window_size=16,calibrated_noise_power_linear=1e-4),
        enable_feature_analysis=False)
    iq=np.full(6100,.003,np.complex64);iq[:6000]=.2
    try:events=p.process_block(IQBlock(stream,0,0,None,iq))+p.flush()
    finally:p.shutdown()
    assert len(events)==1 and events[0].event.num_samples>4096
    worker=FeatureAnalysisWorker(synchronous=False,analysis_backend='process')
    try:
        assert worker.submit(events[0]).feature_status=='pending'
        rejected=worker.get_result(timeout=5)
        assert rejected.feature_status=='failed' and rejected.event_id==events[0].event_id
        assert '4096' in rejected.feature_reason
        assert worker.submit(capture()).feature_status=='pending'
        assert worker.get_result(timeout=5).feature_status=='ok'
    finally:worker.shutdown()
    assert worker.unfinished_work==0


@pytest.mark.parametrize('age',[16,80,300])
def test_preparation_expiry_has_identical_committed_horizon(age):
    stream=StreamInfo('expiry','test',1e6)
    outcomes=[]
    for fast in (False,True):
        e=AdaptiveRobustNoiseEstimator(window_size=64,initial_noise_power_linear=1e-4,
            adaptation_window_size=256,max_staleness_samples=age)
        p=StreamingCapturePipeline(stream,window_size=64,hop_size=32,ring_capacity=4096,
            noise_estimator=e,enable_feature_analysis=False,fixed_quiet_batch=fast)
        failure=None
        try:
            for i in range(8):
                p.process_block(IQBlock(stream,i,i*256,None,np.full(256,.3,np.complex64)))
        except Exception as exc:failure=type(exc).__name__
        outcomes.append((failure,p.state.value,p.consumed_end_sample,p.received_end_sample,e.staleness_samples))
        p.shutdown()
    assert outcomes[0]==outcomes[1] and outcomes[0][0]=='UncalibratedNoiseError'
