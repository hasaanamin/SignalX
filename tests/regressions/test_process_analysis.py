"""Real default extraction in a persistent spawned child, not fabricated results."""
import dataclasses
import os
import threading

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import GaussianNoiseEstimator
from signal_processing.runtime import FeatureAnalysisWorker, StreamingCapturePipeline


def capture():
    stream = StreamInfo('spawn-test', 'synthetic', 1e6)
    p = StreamingCapturePipeline(stream, window_size=16, hop_size=8, ring_capacity=4096,
        start_persistence=1, end_persistence=1, pre_trigger_samples=0,
        pre_signal_margin_samples=0, post_signal_samples=0, min_post_trigger_samples=0,
        enable_feature_analysis=False,
        noise_estimator=GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4))
    x = np.full(128, .003, np.complex64); x[:32] = .5
    result = p.process_block(IQBlock(stream, 0, 1000, None, x)) + p.flush()
    p.shutdown()
    return dataclasses.replace(result[0], stream_epoch=42)


@pytest.mark.parametrize('backend', ['process', 'process_pool'])
def test_process_matches_default_and_preserves_immutable_policy(backend):
    sc = capture()
    reference = FeatureAnalysisWorker()
    expected = reference.submit(sc)
    reference.shutdown()
    worker = FeatureAnalysisWorker(synchronous=False, analysis_backend=backend)
    try:
        assert worker.backend_pid != os.getpid()
        for _ in range(3):
            assert worker.submit(sc).feature_status == 'pending'
            result = worker.get_result(timeout=10)
            assert result.feature_status == 'ok', result.feature_reason
            assert result.event_id == sc.event_id and result.stream_epoch == 42
            assert result.analysis_result == expected.analysis_result
            assert not result.event.iq.flags.writeable
            with pytest.raises(TypeError):
                result.analysis_result.policy.extractor_versions['amplitude'] = 'foreign'
    finally:
        worker.shutdown(timeout=5)
    assert worker.unfinished_work == 0 and worker.state.value == 'CLOSED'


@pytest.mark.parametrize('backend', ['process', 'process_pool'])
def test_process_blocked_sink_timeout_overload_and_reentrant_reset(backend):
    entered, release = threading.Event(), threading.Event()
    delivered = []
    worker = None
    def sink(result):
        delivered.append(result)
        entered.set()
        assert release.wait(5)
        worker.reset()
    worker = FeatureAnalysisWorker(synchronous=False, analysis_backend=backend,
        max_queue_size=1, result_delivery='callbacks', result_sink=sink)
    sc = capture()
    try:
        worker.submit(sc)
        assert entered.wait(5)
        worker.submit(sc)
        assert worker.submit(sc).feature_status == 'dropped_overload'
        with pytest.raises(TimeoutError):
            worker.shutdown(timeout=.02)
        assert worker.state.value == 'CLOSING'
    finally:
        release.set()
        worker.shutdown(timeout=5)
    assert len(delivered) == 1 and worker.unfinished_work == 0


def test_dead_process_reports_failure_with_submitted_identity():
    import signal
    worker = FeatureAnalysisWorker(synchronous=False, analysis_backend='process')
    sc = capture()
    try:
        os.kill(worker.backend_pid, signal.SIGTERM)
        worker.submit(sc)
        result = worker.get_result(timeout=5)
        assert result.feature_status == 'failed' and result.event_id == sc.event_id
        worker.submit(sc)
        assert worker.get_result(timeout=5).feature_status == 'failed'
    finally:
        worker.shutdown(timeout=5)
    assert worker.unfinished_work == 0


@pytest.mark.parametrize('backend', ['process', 'process_pool'])
def test_real_pipeline_jobs_survive_multiple_raw_gaps(backend):
    stream=StreamInfo('gap-process','synthetic',1e6)
    entered, release = threading.Event(), threading.Event()
    delivered=[]
    def sink(sc):
        delivered.append(sc)
        if len(delivered)==1:
            entered.set()
            assert release.wait(8)
    p=StreamingCapturePipeline(stream,window_size=16,hop_size=8,ring_capacity=4096,
        start_persistence=1,end_persistence=1,pre_trigger_samples=0,
        pre_signal_margin_samples=0,post_signal_samples=0,min_post_trigger_samples=0,
        synchronous_analysis=False,analysis_backend=backend,skip_incomplete_quality=False,
        result_delivery='callbacks',
        noise_estimator=GaussianNoiseEstimator(window_size=16,calibrated_noise_power_linear=1e-4))
    p.worker.add_result_sink(sink)
    raw=[]
    try:
        x=np.full(128,.003,np.complex64);x[:32]=.5
        raw+=p.process_block(IQBlock(stream,0,0,None,x))
        assert entered.wait(5)
        raw+=p.process_block(IQBlock(stream,1,128,None,np.full(64,.5,np.complex64)))
        raw+=p.process_block(IQBlock(stream,2,224,None,np.full(64,.003,np.complex64),dropped_samples_before=32))
        raw+=p.process_block(IQBlock(stream,3,320,None,np.full(64,.003,np.complex64),overflow=True))
        raw+=p.flush()
    finally:
        release.set();p.shutdown(timeout=8)
    assert len(raw)==len(delivered)==2
    assert {s.event_id for s in raw}=={s.event_id for s in delivered}
    assert delivered[1].feature_status=='degraded'
    # N16/H8: the first strong trailing window is [120,136), not
    # [128,144). Its eight real quiet-prefix samples must be preserved.
    assert delivered[1].event.event_start_sample==120 and delivered[1].event.event_end_sample==192
    np.testing.assert_array_equal(delivered[1].event.iq,
        np.concatenate((np.full(8,.003,np.complex64),np.full(64,.5,np.complex64))))
    assert delivered[1].known_lost_samples==32
    assert p.worker.discarded_old_generation_count==0 and p.worker.unfinished_work==0


@pytest.mark.parametrize('kwargs', [dict(synchronous=True), dict(synchronous=False, analysis_function=lambda x:x)])
def test_unsupported_process_modes_reject_before_spawn(kwargs):
    with pytest.raises(ValueError):
        FeatureAnalysisWorker(analysis_backend='process', **kwargs)
