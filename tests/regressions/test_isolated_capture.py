"""Actual spawned capture state and exact immutable IQ, not surrogate engines."""
import dataclasses
import multiprocessing
import os
import signal

import numpy as np
import pytest

from signal_processing.core import StreamInfo, IQBlock
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.runtime.isolated_capture import IsolatedCapturePipeline
from signal_processing.runtime.capture_stream import CaptureStreamProcess
from signal_processing.runtime.shared_transport import SharedIQTransport


OPTIONS=dict(window_size=17,hop_size=8,ring_capacity=4096,start_persistence=1,end_persistence=1,
    pre_trigger_samples=32,pre_signal_margin_samples=0,post_signal_samples=0,min_post_trigger_samples=0,
    max_event_samples=1024,fixed_quiet_batch=True)
NOISE=dict(window_size=17,calibrated_noise_power_linear=1e-4)


def test_isolated_atomic_metadata_retry_exact_pretrigger_and_reset():
    stream=StreamInfo('isolated','test',1e6,center_frequency_hz=1e8)
    p=IsolatedCapturePipeline(stream,pipeline_options=OPTIONS,noise_options=NOISE,enable_feature_analysis=False)
    x=np.full(192,.003,np.complex64);x[64:97]=.3
    try:
        block=IQBlock(stream,0,10**12,None,x[:29])
        assert p.process_block(block)==[]
        before=p.retention_snapshot(),p.consumed_end_sample,p.received_end_sample,p.state
        bad=IQBlock(dataclasses.replace(stream,center_frequency_hz=1e8+.05),1,10**12+29,None,x[29:])
        with pytest.raises(StreamConsistencyError):p.process_block(bad)
        assert before==(p.retention_snapshot(),p.consumed_end_sample,p.received_end_sample,p.state)
        events=p.process_block(IQBlock(stream,1,10**12+29,None,x[29:]))+p.flush()
        assert len(events)==1
        e=events[0].event
        assert e.iq.tobytes()==x[e.event_start_sample-10**12:e.event_end_sample-10**12].tobytes()
        assert not e.iq.flags.writeable
        p.reset()
        replay=p.process_block(IQBlock(stream,0,10**12,None,x))+p.flush()
        assert len(replay)==1 and replay[0].event.iq.tobytes()==e.iq.tobytes()
    finally:p.shutdown()


def test_direct_stream_owned_completions_known_gap_and_eof():
    stream=StreamInfo('direct','test',1e6)
    source=SharedIQTransport(multiprocessing.get_context('spawn'),16,1024)
    p=CaptureStreamProcess(source,stream,pipeline_options=OPTIONS,noise_options=NOISE,completion_capacity=2)
    x=np.full(81,.003,np.complex64);x[64:]=.3
    source.put((0.,0,1000,x.tobytes(),0))
    source.put((0.,1,1113,np.full(64,.003,np.complex64).tobytes(),32))
    source.put(None)
    records=[]
    try:
        while not p.done:
            record=p.get(10);records.append(record);p.task_done()
    finally:p.close();source.close()
    assert [r['sequence'] for r in records]==[0,1,-1]
    assert p.drained and p.unfinished_completions==0
    assert not any(r['error'] for r in records)
    captures=[sc for r in records for sc in r['captures']]
    assert len(captures)==1 and captures[0].known_lost_samples==32
    e=captures[0].event
    assert e.event_end_sample==1081
    assert e.iq.tobytes()==x[e.event_start_sample-1000:].tobytes()
    assert captures[0].stream_epoch==0 and not e.iq.flags.writeable


def test_isolated_death_does_not_claim_known_commit_horizon():
    stream=StreamInfo('dead-capture','test',1e6)
    p=IsolatedCapturePipeline(stream,pipeline_options=OPTIONS,noise_options=NOISE,enable_feature_analysis=False)
    try:
        os.kill(p.capture_pid,signal.SIGTERM)
        with pytest.raises((EOFError,BrokenPipeError,OSError)):
            p.process_block(IQBlock(stream,0,0,None,np.full(64,.003,np.complex64)))
        assert not p.consumed_horizon_known and p.state.value=='FAULTED'
    finally:p.shutdown()


def test_stream_completion_capacity_requires_ack_and_death_is_not_drain():
    import queue
    stream=StreamInfo('bounded-completions','test',1e6)
    source=SharedIQTransport(multiprocessing.get_context('spawn'),16,1024)
    p=CaptureStreamProcess(source,stream,pipeline_options=OPTIONS,noise_options=NOISE,completion_capacity=1)
    try:
        for i in range(3):source.put((0.,i,i*64,np.full(64,.003,np.complex64).tobytes(),0))
        source.put(None)
        first=p.get(10)
        assert first['sequence']==0 and p.unfinished_completions==1
        with pytest.raises(queue.Empty):p.get(.02)
        p.task_done()
        assert p.get(10)['sequence']==1
        p.task_done()
        assert p.get(10)['sequence']==2
        p.task_done()
        p.get(10);p.task_done()
        assert p.drained
        with pytest.raises(ValueError):p.task_done()
    finally:p.close();source.close()


def test_dead_stream_reports_unknown_progress_and_releases_resources():
    stream=StreamInfo('dead-stream','test',1e6)
    source=SharedIQTransport(multiprocessing.get_context('spawn'),16,1024)
    p=CaptureStreamProcess(source,stream,pipeline_options=OPTIONS,noise_options=NOISE)
    try:
        os.kill(p.pid,signal.SIGTERM)
        with pytest.raises(RuntimeError,match='lost|died'):
            p.get(5)
        assert not p.drained
    finally:p.close();source.close()
    assert not p.drained
