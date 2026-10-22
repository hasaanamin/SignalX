"""Finite full-capture IPC, exact bytes, real backpressure and pipe fallback."""
import multiprocessing
import time

import numpy as np
import pytest

from signal_processing.core import StreamInfo
from signal_processing.runtime.capture_stream import CaptureStreamProcess
from signal_processing.runtime.shared_transport import SharedIQTransport


def test_full_shared_payload_matches_original_gap_epoch_iq_and_eof():
    from tests.regressions.test_native_history_and_progress import capture_rows
    assert capture_rows('compact',True,'shared_payload')==capture_rows('pickle',True,'pipe')


def test_full_shared_payload_preserves_grouped_commit_and_captures():
    from tests.regressions.test_scheduling_repairs import test_grouped_telemetry_preserves_every_commit_capture_gap_and_tail
    test_grouped_telemetry_preserves_every_commit_capture_gap_and_tail('shared_payload')


@pytest.mark.parametrize('payload_size',[1024,262144])
def test_large_exact_capture_uses_bounded_slot_or_truthful_pipe_fallback(payload_size):
    source=SharedIQTransport(multiprocessing.get_context('spawn'),4,16384)
    stream=StreamInfo('large-completion','real-source',1e6)
    pipe=CaptureStreamProcess(source,stream,completion_capacity=4,completion_backend='shared_payload',
        max_completion_bytes=payload_size,
        noise_options=dict(window_size=16,calibrated_noise_power_linear=1e-4),
        pipeline_options=dict(window_size=16,hop_size=8,ring_capacity=65536,pre_trigger_samples=32,
            pre_signal_margin_samples=0,post_signal_samples=0,min_post_trigger_samples=0,
            max_event_samples=32768,start_persistence=1,end_persistence=1,fixed_quiet_batch=True))
    x=np.full(32768,.003,np.complex64);x[:16384]=.3
    try:
        for k in range(2): source.put((0.,k,k*16384,x[k*16384:(k+1)*16384].tobytes(),0))
        source.put(None)
        if payload_size==262144:
            # Without a receiving parent, a >64-KiB capture must cross into its
            # reserved shared slot before the child can consume/ack source EOF.
            # An OS-pipe-only implementation blocks before that EOF ack.
            deadline=time.monotonic()+8
            while source.unfinished_tasks and time.monotonic()<deadline: time.sleep(.005)
            assert source.unfinished_tasks==0
        events=[]
        while not pipe.done:
            record=pipe.get(8)
            assert record['error'] is None
            events+=record['captures']
            pipe.task_done()
        assert len(events)==1
        event=events[0].event
        assert event.num_samples>=16384
        assert event.iq.tobytes()==x[event.event_start_sample:event.event_end_sample].tobytes()
        assert not event.iq.flags.writeable
        assert pipe.drained and pipe.unfinished_completions==source.unfinished_tasks==0
        assert len(pipe._completions._payloads)==4*payload_size
    finally:
        pipe.close();source.close()
    assert pipe._completions is None


@pytest.mark.parametrize('value',[True,0,1023,1.5,17*1024*1024])
def test_full_payload_capacity_rejects_before_spawning(value):
    with pytest.raises(ValueError):
        CaptureStreamProcess(None,StreamInfo('invalid','capacity',1e6),
            completion_backend='shared_payload',max_completion_bytes=value)


def test_corrupted_real_completion_is_failed_not_a_successful_drain():
    source=SharedIQTransport(multiprocessing.get_context('spawn'),2,64)
    stream=StreamInfo('corrupt-completion','test',1e6)
    pipe=CaptureStreamProcess(source,stream,completion_capacity=2,completion_backend='shared_payload',
        noise_options=dict(window_size=16,calibrated_noise_power_linear=1e-4))
    try:
        source.put((0.,0,0,np.full(64,.003,np.complex64).tobytes(),0));source.put(None)
        deadline=time.monotonic()+5
        while source.unfinished_tasks and time.monotonic()<deadline: time.sleep(.005)
        assert source.unfinished_tasks==0
        # A genuine produced first record, then a malformed transport kind.
        pipe._completions._kinds[0]=99
        with pytest.raises(RuntimeError,match='Invalid capture completion slot'): pipe.get(2)
        assert not pipe.drained and pipe.unfinished_completions==0
    finally:
        pipe.close();source.close()
