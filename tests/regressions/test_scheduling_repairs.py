"""Causal batching, bounded completions and exact LTE sequence preparation."""
import math
import multiprocessing
import os
import queue
import signal

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import AdaptiveRobustNoiseEstimator
from signal_processing.runtime import StreamingCapturePipeline
from signal_processing.runtime.capture_stream import CaptureStreamProcess
from signal_processing.runtime.shared_transport import SharedIQTransport
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
from signal_processing.lte.bch import gold_sequence


def gold_oracle(initialization, length):
    a, b = [1]+[0]*30, [(initialization >> k) & 1 for k in range(31)]
    for k in range(1600+length):
        a.append(a[k+3] ^ a[k])
        b.append(b[k+3] ^ b[k+2] ^ b[k+1] ^ b[k])
    return np.array([a[k] ^ b[k] for k in range(1600,1600+length)], np.uint8)


def test_word_gold_matches_independent_bit_recurrence():
    rng = np.random.default_rng(74120)
    for initialization in (0, 1, 42, 503, 2**31-1, *map(int,rng.integers(0,2**31,20))):
        for length in (0, 1, 30, 31, 32, 440, 1920, 4100):
            assert np.array_equal(gold_sequence(initialization,length), gold_oracle(initialization,length))


def test_shared_progress_preserves_real_gap_iq_epoch_and_eof():
    from tests.regressions.test_native_history_and_progress import capture_rows
    assert capture_rows('compact',True,'shared') == capture_rows('pickle',True,'pipe')


def test_shared_progress_timeout_then_real_child_death_is_not_drain():
    source=SharedIQTransport(multiprocessing.get_context('spawn'),1,64)
    pipe=CaptureStreamProcess(source,StreamInfo('death','owned-child',1e6),completion_backend='shared',
        noise_options=dict(window_size=16,calibrated_noise_power_linear=1e-4))
    try:
        with pytest.raises(queue.Empty):
            pipe.get(0)
        os.kill(pipe.pid,signal.SIGTERM)
        with pytest.raises(RuntimeError,match='without EOF'):
            pipe.get(2)
        assert not pipe.drained
    finally:
        pipe.close();source.close()


@pytest.mark.parametrize('backend',['pipe','shared'])
def test_grouped_telemetry_preserves_every_commit_capture_gap_and_tail(backend):
    stream=StreamInfo('grouped','real-spawn',1e6)
    x=np.full(448,.003,np.complex64);x[80:120]=.3;x[288:320]=.2
    outcomes=[]
    for group in (1,4):
        source=SharedIQTransport(multiprocessing.get_context('spawn'),16,64)
        pipe=CaptureStreamProcess(source,stream,completion_backend=backend,progress_group=group,
            noise_options=dict(window_size=17,calibrated_noise_power_linear=1e-4),
            pipeline_options=dict(window_size=17,hop_size=8,ring_capacity=4096,start_persistence=1,
                end_persistence=1,pre_trigger_samples=32,pre_signal_margin_samples=0,fixed_quiet_batch=True))
        for k in range(7):
            source.put((0.,k,10**12+k*64+(32 if k>=2 else 0),x[k*64:k*64+64].tobytes(),32 if k==2 else 0))
        source.put(None)
        rows=[];captures=[]
        try:
            while not pipe.done:
                result=pipe.get(15)
                assert result['error'] is None
                progress=result['progress']
                if progress is None:
                    progress=[] if result['sequence']<0 else [(result['arrival'],result['sequence'],result['start_sample'],result['num_samples'],0,0,0)]
                assert len(progress)<=group
                rows.extend(tuple(r[1:4]) for r in progress)
                for sc in result['captures']:
                    e=sc.event
                    captures.append((e.event_id,e.event_start_sample,e.event_end_sample,e.iq.tobytes(),
                        sc.stream_epoch,sc.known_lost_samples,sc.termination_reason))
                pipe.task_done()
            assert source.unfinished_tasks==0 and pipe.drained
            outcomes.append((rows,captures,result['snapshot']))
        finally:
            pipe.close();source.close()
    assert outcomes[0]==outcomes[1] and len(outcomes[0][0])==7 and outcomes[0][1]


@pytest.mark.parametrize('fault',[False,True])
def test_grouped_quiet_eof_or_fault_publishes_pending_progress(fault):
    source=SharedIQTransport(multiprocessing.get_context('spawn'),4,32)
    stream=StreamInfo('group-tail','real-spawn',1e6)
    pipe=CaptureStreamProcess(source,stream,progress_group=4,noise_mode='adaptive',
        noise_options=dict(window_size=16,initial_noise_power_linear=1e-4,
            adaptation_window_size=128,max_staleness_samples=64 if fault else 10000),
        pipeline_options=dict(window_size=16,hop_size=8,ring_capacity=4096))
    blocks=3 if fault else 2
    for k in range(blocks):source.put((0.,k,k*32,np.full(32,.003,np.complex64).tobytes(),0))
    source.put(None)
    try:
        row=pipe.get(15)
        assert len(row['progress'])==blocks
        assert [r[1] for r in row['progress']]==list(range(blocks))
        assert sum(r[3] for r in row['progress'])==row['snapshot']['consumed_end_sample']
        if fault:
            assert row['error'][0]=='UncalibratedNoiseError' and row['snapshot']['state']=='FAULTED'
            assert row['snapshot']['received_end_sample']==96 and not pipe.drained
        else:
            assert row['error'] is None and pipe.done
        pipe.task_done()
        if not fault:assert pipe.drained and source.unfinished_tasks==0
    finally:
        pipe.close();source.close()


@pytest.mark.parametrize('origin', [0,1,1000,10**12])
@pytest.mark.parametrize('m', [63,127,1024])
def test_idle_transaction_matches_reference_statistics_and_exact_capture(origin,m):
    stream = StreamInfo('causal-idle','test',1e6)
    rng = np.random.default_rng(74121)
    x = ((rng.normal(size=12001)+1j*rng.normal(size=12001))*.005).astype(np.complex64)
    x[4099:4461] += .2
    x[9001:9333] += .1
    # Both long quiet transactions and fallback active/candidate transitions.
    partitions = (1024,1024,333,691,1024,2048,1024,17,1007,2048,1024,1024)
    outcomes = []
    for fast in (False,True):
        noise = AdaptiveRobustNoiseEstimator(window_size=256, adaptation_window_size=m,
            initial_noise_power_linear=1e-4, time_constant_s=.001, max_staleness_samples=100000)
        pipe = StreamingCapturePipeline(stream,noise_estimator=noise,window_size=256,hop_size=64,
            pre_trigger_samples=256,pre_signal_margin_samples=0,ring_capacity=4096,
            max_event_samples=4096,fixed_quiet_batch=fast,enable_feature_analysis=False)
        rows=[];snapshots=[];offset=0
        try:
            for seq,size in enumerate(partitions):
                size=min(size,len(x)-offset)
                if not size: break
                rows.extend(pipe.process_block(IQBlock(stream,seq,origin+offset,None,x[offset:offset+size])))
                offset+=size
                snapshots.append((noise._noise_power_linear, noise.total_accepted_samples,
                    noise.staleness_samples, noise._adaptation_residual.tobytes(),
                    tuple(noise._history_powers), noise._frozen, pipe._noise_admission_sample,
                    pipe._pending_noise_buf.tobytes(), pipe.consumed_end_sample))
            rows.extend(pipe.flush())
            events=[]
            for sc in rows:
                e=sc.event
                assert e.iq.tobytes()==x[e.event_start_sample-origin:e.event_end_sample-origin].tobytes()
                events.append((e.event_start_sample,e.event_end_sample,e.iq.tobytes()))
            outcomes.append((snapshots,events))
        finally:
            pipe.shutdown()
    assert outcomes[0]==outcomes[1] and outcomes[0][1]


@pytest.mark.parametrize('backend', ['shared','pipe'])
@pytest.mark.parametrize('protocol', ['compact','pickle'])
def test_real_spawn_completion_slots_capacity_and_large_record(backend,protocol):
    stream=StreamInfo('slots','real-spawn',1e6)
    source=SharedIQTransport(multiprocessing.get_context('spawn'),4,32768)
    pipe=CaptureStreamProcess(source,stream,completion_capacity=1,completion_backend=backend,
        progress_protocol=protocol,noise_options=dict(window_size=16,calibrated_noise_power_linear=1e-4),
        pipeline_options=dict(window_size=16,hop_size=8,ring_capacity=4096,start_persistence=1,
            end_persistence=1,pre_trigger_samples=32,pre_signal_margin_samples=0,
            max_event_samples=32768,fixed_quiet_batch=True))
    x=np.full(32768,.3,np.complex64)
    try:
        # A full event >64KiB must not deadlock behind a publication after send.
        source.put((0.,0,10**12,x.tobytes(),0));source.put(None)
        captures=[]
        while not pipe.done:
            row=pipe.get(15)
            assert row['error'] is None
            captures.extend(row['captures'])
            pipe.task_done()
        assert captures and sum(c.event.num_samples for c in captures)>=len(x)
        assert any(c.event.iq.nbytes>65536 for c in captures)
        assert all(c.event.iq.tobytes()==x[c.event.event_start_sample-10**12:c.event.event_end_sample-10**12].tobytes()
                   for c in captures)
        assert pipe.drained and pipe.unfinished_completions==0
        with pytest.raises(queue.Empty):
            pipe.get(0)
    finally:
        pipe.close();source.close()


def test_fft_workers_have_same_real_iq_outputs_and_bounded_preparation():
    from tests.lte_pbch_fixture import waveform
    stream=StreamInfo('fft-workers','test',1_920_000.,2.14e9)
    x=waveform(42,ports=2,snr_db=-5,cfo=12345,frames=4,rate=1_920_000.)
    outcomes=[]
    for workers in (1,2):
        receiver=LTEBroadcastReceiver(stream,LTESyncConfig.native(low_snr=True),fft_workers=workers)
        assert len(receiver._fft_cache)<=2
        rows=[]
        for seq,a in enumerate(range(0,len(x),16384)):
            rows.extend(receiver.process_block(IQBlock(stream,seq,10**12+a,None,x[a:a+16384])).mibs)
        rows.extend(receiver.flush().mibs)
        outcomes.append(rows)
    assert outcomes[0]==outcomes[1] and len(outcomes[0])==4


def test_native_hypothesis_batch_matches_all_independent_encoded_masks_and_quarters():
    from signal_processing.lte._native import initialize, _recover_batch
    from signal_processing.lte.bch import recover_bch, _pbch_signs, rate_match_indices, _trellis
    from tests.lte_pbch_fixture import fixture, gold
    initialize('numba',1)
    pci=174
    rng=np.random.default_rng(74122)
    for actual_ports in (1,2,4):
        payload,code=fixture(4,actual_ports)
        signs=1-2*((code.astype(int)+gold(pci,1920))%2)
        for count in (1,2,4):
            for current in range(count-1,4):
                inputs=rng.normal(size=(count,3,480))*.1
                inputs[:,(1,2,4).index(actual_ports),:] += signs[(current-count+1)*480:(current+1)*480].reshape(count,480)
                valid,values=_recover_batch(inputs,_pbch_signs(pci),rate_match_indices(),*_trellis())
                for p,ports in enumerate((1,2,4)):
                    for quarter in range(count-1,4):
                        reference=recover_bch(list(inputs[:,p,:]),pci,quarter,ports,backend='reference')
                        actual=tuple(map(int,values[p,quarter])) if valid[p,quarter] else None
                        assert actual==reference
                index=(1,2,4).index(actual_ports)
                assert valid[index,current] and tuple(map(int,values[index,current]))==tuple(map(int,payload))


@pytest.mark.parametrize('average', [False,True])
@pytest.mark.parametrize('scale', [1e-20,1.,1e20])
def test_received_cp_reduction_and_fft_match_independent_direct_oracle(average,scale):
    from signal_processing.lte.pbch import extract_slot, CENTRAL_BINS
    rng=np.random.default_rng(74123)
    samples=((rng.normal(size=24000)+1j*rng.normal(size=24000))*scale).astype(np.complex64)
    frame,base,frequency=1920,0,12345.
    slot=samples[frame+960:frame+1920].astype(np.complex128)
    t=np.arange(960)/1920000.
    slot*=np.exp(-2j*np.pi*frequency*t)
    pairs=[]
    if average:
        last=(len(samples)-frame)//960-1
        first=max(last-19,0)
        for number in range(first,last+1):
            received=samples[frame+number*960:frame+(number+1)*960].astype(np.complex128)
            position=0
            for l in range(7):
                cp=10 if l==0 else 9
                pairs.extend(received[position+k].conjugate()*received[position+k+128] for k in range(3,cp))
                position+=cp+128
    else:
        position=0
        for l in range(7):
            cp=10 if l==0 else 9
            pairs.extend(slot[position+k].conjugate()*slot[position+k+128] for k in range(3,cp))
            position+=cp+128
    correlation=complex(math.fsum(z.real for z in pairs),math.fsum(z.imag for z in pairs))
    if average:correlation*=np.exp(-2j*np.pi*frequency*128/1920000.)
    residual=np.angle(correlation)*1920000/(2*np.pi*128)
    slot*=np.exp(-2j*np.pi*residual*t)
    expected=np.array([np.fft.fft(slot[p:p+128])[CENTRAL_BINS]/np.sqrt(128) for p in (8,145,282,419,556)])
    grid,actual=extract_slot(samples,frame,base,frequency,average_cfo=average)
    assert abs(actual-frequency-residual)<1e-10
    np.testing.assert_allclose(grid,expected,rtol=1e-12,atol=scale*1e-12)
