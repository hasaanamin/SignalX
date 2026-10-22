"""Real on-demand analysis; no feature work on the raw capture hot path."""
import dataclasses
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from signal_processing.core import IQBlock
from signal_processing.runtime import FeatureAnalysisWorker, FeatureRequestGate
from tests.regressions.test_g_revision import pipeline, sidecar, aggregate, S


def test_capture_pretrigger_unchanged_until_explicit_request():
    x=np.full(512,.003,np.complex64);x[128:192]=.3;x[320:384]=.2
    outputs=[]
    for mode in ('capture_only','on_demand'):
        p=pipeline(enable_feature_analysis=mode=='on_demand',synchronous_analysis=False,
            feature_analysis_mode='on_demand' if mode=='on_demand' else 'automatic',
            pre_trigger_samples=32,skip_incomplete_quality=False)
        try:
            captures=p.process_block(IQBlock(S,0,10**12,None,x))+p.flush()
            assert captures and not p.worker.unfinished_work and not p.worker.processed_results
            outputs.append([(s.event.event_start_sample,s.event.event_end_sample,s.event.iq.tobytes()) for s in captures])
            for s in captures:
                start=s.event.event_start_sample-10**12
                assert s.event.iq.tobytes()==x[start:start+s.event.num_samples].tobytes()
            if mode=='on_demand':
                assert p.request_features(captures[0]).feature_status=='pending'
                p.shutdown()
                completed=p.worker.processed_results
                assert len(completed)==1 and completed[0].feature_status in ('ok','degraded')
                assert completed[0].event_id==captures[0].event_id
                assert len(completed[0].features.features)==56
        finally:p.shutdown()
    assert outputs[0]==outputs[1]


def test_ui_throttle_classification_and_overload_never_wait_or_call_sink(monkeypatch):
    import signal_processing.runtime.feature_requests as module
    clock=[100.]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    entered,release=threading.Event(),threading.Event()
    results=[]
    def backend(ctx):
        entered.set()
        assert release.wait(5)
        return aggregate(ctx)
    w=FeatureAnalysisWorker(synchronous=False,max_queue_size=1,analysis_function=backend,
        result_delivery='callbacks',result_sink=results.append,skip_incomplete_quality=False)
    gate=FeatureRequestGate(w)
    a,b,c=[sidecar(name) for name in ('A','B','C')]
    try:
        assert gate.request(a,purpose='ui').feature_status=='pending'
        assert entered.wait(2)
        assert gate.request(b,purpose='ui').feature_status=='skipped'
        assert gate.request(b,purpose='classification').feature_status=='pending'
        assert gate.request(c,purpose='classification').feature_status=='dropped_overload'
        assert not results  # running analysis has not returned; rejection had no callback
        assert vars(gate.snapshot())==dict(ui_requests=2,classification_requests=2,
            accepted=2,throttled_ui=1,rejected_capacity=1)
    finally:
        release.set();w.shutdown(timeout=5)
    assert [s.event_id for s in results]==['A','B']
    assert all(s.feature_status in ('ok','degraded') for s in results)
    assert w.unfinished_work==0 and w.state.value=='CLOSED'


def test_ui_rate_cap_no_catchup_burst_and_polling_capacity(monkeypatch):
    import signal_processing.runtime.feature_requests as module
    clock=[100.]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    w=FeatureAnalysisWorker(synchronous=False,max_results=1,skip_incomplete_quality=False)
    gate=FeatureRequestGate(w)
    sc=sidecar()
    try:
        assert gate.request(sc,purpose='ui').feature_status=='pending'
        clock[0]+=.01
        assert gate.request(sc,purpose='ui').feature_status=='skipped'
        clock[0]+=10.
        # A completed but unpolled result still owns its reservation.
        assert gate.request(sc,purpose='ui').feature_status=='dropped_overload'
        assert gate.snapshot().accepted==1
    finally:w.shutdown()


@pytest.mark.parametrize('rate',[True,0,-1,61,float('nan'),float('inf')])
def test_invalid_ui_rate_rejected(rate):
    w=FeatureAnalysisWorker(synchronous=False)
    try:
        with pytest.raises(ValueError):FeatureRequestGate(w,ui_rate_hz=rate)
    finally:w.shutdown()


def test_synchronous_gate_and_invalid_pipeline_mode_rejected():
    w=FeatureAnalysisWorker()
    try:
        with pytest.raises(ValueError,match='asynchronous'):FeatureRequestGate(w)
    finally:w.shutdown()
    with pytest.raises(ValueError,match='asynchronous'):
        pipeline(enable_feature_analysis=True,feature_analysis_mode='on_demand')


@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_native_sync_direct_fsum_oracles(scale):
    from signal_processing.lte._native import initialize_sync,_tracking_scores,_sss_scores
    from signal_processing.lte.sequences import sss_hypotheses
    initialize_sync('numba')
    rng=np.random.default_rng(741231)
    x=((rng.normal(size=30000)+1j*rng.normal(size=30000))*scale).astype(np.complex64).astype(np.complex128)
    positions=np.array([29000,29003])
    templates=(rng.normal(size=(3,128))+1j*rng.normal(size=(3,128))).astype(complex)
    observed=_tracking_scores(x,positions,4,templates)
    expected=np.zeros_like(observed)
    for i,p in enumerate(positions):
        windows=[x[p-repeat*9600:p-repeat*9600+128] for repeat in range(4)]
        energy=math.fsum(float(z.real)**2+float(z.imag)**2 for row in windows for z in row)
        for branch,row in enumerate(templates):
            dots=[]
            for window in windows:
                values=list(window*row)
                dot=complex(math.fsum(float(z.real) for z in values),math.fsum(float(z.imag) for z in values))
                dots.append(dot.real**2+dot.imag**2)
            expected[i,branch]=math.fsum(dots)/(62*energy)
    np.testing.assert_allclose(observed,expected,rtol=1e-12,atol=1e-15)
    bank=sss_hypotheses(1)
    eq=x[:4*62].reshape(4,62)
    direct=np.array([[complex(math.fsum(float(z.real)*float(s) for z,s in zip(row,h)),
                              math.fsum(float(z.imag)*float(s) for z,s in zip(row,h))) for h in bank] for row in eq])
    np.testing.assert_allclose(_sss_scores(bank,eq),direct,rtol=1e-12,atol=scale*1e-13)


@pytest.mark.parametrize('quarter',[0,2])
def test_bch_block_cadence_decodes_each_new_real_codeword(quarter):
    from signal_processing.core import StreamInfo
    from signal_processing.lte import LTEBroadcastReceiver,LTESyncConfig
    from tests.lte_pbch_fixture import waveform
    stream=StreamInfo('bch-cadence','encoded',1_920_000.,2.14e9)
    x=waveform(42,ports=2,rate=1_920_000.,frames=12,first_quarter=quarter,snr_db=20.,cfo=12345)
    outputs=[]
    for cadence in ('every_frame','bch_block'):
        receiver=LTEBroadcastReceiver(stream,LTESyncConfig.native(low_snr=True),decode_cadence=cadence)
        found=[]
        for seq,k in enumerate(range(0,len(x),16384)):
            found.extend(receiver.process_block(IQBlock(stream,seq,10**12+k,None,x[k:k+16384])).mibs)
        found.extend(receiver.flush().mibs)
        frames={round(float(m.input_frame_start-10**12-431)/19200):m for m in found}
        assert len(frames)==len(found)
        outputs.append((frames,receiver.snapshot()))
    assert len(outputs[0][0])==12
    expected_blocks={(f+quarter)//4 for f in outputs[0][0]}
    assert {(f+quarter)//4 for f in outputs[1][0]}==expected_blocks
    assert len(outputs[1][0])==len(expected_blocks)
    for f,result in outputs[1][0].items():
        ref=outputs[0][0][f]
        assert dataclasses.replace(result,profile_id=ref.profile_id)==ref
    assert outputs[1][1]['decoded_repetitions_avoided']==12-len(expected_blocks)
    assert outputs[1][1]['pbch_attempted_frames']==len(expected_blocks)
    assert outputs[1][1]['retained_bch_frames']<=4


def test_bch_cadence_does_not_claim_undecodable_data_or_cross_gap_cache():
    from signal_processing.core import StreamInfo
    from signal_processing.lte import LTEBroadcastReceiver,LTESyncConfig
    from tests.lte_pbch_fixture import waveform
    stream=StreamInfo('bch-gap','encoded',1_920_000.,2.14e9)
    r=LTEBroadcastReceiver(stream,LTESyncConfig.native(low_snr=True),decode_cadence='bch_block')
    valid=waveform(42,ports=2,rate=1_920_000.,snr_db=20.,frames=4)
    first=r.process_block(IQBlock(stream,0,10**12,None,valid))
    assert len(first.mibs)==1
    corrupt=waveform(42,ports=2,rate=1_920_000.,snr_db=20.,frames=4,corrupt_pbch=True)
    bad=r.process_block(IQBlock(stream,1,10**12+len(valid)+32,None,corrupt,dropped_samples_before=32))
    tail=r.flush()
    assert not bad.mibs and not tail.mibs
    assert r.snapshot()['decode_cadence']=='bch_block'
    assert r.snapshot()['epoch']==1 and r.snapshot()['decoded_repetitions_avoided']==0


@pytest.mark.parametrize('pci,cfo,rate,expected_count',[(42,12345.,1_920_000.,3),(251,-7500.,2_000_000.,4)])
def test_batched_sync_matches_real_reference_and_real_mibs(pci,cfo,rate,expected_count):
    from signal_processing.core import StreamInfo
    from signal_processing.lte import LTEBroadcastReceiver,LTESyncConfig
    from tests.lte_pbch_fixture import waveform
    x=waveform(pci,ports=2,rate=rate,snr_db=-5,cfo=cfo,seed=2233)
    stream=StreamInfo('parity','encoded',rate,2.14e9)
    outputs=[]
    for backend,native in (('reference','numpy'),('bounded','numpy'),('bounded','numba')):
        r=LTEBroadcastReceiver(stream,LTESyncConfig.low_snr(rate),search_backend=backend,sync_backend=native,measure_timing=True)
        sync=[];mibs=[]
        for seq,k in enumerate(range(0,len(x),4093)):
            batch=r.process_block(IQBlock(stream,seq,10**12+k,None,x[k:k+4093]))
            sync.extend(batch.synchronization);mibs.extend(batch.mibs)
        batch=r.flush();sync.extend(batch.synchronization);mibs.extend(batch.mibs)
        # Seed 2233 at native rate has one undecodable -5-dB frame in BOTH
        # paths. Preserve that failure, rather than change seed/claim four CRCs.
        assert len(mibs)==expected_count
        assert r.stage_timing['pbch_decode']['calls']>=4
        assert r.stage_timing['synchronization']['calls']>0
        outputs.append((sync,mibs))
    for actual in outputs[1:]:
        assert outputs[0][1]==actual[1]
        for a,b in zip(outputs[0][0],actual[0],strict=True):
            assert dataclasses.replace(a,pss_metric=b.pss_metric,sss_metric=b.sss_metric,sss_peak_ratio=b.sss_peak_ratio)==b
            assert a.pss_metric==pytest.approx(b.pss_metric,rel=1e-12,abs=1e-15)
            assert a.sss_metric==pytest.approx(b.sss_metric,rel=1e-12,abs=1e-15)
