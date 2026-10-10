"""Public profile integration, evidence requirements and exact retained IQ."""
from dataclasses import replace, FrozenInstanceError
import json
import math
import numpy as np
import pytest
from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import ThresholdPolicy
from signal_processing.detection.profiles import LONG_BURST_IID
from signal_processing.detection.exceptions import PolicyCompatibilityError
from benchmarks.qualify_detection import calibrate


@pytest.fixture(scope='module')
def configured():
    # Faster moderate-tail control only. Release -10 dB / 1e-4 qualification is
    # separately executed with 100,000 genuine held-out 4096-sample windows.
    profile=replace(LONG_BURST_IID,pfa_on=.01,pfa_off=.1)
    policy,_=calibrate(profile,validation_windows=2000)
    assert policy.supported_pfa
    return profile,policy


@pytest.mark.parametrize('origin,partition', [(0,1024),(1,31),(1000,4096),(10**12,777)])
def test_real_profile_exact_iq_retention_and_feature_provenance(configured,origin,partition):
    profile,policy=configured; s=StreamInfo('profile','synthetic',1e6)
    p=profile.create_pipeline(s,ThresholdPolicy.from_dict(json.loads(json.dumps(policy.to_dict()))))
    rng=np.random.default_rng(2444)
    x=((rng.normal(size=28673)+1j*rng.normal(size=28673))*math.sqrt(1e-4/2)).astype(np.complex64)
    x[12288:16384]+=.1
    events=[]
    for seq,i in enumerate(range(0,len(x),partition)):
        events+=p.process_block(IQBlock(s,seq,origin+i,None,x[i:i+partition]))
    events+=p.flush(); assert p.flush()==[];p.shutdown()
    assert len(events)==1
    sc=events[0]; ev=sc.event
    assert ev.iq.tobytes()==x[ev.event_start_sample-origin:ev.event_end_sample-origin].tobytes()
    assert sc.actual_leading_samples>=4096
    assert sc.boundary_uncertainty_samples==4096 and sc.detection_profile_id==profile.profile_id
    assert sc.analysis_result.capture_quality.boundary_uncertainty_samples==4096
    assert len(sc.features.features)==56
    assert sc.features.event_id==sc.event_id


def test_profile_rejects_uncertified_or_changed_policy(configured):
    profile,policy=configured; s=StreamInfo('profile','synthetic',1e6)
    with pytest.raises(PolicyCompatibilityError): profile.create_pipeline(s,policy.without_validation())
    with pytest.raises(PolicyCompatibilityError): replace(profile,hop_size=128).create_pipeline(s,policy)
    with pytest.raises(FrozenInstanceError): profile.window_size=64
    assert replace(profile,pre_trigger_samples=4097).profile_id!=profile.profile_id


def test_strict_metadata_rejection_and_gap_keep_profile(configured):
    profile,policy=configured;s=StreamInfo('profile','synthetic',1e6,100e6)
    p=profile.create_pipeline(s,policy,enable_feature_analysis=False)
    x=np.full(16385,.003,np.complex64);x[8192:]=.1
    p.process_block(IQBlock(s,0,1000,None,x))
    from tests.regressions.test_h_metadata import snapshot
    from signal_processing.core.exceptions import StreamConsistencyError
    before=snapshot(p)
    b=IQBlock(replace(s,center_frequency_hz=100000000.05),1,17417,None,np.full(512,.003,np.complex64),dropped_samples_before=32)
    with pytest.raises(StreamConsistencyError): p.process_block(b)
    assert snapshot(p)==before
    out=p.process_block(replace(b,stream=s))+p.flush();p.shutdown()
    assert len(out)==1 and out[0].known_lost_samples==32
    assert out[0].detection_profile_id==profile.profile_id


def test_long_profile_cap_and_quiet_rearm_preserve_exact_immutable_iq(configured):
    profile,policy=configured;s=StreamInfo('profile-cap','synthetic',1e6)
    p=profile.create_pipeline(s,policy,enable_feature_analysis=False)
    # A capped continuous emitter does not create an endless train of events.
    # The established A17 contract requires quiet before a new trigger.
    x=np.full(140003,.003,np.complex64);x[8192:100000]=.1;x[112288:120480]=.1
    events=[]
    for seq,i in enumerate(range(0,len(x),777)):
        events+=p.process_block(IQBlock(s,seq,10**12+i,None,x[i:i+777]))
    events+=p.flush();assert p.flush()==[];p.shutdown()
    assert len(events)==2
    assert events[0].termination_reason=='max_duration'
    assert len({e.event_id for e in events})==len(events)
    assert any(len(e.event.iq)==profile.max_event_samples for e in events)
    for sc in events:
        ev=sc.event
        assert len(ev.iq)<=profile.max_event_samples
        assert ev.iq.tobytes()==x[ev.event_start_sample-10**12:ev.event_end_sample-10**12].tobytes()
        assert not ev.iq.flags.writeable
        assert sc.detection_profile_id==profile.profile_id
