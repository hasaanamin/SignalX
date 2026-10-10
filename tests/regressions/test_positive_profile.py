"""Real public adaptive profile and immutable effective-control identity."""
import dataclasses
import numpy as np
import pytest
from signal_processing.core import IQBlock,StreamInfo
from signal_processing.detection import POSITIVE_SNR_IID, AdaptiveDetectionProfile


def test_public_profile_captures_received_bytes_and_retains_pretrigger():
    profile=POSITIVE_SNR_IID
    stream=StreamInfo('positive-profile','synthetic',1e6)
    p=profile.create_pipeline(stream,1e-4,enable_feature_analysis=False)
    rng=np.random.default_rng(99)
    x=((rng.normal(size=4096)+1j*rng.normal(size=4096))*.007).astype(np.complex64)
    x[1024:2048]+=.04
    events=[]
    try:
        for seq,a in enumerate(range(0,len(x),1024)):
            events+=p.process_block(IQBlock(stream,seq,10**12+a,None,x[a:a+1024]))
        events+=p.flush()
    finally:p.shutdown()
    assert len(events)==1
    sc=events[0];event=sc.event
    assert sc.actual_leading_samples>=256 and event.complete
    assert event.iq.tobytes()==x[event.event_start_sample-10**12:event.event_end_sample-10**12].tobytes()
    assert sc.detection_profile_id==profile.profile_id


@pytest.mark.parametrize('key,value',[('adaptation_window_size',512),('time_constant_s',.1),('persistence',3),('pfa_on',.0002)])
def test_effective_adaptive_control_changes_identity(key,value):
    assert dataclasses.replace(POSITIVE_SNR_IID,**{key:value}).profile_id!=POSITIVE_SNR_IID.profile_id


@pytest.mark.parametrize('key,value',[('pfa_on',True),('sample_rate_hz',float('nan')),('adaptation_window_size',1.5),('name',None)])
def test_invalid_profile_controls_reject(key,value):
    with pytest.raises(ValueError):AdaptiveDetectionProfile(**{key:value})
