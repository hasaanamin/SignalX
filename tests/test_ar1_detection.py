"""Independent AR1 math and public streaming integration checks."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import lfilter

from signal_processing.core.contracts import IQBlock, StreamInfo, NoiseReferenceDomain
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.detection.ar_noise import fit_ar1_noise
from signal_processing.detection.energy_detector import EnergyDetector
from signal_processing.detection.noise_estimation import GaussianNoiseEstimator
from signal_processing.detection.exceptions import PolicyCompatibilityError
from signal_processing.runtime.pipeline import StreamingCapturePipeline


@pytest.fixture
def calibrated():
    rng = np.random.default_rng(2233)
    z = (rng.normal(size=40000)+1j*rng.normal(size=40000))*np.sqrt(3.6e-5/2)
    x = lfilter([1],[1,-.8],z)[2048:].astype(np.complex64)
    return x, fit_ar1_noise(x[:14336],1e6)


def reference(model):
    return GaussianNoiseEstimator(window_size=32, calibrated_noise_power_linear=model.innovation_power,
            reference_domain=NoiseReferenceDomain(1e6,preprocessing_id=model.model_id))


def test_fit_and_separate_residual_reference(calibrated):
    x,model=calibrated
    assert abs(model.coefficient-.8)<.025
    cut=14336//2
    u=x[:cut].astype(np.complex128)
    mu=np.mean(u)
    a=np.sum((u[1:]-mu)*np.conj(u[:-1]-mu))/np.sum(abs(u[:-1]-mu)**2)
    residual=(x[cut:14336].astype(np.complex128)-mu)-a*(x[cut-1:14335].astype(np.complex128)-mu)
    assert model.innovation_power==pytest.approx(np.mean(abs(residual)**2),rel=1e-12)
    assert model.raw_noise_power>2*model.innovation_power
    assert model.model_id!=replace(model,coefficient=model.coefficient+.001).model_id


@pytest.mark.parametrize('origin',[0,1000,10**12])
def test_whitened_partition_power_oracle(calibrated,origin):
    x,model=calibrated
    data=x[15000:17048]
    stream=StreamInfo('ar','test',1e6)
    u=data.astype(np.complex128)-model.mean
    expected=u[1:]-model.coefficient*u[:-1]
    oracle=[float(np.sum(abs(expected[i:i+32])**2)/32) for i in range(0,len(expected)-31,8)]
    for block_size in (1,7,8,31,128,2048):
        d=EnergyDetector(window_size=32,hop_size=8,ar1_model=model)
        est=reference(model).get_estimate('ar',0)
        out=[]
        for seq,i in enumerate(range(0,len(data),block_size)):
            out.extend(d.process_windows(IQBlock(stream,seq,origin+i,None,data[i:i+block_size]),est))
        assert [m.start_sample for m in out]==[origin+1+8*i for i in range(len(oracle))]
        np.testing.assert_allclose([m.signal_power_linear for m in out],oracle,rtol=1e-12,atol=0)
        d.reset()
        replay=d.process_windows(IQBlock(stream,0,origin,None,data),est)
        np.testing.assert_allclose([m.signal_power_linear for m in replay],oracle,rtol=1e-12)


def test_direct_gap_rejection_preserves_whitener(calibrated):
    x,model=calibrated
    d=EnergyDetector(window_size=32,hop_size=8,ar1_model=model)
    s=StreamInfo('ar','test',1e6)
    e=reference(model).get_estimate('ar',0)
    d.process_windows(IQBlock(s,0,0,None,x[:15]),e)
    state=(d._ar1_previous,d._input_end,d._residual_iq.copy())
    with pytest.raises(StreamConsistencyError):
        d.process_windows(IQBlock(s,1,100,None,x[15:40]),e)
    assert (d._ar1_previous,d._input_end)==state[:2]
    np.testing.assert_array_equal(d._residual_iq,state[2])
    assert d.process_windows(IQBlock(s,1,15,None,x[15:64]),e)


def test_raw_reference_cannot_be_attached_to_whitened_path(calibrated):
    x,m=calibrated
    d=EnergyDetector(window_size=32,hop_size=8,ar1_model=m)
    raw=GaussianNoiseEstimator(calibrated_noise_power_linear=m.raw_noise_power,reference_domain=NoiseReferenceDomain(1e6)).get_estimate('ar',0)
    with pytest.raises(PolicyCompatibilityError):
        d.process_windows(IQBlock(StreamInfo('ar','test',1e6),0,0,None,x[:64]),raw)
    assert d._ar1_previous is None


def test_real_capture_keeps_raw_iq_and_features(calibrated):
    x,m=calibrated
    iq=x[16000:24000].copy()
    iq[2000:2064]+=(.1*np.exp(-2j*np.pi*.22*np.arange(64))).astype(np.complex64)
    stream=StreamInfo('ar','test',1e6)
    p=StreamingCapturePipeline(stream,window_size=256,hop_size=8,noise_estimator=reference(m),
       ar1_model=m,ar1_windows=(32,64,256),refine_ar1_boundaries=True,
       pre_trigger_samples=256,post_signal_samples=256,min_post_trigger_samples=256,
       end_persistence=3,max_event_samples=8192)
    captures=[]
    try:
        for seq,start in enumerate(range(0,len(iq),137)):
            captures.extend(p.process_block(IQBlock(stream,seq,start,None,iq[start:start+137])))
        captures.extend(p.flush())
        c=next(c for c in captures if c.event.detected_start_sample<2064 and c.event.detected_end_sample>2000)
        np.testing.assert_array_equal(c.event.iq,iq[c.event.event_start_sample:c.event.event_end_sample])
        assert c.event.pre_trigger_samples>=256
        assert c.boundary_method=='ar1_local_variance_edges_v1'
        assert c.coarse_detected_start_sample is not None
        assert c.feature_status=='ok'
        assert len(c.features.features)==56
        assert p.flush()==[]
    finally:
        p.shutdown()


@pytest.mark.parametrize('data',[np.zeros(2048,np.complex64),np.ones(2048,np.complex64),np.ones(100,np.complex64),np.ones(65537,np.complex64)])
def test_invalid_training(data):
    with pytest.raises(ValueError):
        fit_ar1_noise(data,1e6)


def test_unfitted_pipeline_is_rejected(calibrated):
    _,m=calibrated
    with pytest.raises(ValueError,match='fixed residual'):
        StreamingCapturePipeline(StreamInfo('ar','test',1e6),ar1_model=m)


def test_untagged_reference_rejected_before_whitening(calibrated):
    x,m=calibrated
    d=EnergyDetector(ar1_model=m)
    e=GaussianNoiseEstimator(calibrated_noise_power_linear=m.innovation_power).get_estimate('ar',0)
    with pytest.raises(PolicyCompatibilityError,match='tagged'):
        d.process_windows(IQBlock(StreamInfo('ar','test',1e6),0,0,None,x[:512]),e)
    assert d._ar1_previous is None and d._input_end is None


def test_spectral_bank_direct_oracle_and_partitions(calibrated):
    from scipy.stats import gamma
    x,m=calibrated
    data=x[16000:18048].copy()
    data[700:956]+=.045  # DC; no known frequency is supplied to the detector.
    u=data.astype(np.complex128)-m.mean
    y=u[1:]-m.coefficient*u[:-1]
    oracle=[]
    for i in range(0,len(y)-255,8):
        window=y[i:i+256]
        candidates=[]
        for n in (32,64,256):
            # Direct component sum, independent of production helper.
            z=window[-n:]
            p=sum(float(v.real)**2+float(v.imag)**2 for v in z)/n
            candidates.append((p,n,'ar1_innovation_power',n))
        # Explicit DFT, not the implementation's FFT.
        k=np.arange(256)
        dft=np.exp(-2j*np.pi*np.outer(k,k)/256)@window
        powers=abs(dft)**2/256
        for center in range(0,256,32):
            candidates.append((float(sum(powers[(center+j)%256] for j in range(-16,16))/32),
                               32,'ar1_subband_bin_mean_power',256))
        ranked=[]
        for p,n,domain,extent in candidates:
            on=m.innovation_power*gamma.isf(1e-4/11,a=n,scale=1/n)
            off=m.innovation_power*gamma.isf(1e-2/11,a=n,scale=1/n)
            ranked.append((p,on,off,domain,extent))
        active=[r for r in ranked if r[0]>=r[1]]
        chosen=max(active,key=lambda r:r[0]/r[1]) if active else max(ranked,key=lambda r:r[0]/r[2])
        oracle.append((1001+i+256-chosen[4],1001+i+256,*chosen[:4]))
    for size in (1,47,2048):
        d=EnergyDetector(window_size=256,hop_size=8,ar1_model=m,ar1_windows=(32,64,256),ar1_subbands=8)
        out=[]
        for seq,start in enumerate(range(0,len(data),size)):
            out.extend(d.process_windows(IQBlock(StreamInfo('ar','test',1e6),seq,1000+start,None,data[start:start+size]),
                                         reference(m).get_estimate('ar',0)))
        assert len(out)==len(oracle)
        for actual,(a,b,p,on,off,domain) in zip(out,oracle):
            assert (actual.start_sample,actual.end_sample)==(a,b)
            assert actual.signal_power_linear==pytest.approx(p,rel=1e-11,abs=0)
            assert actual.above_on_threshold==(p>=on)
            assert actual.below_off_threshold==(p<off)
            assert actual.measurement_domain==domain
        assert len(d._residual_iq)<256


def test_refinement_failure_preserves_finalized_capture(calibrated,monkeypatch):
    x,m=calibrated
    data=x[16000:20000].copy()
    data[1000:1200]+=.5
    def fail(*args,**kwargs):
        raise ArithmeticError('injected edge failure')
    monkeypatch.setattr('signal_processing.detection.boundaries.refine_ar1_capture',fail)
    stream=StreamInfo('ar','test',1e6)
    p=StreamingCapturePipeline(stream,window_size=256,hop_size=8,noise_estimator=reference(m),
        ar1_model=m,ar1_windows=(32,64,256),refine_ar1_boundaries=True,
        enable_feature_analysis=False,max_event_samples=8192)
    try:
        with pytest.raises(ArithmeticError,match='edge failure') as error:
            p.process_block(IQBlock(stream,0,0,None,data))
        recovered=error.value.completed_captures
        assert len(recovered)==1
        event=recovered[0].event
        np.testing.assert_array_equal(event.iq,data[event.event_start_sample:event.event_end_sample])
        assert recovered[0].feature_status=='failed'
        assert p.flush()==[]
    finally:
        p.shutdown()
