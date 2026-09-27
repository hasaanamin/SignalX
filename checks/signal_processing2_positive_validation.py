"""Independent component oracles. Run beside the extracted signal_processing2_review directory."""
from pathlib import Path
import sys, json
import numpy as np
from scipy import signal
import os
ROOT = Path(__file__).resolve().parent
SRC = Path(os.environ.get('SIGNAL_PROCESSING_SRC', ROOT.parent if (ROOT.parent / 'signal_processing').exists() else ROOT / 'signal_processing2_review')).resolve()
sys.path.insert(0, str(SRC))
from signal_processing.core import StreamInfo,IQBlock,NoiseEstimate,SignalEvent
from signal_processing.detection import EnergyDetector
from signal_processing.buffering import CircularIQBuffer
from signal_processing.transforms import compute_fft,compute_psd,compute_welch_psd,compute_autocorrelation,compute_phase
from signal_processing.features import FeatureAggregator,build_default_feature_registry
from signal_processing.context import SignalContext,AnalysisConfig

rng=np.random.default_rng(84017)
stream=StreamInfo('oracles','synthetic',1e6)
results={}
def record(name, fn):
    try:
        details=fn();results[name]={'passed':True,'details':details}
    except Exception as e:results[name]={'passed':False,'exception':type(e).__name__,'message':str(e)}
def freeze_check(a):
    assert not a.flags.writeable
    try:a.setflags(write=True)
    except ValueError:return
    raise AssertionError('Array could be made writable')
def psd_oracle():
    worst=0.;count=0
    for n in [2,3,64,257,2048]:
        z=(rng.normal(size=n)+1j*rng.normal(size=n)).astype(np.complex64)
        for window,wfn in [('rectangular',np.ones),('hann',np.hanning),('hamming',np.hamming),('blackman',np.blackman)]:
            if window in ['hann','blackman'] and n==2:continue
            w=wfn(n)
            for centered in [True,False]:
                for fftlen in [n,2*n]:
                    ours=compute_psd(z,1e6,nfft=fftlen,window=window,centered=centered)
                    f,p=signal.periodogram(z.astype(np.complex128),fs=1e6,window=w,nfft=fftlen,detrend=False,return_onesided=False,scaling='density')
                    if centered:f,p=np.fft.fftshift(f),np.fft.fftshift(p)
                    np.testing.assert_array_equal(ours.frequency_offset_hz,f)
                    np.testing.assert_allclose(ours.power_linear,p,rtol=2e-12,atol=1e-18)
                    freeze_check(ours.power_linear)
                    worst=max(worst,float(np.max(np.abs(ours.power_linear-p))));count+=1
    return {'comparisons':count,'maximum_absolute_density_error':worst}
record('periodogram_matches_scipy',psd_oracle)
def welch_oracle():
    worst=0.;count=0
    for n in [3,65,257,2048]:
        z=(rng.normal(size=n)+1j*rng.normal(size=n)).astype(np.complex64)
        seg=min(n,64)
        for name,wfn in [('rectangular',np.ones),('hann',np.hanning),('hamming',np.hamming),('blackman',np.blackman)]:
            for overlap in [0,seg//2]:
                for fftlen in [seg,2*seg]:
                    ours=compute_welch_psd(z,1e6,nperseg=seg,noverlap=overlap,nfft=fftlen,window=name,centered=False)
                    f,p=signal.welch(z.astype(np.complex128),fs=1e6,window=wfn(seg),nperseg=seg,noverlap=overlap,nfft=fftlen,detrend=False,return_onesided=False,scaling='density')
                    np.testing.assert_array_equal(ours.frequency_offset_hz,f)
                    np.testing.assert_allclose(ours.power_linear,p,rtol=2e-12,atol=1e-18)
                    count+=1;worst=max(worst,float(np.max(np.abs(ours.power_linear-p))))
    return {'comparisons':count,'maximum_absolute_density_error':worst}
record('direct_welch_matches_scipy',welch_oracle)
def corr_oracle():
    worst=0.
    for n in [1,2,3,64,257]:
        z=(rng.normal(size=n)+1j*rng.normal(size=n)).astype(np.complex64)
        ref=np.correlate(z.astype(np.complex128),z.astype(np.complex128),'full')[n-1:]
        for normalized in [False,True]:
            a=compute_autocorrelation(z,normalized=normalized)
            expected=ref/ref[0].real if normalized else ref
            np.testing.assert_allclose(a.values,expected,rtol=1e-11,atol=1e-11)
            freeze_check(a.values);worst=max(worst,float(np.max(np.abs(a.values-expected))))
    return {'comparisons':10,'maximum_absolute_error':worst}
record('fft_autocorrelation_matches_direct_sum',corr_oracle)
def ring_oracle():
    checks=0
    for cap in [1,2,3,7,64,257]:
        b=CircularIQBuffer(capacity=cap);origin=137;b.reset(start_sample=origin);truth=[];cursor=origin
        for _ in range(100):
            n=int(rng.integers(1,3*cap+1));x=(rng.normal(size=n)+1j*rng.normal(size=n)).astype(np.complex64)
            truth.extend(x);b.write(cursor,x);cursor+=n
            lo,hi=b.available_range();assert (lo,hi)==(max(origin,cursor-cap),cursor)
            snap=b.read_range(lo,hi)
            np.testing.assert_array_equal(snap,np.array(truth,dtype=np.complex64)[lo-origin:hi-origin]);checks+=1
            if len(snap):
                old=snap.copy();b.write(cursor,np.ones(1,np.complex64));truth.append(np.complex64(1));cursor+=1
                np.testing.assert_array_equal(snap,old)
    return {'random_write_read_cases':checks,'snapshot_isolation':True}
record('ring_random_wrap_and_oversize_oracle',ring_oracle)
def detector_oracle():
    x=(rng.normal(size=3001)+1j*rng.normal(size=3001)).astype(np.complex64)
    ref=np.array([np.mean(np.abs(x[k:k+31].astype(np.complex128))**2) for k in range(0,len(x)-30,7)])
    out=[]
    for bs in [1,7,30,31,64,512,3001]:
        d=EnergyDetector(window_size=31,hop_size=7);ms=[]
        for seq,k in enumerate(range(0,len(x),bs)):
            b=IQBlock(stream,seq,k,None,x[k:k+bs]);ne=NoiseEstimate(stream.stream_id,seq,0.)
            ms.extend(d.process_windows(b,ne))
        powers=10**(np.array([m.signal_power_dbfs for m in ms])/10)
        np.testing.assert_allclose(powers,ref,rtol=1e-12,atol=1e-12)
        out.append({'block_size':bs,'measurements':len(ms)})
    return out
record('detector_power_and_window_partition_oracle',detector_oracle)
def feature_isolation():
    agg=FeatureAggregator(build_default_feature_registry())
    def c(x):
        return SignalContext(SignalEvent('e',stream,x,0,len(x),0,len(x),0))
    a=(rng.normal(size=2048)+1j*rng.normal(size=2048)).astype(np.complex64)
    b=np.exp(.2j*np.arange(2048)).astype(np.complex64)
    av=agg.extract(c(a));agg.extract(c(b));again=agg.extract(c(a))
    assert len(av.features)==56
    for k,v in av.features.items():
        q=again.features[k];assert v.valid==q.valid and v.reason==q.reason
        if v.valid:assert v.value==q.value
    return {'keys':56,'reused_aggregator_A_B_A_equal':True}
record('feature_extractors_do_not_leak_event_history',feature_isolation)
def phase_isolation():
    z=np.array([1j,0,1j,1j,0,-1j],np.complex64);r=compute_phase(z)
    np.testing.assert_array_equal(r.valid_mask,[True,False,True,True,False,True])
    for a in [r.wrapped_rad,r.unwrapped_rad,r.valid_mask]:freeze_check(a)
    return {'undefined_samples_masked':True,'arrays_bytes_backed':True}
record('phase_validity_and_immutability',phase_isolation)
Path(__file__).with_suffix('.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
if any(not x['passed'] for x in results.values()):sys.exit(1)
