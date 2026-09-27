"""Independent mathematical checks for all ten feature families.

Higher-order expectations use the general set-partition definition of a
joint cumulant, rather than copying the implementation's closed formulas.
"""
from pathlib import Path
import json
import math
import sys
import numpy as np
from scipy import signal, stats
import os
ROOT = Path(__file__).parent
SRC = Path(os.environ.get('SIGNAL_PROCESSING_SRC', ROOT.parent if (ROOT.parent / 'signal_processing').exists() else ROOT / 'signal_processing2_review')).resolve()
sys.path.insert(0, str(SRC))
from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import SignalEvent, StreamInfo
from signal_processing.features import FeatureAggregator, build_default_feature_registry
from signal_processing.features.higher_order import HigherOrderFeatureExtractor
from signal_processing.features.bandwidth import BandwidthFeatureExtractor, BandwidthFeatureConfig


def context(x):
    x=np.asarray(x,np.complex64)
    return SignalContext(SignalEvent("formula",StreamInfo("formula","oracle",1e6),x,0,len(x),0,len(x),0))


def partitions(items):
    if not items:
        yield []
        return
    first,*rest=items
    for part in partitions(rest):
        yield [[first]]+[list(b) for b in part]
        for i in range(len(part)):
            yield [([first]+list(b) if j==i else list(b)) for j,b in enumerate(part)]


def cumulant(variables):
    result=0j
    for blocks in partitions(list(range(len(variables)))):
        term=complex((-1)**(len(blocks)-1)*math.factorial(len(blocks)-1))
        for block in blocks:
            product=np.ones_like(variables[0])
            for i in block:product=product*variables[i]
            term*=np.mean(product)
        result+=term
    return result


def run():
    rng=np.random.default_rng(551)
    samples=(rng.normal(size=171)+1j*rng.normal(size=171)).astype(np.complex64)
    ctx=context(samples)
    actual=FeatureAggregator(build_default_feature_registry()).extract(ctx).features
    x=samples.astype(np.complex128);amplitude=np.abs(x);power=amplitude**2
    expected={
        "amplitude.mean":amplitude.mean(),"amplitude.rms":np.sqrt(np.mean(power)),
        "amplitude.peak":amplitude.max(),"amplitude.min":amplitude.min(),
        "amplitude.std":amplitude.std(),"amplitude.variance":amplitude.var(),
        "amplitude.median":np.median(amplitude),"amplitude.peak_to_peak":np.ptp(amplitude),
        "amplitude.crest_factor":amplitude.max()/np.sqrt(np.mean(power)),
        "power.mean_linear":power.mean(),"power.mean_dbfs":10*np.log10(power.mean()),
        "power.peak_linear":power.max(),"power.peak_dbfs":10*np.log10(power.max()),
        "power.min_linear":power.min(),"power.std_linear":power.std(),
        "power.variance_linear":power.var(),"power.median_linear":np.median(power),
        "power.papr_linear":power.max()/power.mean(),"power.papr_db":10*np.log10(power.max()/power.mean()),
        "temporal.duration_samples":len(x),"temporal.duration_s":len(x)/1e6,
        "stats.amplitude.skewness":stats.skew(amplitude,bias=True),
        "stats.amplitude.excess_kurtosis":stats.kurtosis(amplitude,fisher=True,bias=True),
    }
    for percentile in (1,5,25,75,95,99):
        expected[f"stats.amplitude.q{percentile:02d}"]=np.percentile(amplitude,percentile)
    expected["stats.amplitude.iqr"]=np.percentile(amplitude,75)-np.percentile(amplitude,25)
    increments=np.angle(x[1:]*np.conj(x[:-1]));resultant=abs(np.mean(x/amplitude))
    expected.update({"phase.circular_resultant_length":resultant,"phase.circular_variance":1-resultant,
                     "phase.increment_mean_rad":increments.mean(),"phase.increment_std_rad":increments.std(),
                     "phase.increment_rms_rad":np.sqrt(np.mean(increments**2)),"phase.increment_abs_mean_rad":np.mean(abs(increments))})
    # The project documents a symmetric Hann window (np.hanning), not SciPy's
    # default periodic Hann. Supply the corresponding explicit window.
    freqs,psd=signal.periodogram(x,fs=1e6,window=signal.windows.hann(len(x),sym=True),
                                 detrend=False,return_onesided=False,scaling="density")
    order=np.argsort(freqs);freqs=freqs[order];psd=psd[order];prob=psd/psd.sum()
    expected.update({"frequency.peak_offset_hz":freqs[np.argmax(psd)],"frequency.center_offset_hz":np.dot(freqs,prob),
                     "spectral.entropy":stats.entropy(prob)/np.log(len(psd)),
                     "spectral.flatness":np.exp(np.mean(np.log(psd)))/np.mean(psd),
                     "spectral.rolloff_85_offset_hz":freqs[np.searchsorted(np.cumsum(prob),.85)],
                     "spectral.rolloff_95_offset_hz":freqs[np.searchsorted(np.cumsum(prob),.95)]})
    rho=np.array([np.sum(x[k:]*np.conj(x[:len(x)-k])) for k in range(len(x))])/np.sum(power)
    mags=np.abs(rho[1:]);peak=int(np.argmax(mags))+1
    expected.update({"correlation.nonzero_peak_magnitude":abs(rho[peak]),
                     "correlation.nonzero_peak_lag_samples":peak,"correlation.nonzero_peak_lag_s":peak/1e6,
                     "correlation.nonzero_peak_phase_rad":np.angle(rho[peak]),
                     "correlation.mean_nonzero_magnitude":mags.mean(),"correlation.rms_nonzero_magnitude":np.sqrt(np.mean(mags**2))})
    errors={}
    for key,value in expected.items():
        assert actual[key].valid,key
        errors[key]=abs(actual[key].value-float(value))
        assert np.isclose(actual[key].value,value,rtol=2e-6,atol=2e-7),(key,actual[key].value,value)
    cumulant_cases=[]
    cases=[np.tile([-1,1],20),np.tile([1,1j,-1,-1j],10),
           np.repeat([-.2+1j,1.5-.3j,2+2j],[7,23,10]),samples]
    indices=((2,0),(4,0),(4,1),(4,2),(6,0),(6,3))
    for case_no,values in enumerate(cases):
        ctx=context(values);z=ctx.signal_iq.astype(np.complex128);z=z-z.mean();z=z/np.sqrt(np.mean(abs(z)**2))
        measured=HigherOrderFeatureExtractor().extract(ctx).features
        for n,q in indices:
            oracle=abs(cumulant([z]*(n-q)+[np.conj(z)]*q))
            key=f"higher_order.normalized_abs_c{n}{q}"
            assert np.isclose(measured[key].value,oracle,rtol=1e-8,atol=1e-8),(key,measured[key].value,oracle)
            cumulant_cases.append(dict(case=case_no,key=key,error=abs(measured[key].value-oracle)))
    class PSD:
        frequency_offset_hz=np.arange(9,dtype=float)
        power_linear=np.array([0,0,1,2,4,2,1,0,0],dtype=float)
    class C:
        def psd(self):return PSD()
    bw=BandwidthFeatureExtractor(BandwidthFeatureConfig(mode="raw_psd")).extract(C()).features
    assert bw["bandwidth.occupied_99_hz"].valid and bw["bandwidth.occupied_99_hz"].value==5
    assert bw["bandwidth.three_db_hz"].valid and bw["bandwidth.three_db_hz"].value==3
    return dict(passed=True,direct_scalar_comparisons=len(expected),
                covered_families=sorted({k.split('.')[0] for k in expected}|{"higher_order","bandwidth"}),
                general_joint_cumulant_comparisons=len(cumulant_cases),
                maximum_cumulant_absolute_error=max(c["error"] for c in cumulant_cases),
                exact_known_spectrum_bandwidth_comparisons=2,
                maximum_direct_scalar_absolute_error=max(errors.values()))


if __name__=="__main__":
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
