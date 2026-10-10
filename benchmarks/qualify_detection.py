"""Offline live-IQ qualification of the predeclared long-burst candidate.

Truth never enters the pipeline. One independently seeded recording contains one
burst; success is one-to-one IoU >= .2, not an arbitrary fragment. Fixed training,
held-out and trial seeds are disjoint. These are synthetic, not receiver records.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import sys
import time
import numpy as np
from scipy.stats import gamma, beta, ncx2, chi2
from scipy import signal

sys.path.insert(0, os.environ.get('SIGNAL_PROCESSING_SRC', str(Path(__file__).resolve().parents[1])))
from signal_processing.core import StreamInfo, IQBlock
from signal_processing.detection import EnergyDetector, GaussianNoiseEstimator, ThresholdPolicy
from signal_processing.detection.profiles import LONG_BURST_IID, LONG_BURST_AR1
from benchmarks.sustained_evaluation import match_intervals


def noise(rng,n):
    return ((rng.normal(size=n)+1j*rng.normal(size=n))*math.sqrt(1e-4/2)).astype(np.complex64)


def model_noise(rng,shape,colored=False):
    if not colored: return noise(rng,int(np.prod(shape))).reshape(shape)
    innovation=noise(rng,int(np.prod(shape))).reshape(shape)
    initial=noise(rng,shape[0]).reshape(-1,1)*.8
    result,_=signal.lfilter([.6],[1,-.8],innovation,axis=1,zi=initial)
    return result.astype(np.complex64)


def calibrate(profile=LONG_BURST_IID, validation_windows=100000):
    colored=profile.noise_assumption.endswith('ar1_0.8')
    domain=EnergyDetector(window_size=profile.window_size,hop_size=profile.hop_size).reference_domain
    estimator=GaussianNoiseEstimator(window_size=profile.window_size,warmup_samples=1048576,reference_domain=domain)
    rng=np.random.default_rng(240001)
    for _ in range(256): estimator.update_samples(model_noise(rng,(1,4096),colored).ravel())
    mean=estimator.noise_power_linear
    # Conservative design point declared before evaluation, not fitted to held-out
    # observations. Nominal ON/OFF requirements remain 1e-4 and 1e-2.
    on=mean*gamma.isf(profile.pfa_on/8,a=profile.window_size,scale=1/profile.window_size)
    off=mean*gamma.isf(profile.pfa_off/2,a=profile.window_size,scale=1/profile.window_size)
    training_windows=1048576//profile.window_size
    if colored:
        # Predeclared empirical design on separate actual colored IQ, never iid
        # thresholds reused for AR noise. Held-out support is still mandatory.
        training=np.empty(10000)
        for start in range(0,10000,16):
            z=model_noise(rng,(min(16,10000-start),profile.window_size),True).astype(np.complex128)
            training[start:start+len(z)]=np.mean(z.real*z.real+z.imag*z.imag,axis=1)
        on=float(training.max()*1.01)
        off=float(np.quantile(training,1-profile.pfa_off/2))
        training_windows=10000
    policy=ThresholdPolicy(mode='empirical_matched' if colored else 'analytic_gamma',window_size=profile.window_size,hop_size=profile.hop_size,
        pfa_on=profile.pfa_on,pfa_off=profile.pfa_off,mean_power_linear=mean,
        t_on_linear=float(on),t_off_linear=float(off),t_on_dbfs=10*math.log10(on),t_off_dbfs=10*math.log10(off),
        training_samples=1048576+(10000*profile.window_size if colored else 0),training_windows=training_windows,reference_domain=domain)
    rng=np.random.default_rng(240002)
    powers=np.empty(validation_windows)
    digest=hashlib.sha256()
    for start in range(0,validation_windows,64):
        size=min(64,validation_windows-start)
        iq=model_noise(rng,(size,profile.window_size),colored)
        digest.update(iq.tobytes())
        # Independent direct mathematical reduction, not a saved distribution draw.
        z=iq.astype(np.complex128)
        powers[start:start+size]=np.mean(z.real*z.real+z.imag*z.imag,axis=1)
    policy=policy.validate_held_out(powers,reference_domain=domain,
        independence_assumption='independent_held_out_windows',evidence_id='synthetic_'+profile.name+'_seed240002_'+digest.hexdigest())
    return policy,dict(iq_sha256=digest.hexdigest(),windows=validation_windows,
        samples=validation_windows*profile.window_size,seed=240002,
        independence='independent generated records, stationary independent Gaussian AR initial states when colored; not inferred from spacing on a receiver',
        maximum_power=float(powers.max()))


def waveform(rng,duration,kind,frequency,bandwidth):
    t=np.arange(duration)/1e6
    if kind=='cw': wave=np.ones(duration,complex)
    else:
        # Fixed-dataset complex periodic bandlimited Gaussian emitter. All occupied
        # bins are inside the declared bandwidth. It is not a receiver noise model.
        spectrum=np.zeros(duration,complex)
        bins=np.abs(np.fft.fftfreq(duration,1e-6))<=bandwidth/2
        spectrum[bins]=rng.normal(size=bins.sum())+1j*rng.normal(size=bins.sum())
        wave=np.fft.ifft(spectrum)
    wave*=np.exp(2j*np.pi*frequency*t)
    return wave/math.sqrt(float(np.mean(np.abs(wave)**2)))


def evaluate_cell(policy, duration, offset, kind, frequency, bandwidth, snr, trials, cell_id, profile=LONG_BURST_IID):
    s=StreamInfo('long-burst-qualification','synthetic-iid',1e6)
    hits=extras=complete=features_ok=valid_values=total_values=exact_iq=0
    onset=[]; ending=[]; leading=[]
    for trial in range(trials):
        rng=np.random.default_rng(240100+cell_id*1000+trial)
        start=3*profile.window_size+offset; end=start+duration; length=end+3*profile.window_size
        iq=model_noise(rng,(1,length),profile.noise_assumption.endswith('ar1_0.8')).ravel()
        iq[start:end]+=(waveform(rng,duration,kind,frequency,bandwidth)*math.sqrt(1e-4*10**(snr/10))).astype(np.complex64)
        p=profile.create_pipeline(s,policy)
        events=[]
        try:
            for seq,i in enumerate(range(0,length,1024)):
                events.extend(p.process_block(IQBlock(s,seq,i,None,iq[i:i+1024])))
            events.extend(p.flush())
        finally: p.shutdown()
        # Match the original compatibility evaluator's DETECTED signal extent.
        # Capture pre-trigger padding must never rescue a late/missed decision.
        intervals=[(e.event.detected_start_sample,e.event.detected_end_sample) for e in events]
        pairs,missing,extra=match_intervals([(start,end)],intervals)
        hits+=bool(pairs);extras+=len(extra)
        for e in events:
            ev=e.event
            assert ev.iq.tobytes()==iq[ev.event_start_sample:ev.event_end_sample].tobytes()
            exact_iq+=1
            assert e.detection_profile_id==profile.profile_id
            assert e.boundary_uncertainty_samples==profile.window_size
            complete+=ev.complete
            features_ok+=e.feature_status=='ok'
            if e.features:
                vals=list(e.features.features.values())
                total_values+=len(vals);valid_values+=sum(v.valid for v in vals)
        if pairs:
            e=events[pairs[0][1]]
            onset.append(e.event.detected_start_sample-start)
            ending.append(e.event.detected_end_sample-end)
            leading.append(e.actual_leading_samples)
    lower=float(beta.ppf(.05,hits,trials-hits+1)) if hits else 0.
    quant=lambda a: np.percentile(a,[0,50,95,100]).tolist() if a else []
    return dict(duration_samples=duration,onset_hop_offset=offset,modulation=kind,frequency_hz=frequency,
        bandwidth_hz=bandwidth,snr_db=snr,independent_trials=trials,matched=hits,missed=trials-hits,
        extras=extras,pd_lower_one_sided_95=lower,pd_gate=trials>=400 and lower>=.9,
        onset_error_samples_min_p50_p95_max=quant(onset),end_error_samples_min_p50_p95_max=quant(ending),
        retained_pretrigger_samples_min_p50_p95_max=quant(leading),complete=complete,exact_iq_checks=exact_iq,
        feature_status_ok=features_ok,feature_values_valid=valid_values,feature_values_total=total_values)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--trials',type=int,default=400)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--colored',action='store_true')
    args=parser.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    profile=LONG_BURST_AR1 if args.colored else LONG_BURST_IID
    # Persist the plan BEFORE observing held-out or detection results.
    shapes=[('cw',0.,0.),('cw',200000.,0.),('cw',-200000.,0.),
            ('bandlimited_noise',30000.,10000.),('bandlimited_noise',-100000.,100000.)]
    cells=[(d,o,*s,-10.) for d in (4096,8192) for o in (0,255) for s in shapes]
    characterization=[(d,127,*s,snr) for d,snr in ((2048,-10.),(4096,-15.),(8192,-15.)) for s in shapes]
    if args.colored:
        cells=[(d,o,*s,-10.) for d in (32768,65536) for o in (0,1023) for s in shapes]
        characterization=[(d,511,*s,snr) for d,snr in ((4096,-10.),(16384,-10.),(32768,-15.)) for s in shapes]
    (out/'plan.json').write_text(json.dumps(dict(profile=asdict(profile),profile_id=profile.profile_id,
        supported_candidate_cells=cells,characterized_cells=characterization,trials=args.trials,
        matching_iou=.2,matching_extent='detected_signal',lower_confidence=.95,required_lower_pd=.9,noise=profile.noise_assumption,
        training_seed=240001,validation_seed=240002),indent=2))
    policy,validation=calibrate(profile)
    (out/'policy.json').write_text(json.dumps(policy.to_dict(),indent=2))
    (out/'held_out.json').write_text(json.dumps(validation,indent=2))
    print('held_out',policy.supported_pfa,policy.pfa_upper_on,policy.pfa_upper_off,flush=True)
    if not policy.supported_pfa: return 1
    rows=[]
    selected=cells[:1] if args.smoke else cells+characterization
    for cell_id,cell in enumerate(selected):
        row=evaluate_cell(policy,*cell,args.trials,cell_id,profile)
        row['role']='candidate' if cell_id<len(cells) else 'characterization'
        rows.append(row)
        (out/'cells.json').write_text(json.dumps(rows,indent=2))
        print('cell',cell_id,json.dumps(row),flush=True)
    passed=not args.smoke and len(rows)==len(cells)+len(characterization) and all(r['pd_gate'] for r in rows if r['role']=='candidate')
    summary=dict(profile_candidate_passed=passed,profile_id=profile.profile_id,
        release_detection_gate='requires both profile runs and final installed verification',
        seconds=time.perf_counter()-started,source=__import__('signal_processing').__file__,
        ideal_conditional_pd={str(n):float(ncx2.sf(chi2.isf(1e-4,2*n),2*n,2*n*.1)) for n in (64,2048,4096)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return int(not passed)


if __name__=='__main__':
    raise SystemExit(main())
