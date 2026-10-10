"""Independent positive-SNR burst opportunities, causal adaptive references.

This is a declared synthetic iid complex-Gaussian envelope, not certification of
arbitrary receiver noise. Calibration is a separate preceding quiet acquisition;
runtime receives its measured mean, never trial truth or future quiet samples.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def qualify(trials=400):
    import numpy as np
    from scipy import signal
    from scipy.stats import beta
    from signal_processing.core import StreamInfo, IQBlock
    from signal_processing.detection import POSITIVE_SNR_IID
    from signal_processing.runtime import StreamingCapturePipeline, FeatureAnalysisWorker
    from benchmarks.sustained_evaluation import match_intervals
    stream=StreamInfo('positive-snr','synthetic',1e6)
    rows=[]
    worker=FeatureAnalysisWorker()
    try:
        for duration in (512,2048):
            for kind,frequency in (('cw',0.),('qpsk',-200000.),('noise',200000.)):
                for snr in (0.,3.,10.):
                    hits=extras=exact=features=complete=leading=0
                    quality={}
                    onset=[];ending=[]
                    for trial in range(trials):
                        rng=np.random.default_rng(710000+len(rows)*10000+trial)
                        calibration=((rng.normal(size=8192)+1j*rng.normal(size=8192))*math.sqrt(1e-4/2)).astype(np.complex64)
                        mean=math.fsum(float(z.real)**2+float(z.imag)**2 for z in calibration)/len(calibration)
                        count=duration+3072
                        iq=((rng.normal(size=count)+1j*rng.normal(size=count))*math.sqrt(1e-4/2)).astype(np.complex64)
                        start=1024+(trial*17)%64;end=start+duration
                        if kind=='cw': wave=np.ones(duration,complex)
                        elif kind=='qpsk':
                            symbols=np.exp(1j*(np.pi/4+np.pi/2*rng.integers(0,4,size=(duration+7)//8)))
                            wave=np.repeat(symbols,8)[:duration]
                        else:
                            raw=rng.normal(size=duration+512)+1j*rng.normal(size=duration+512)
                            wave=signal.sosfilt(signal.butter(4,.2,output='sos'),raw)[512:]
                            wave/=math.sqrt(float(np.mean(abs(wave)**2)))
                        wave*=np.exp(2j*np.pi*frequency*np.arange(duration)/1e6)
                        iq[start:end]+=(wave*math.sqrt(1e-4*10**(snr/10))).astype(np.complex64)
                        pipe=POSITIVE_SNR_IID.create_pipeline(stream,mean,enable_feature_analysis=False)
                        events=[]
                        try:
                            for seq,a in enumerate(range(0,count,1024)):
                                events+=pipe.process_block(IQBlock(stream,seq,a,None,iq[a:a+1024]))
                            events+=pipe.flush()
                        finally:pipe.shutdown()
                        spans=[(s.event.detected_start_sample,s.event.detected_end_sample) for s in events]
                        pairs,missed,false=match_intervals([(start,end)],spans)
                        hits+=bool(pairs);extras+=len(false)
                        for sc in events:
                            ev=sc.event
                            assert ev.iq.tobytes()==iq[ev.event_start_sample:ev.event_end_sample].tobytes()
                            assert not ev.iq.flags.writeable
                            exact+=1
                        for _,j in pairs:
                            sc=events[j]
                            complete+=sc.event.complete
                            leading+=sc.actual_leading_samples>=256
                            onset.append(spans[j][0]-start);ending.append(spans[j][1]-end)
                            result=worker.submit(sc)
                            assert result.feature_status=='ok' and len(result.features.features)==56
                            features+=1
                            for key,value in result.features.features.items():
                                if not value.valid:quality[key]=quality.get(key,0)+1
                    lower=float(beta.ppf(.05,hits,trials-hits+1)) if hits else 0.
                    row=dict(duration_samples=duration,kind=kind,frequency_hz=frequency,snr_db=snr,
                        independent_opportunities=trials,matched=hits,missed=trials-hits,extras=extras,
                        pd_one_sided_95_lower=lower,pd_passed=lower>=.9,exact_iq_checks=exact,
                        complete_matched=complete,pretrigger_retained_matched=leading,analysed=features,
                        invalid_feature_counts=quality,
                        onset_error_percentiles=np.percentile(onset,[0,50,95,100]).tolist() if onset else [],
                        end_error_percentiles=np.percentile(ending,[0,50,95,100]).tolist() if ending else [])
                    rows.append(row);print(json.dumps(row),flush=True)
    finally:worker.shutdown()
    return dict(profile=dict(profile_id=POSITIVE_SNR_IID.profile_id,fs=1e6,n=256,h=64,pretrigger=256,ring=4096,pfa_on=1e-4,pfa_off=1e-2,
        adaptation_window_size=POSITIVE_SNR_IID.adaptation_window_size,time_constant_s=POSITIVE_SNR_IID.time_constant_s,
        noise='iid circular complex Gaussian, independently measured preceding calibration',
        reference='causal adaptive robust; model-based thresholds, not held-out fixed-threshold certification',
        minimum_duration_samples=512,matched_extent='detected signal interval, IoU >= 0.2, NOT padded capture',
        independence='one seeded recording and one burst per independent trial'),rows=rows,
        passed=trials>=400 and all(r['pd_passed'] and r['complete_matched']==r['matched'] and
                   r['pretrigger_retained_matched']==r['matched'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=400);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=qualify(a.trials);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));raise SystemExit(not result['passed'])
