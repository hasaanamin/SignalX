"""Additional public-path checks; no modification of the reviewed package."""
from pathlib import Path
import sys, json, math, dataclasses, hashlib, time, threading, tracemalloc
import numpy as np
from scipy import signal
import signal_processing2_deep_probes as h
from signal_processing.detection import AdaptiveRobustNoiseEstimator,GaussianNoiseEstimator,EmpiricalNoiseCalibrator,EnergyDetector
from signal_processing.detection.noise_estimation import ThresholdPolicy
from signal_processing.features.bandwidth import BandwidthFeatureExtractor,BandwidthFeatureConfig
from signal_processing.core import PSDResult,StreamInfo,NoiseEstimate
ROOT=Path(__file__).parent
RESULTS={}
def record(name,fn):
    if __name__ != '__main__':return
    try:RESULTS[name]={'executed':True,'observed':fn()}
    except Exception as e:RESULTS[name]={'executed':False,'exception':type(e).__name__,'message':str(e)}
def misaligned_decisions():
    rows=[]
    found=None
    for m in (5,11,64):
        for amp in (.0045,.005,.006,.007):
            x=np.full(512,.003,np.complex64);x[257:321]=amp
            baseline=None
            for bs in (8,7,11,31):
                est=AdaptiveRobustNoiseEstimator(window_size=16,initial_noise_power_linear=1e-4,time_constant_s=1e-5,adaptation_window_size=m)
                r=h.run(x,bs,noise_estimator=est,pre_trigger_samples=0,post_signal_samples=0)
                if baseline is None:baseline=r
                timeline=r.pop('measurements')
                base_ms=baseline.get('measurements',timeline) if r is not baseline else timeline
                if r is baseline:baseline['measurements']=timeline
                mismatches=sum(a[2:]!=b[2:] for a,b in zip(base_ms,timeline))
                max_diff=max(abs(a[1]-b[1]) for a,b in zip(base_ms,timeline))
                detail={'M':m,'amplitude':amp,'block':bs,'decision_mismatches_vs_block8':mismatches,
                        'maximum_reference_db_difference':max_diff,'events':r['events'],'accepted':r['accepted'],'reference':r['reference']}
                rows.append(detail)
                if found is None and (mismatches or r['events']!=baseline['events']):found=detail
    return {'first_counterexample':found,'cases':rows}
record('misaligned_changes_decisions',misaligned_decisions)
def no_update_age():
    rows=[]
    for bs in (1,16,128):
        est=AdaptiveRobustNoiseEstimator(window_size=16,adaptation_window_size=16,initial_noise_power_linear=1e-4,max_staleness_samples=16)
        r=h.run(np.zeros(128,np.complex64),bs,noise_estimator=est);r.pop('measurements');rows.append(r)
    return rows
record('zero_noise_no_successful_update_age',no_update_age)
def timing_bound():
    x=np.zeros(1000,np.complex64);x[500:700]=.5
    r=h.run(x,64);e=r['events'][0]
    return {'true_onset':500,'true_end':700,'event':e,'onset_error':abs(e['ds']-500),'end_error':abs(e['de']-700)}
record('physical_boundary_error_exceeds_N_claim',timing_bound)
def warmup_gap_signal():
    est=GaussianNoiseEstimator(window_size=16,warmup_samples=32)
    pipe=h.p(noise_estimator=est);pipe.process_block(h.b(np.full(32,.01,np.complex64)))
    before=est.noise_power_linear
    ev=pipe.process_block(h.b(np.full(512,.5,np.complex64),1,64,dropped_samples_before=32))
    ev+=pipe.flush();pipe.shutdown()
    return {'first_noise_reference':before,'new_reference_after_gap_signal':est.noise_power_linear,
            'samples_retrained':est.samples_trained,'events':[h.summ(q) for q in ev]}
record('gap_silently_reauthorizes_signal_warmup',warmup_gap_signal)
class SpectralContext:
    def __init__(self,f,p):self.f=f;self.p=p
    def psd(self):return PSDResult(self.f,self.p,np.zeros_like(self.p))
def width_cases():
    f=np.arange(-10,11,dtype=float);p=1+100*np.exp(-f*f/8)
    cfg=BandwidthFeatureConfig(mode='emitter',roi_hz=(-5,5),noise_floor_dbfs_per_hz=0)
    widths=BandwidthFeatureExtractor(cfg).extract(SpectralContext(f,p))
    frac=float(np.sum((p-1)[abs(f)>5])/np.sum(p-1))
    ffull=np.arange(-1000,1001,dtype=float);efull=100/(1+ffull**2)
    mask=abs(ffull)<=11;ctx=SpectralContext(ffull[mask],1+efull[mask])
    widths2=BandwidthFeatureExtractor(BandwidthFeatureConfig(mode='emitter',noise_floor_dbfs_per_hz=0)).extract(ctx)
    def describe(fs):return {k:{'valid':v.valid,'value':v.value,'reason':v.reason} for k,v in fs.features.items()}
    return {'contained_gaussian':{'known_energy_outside_ROI_fraction':frac,'features':describe(widths)},
            'clipped_heavy_tails':{'known_energy_outside_observation_fraction':float(efull[~mask].sum()/efull.sum()),
              'boundary_peak_ratio':float(efull[mask][0]/efull.max()),'features':describe(widths2)}}
record('bandwidth_edge_heuristic_counterexamples',width_cases)
def direct_policy_invariants():
    c=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
    rng=np.random.default_rng(1);x=((rng.normal(size=20000)+1j*rng.normal(size=20000))*.01).astype(np.complex64)
    policy=c.create_policy(x)
    rows=[]
    replacements=[
        ('inconsistent_dbfs',{'t_on_dbfs':100.0,'t_off_dbfs':90.0}),
        ('hop_exceeds_window',{'hop_size':17}),
        ('valid_flag_string',{'valid':'yes'}),
        ('mutable_invalid_band',{'monitored_band_hz':[500000,600000]}),
    ]
    for name,changes in replacements:
        try:
            q=dataclasses.replace(policy,**changes);rows.append({'case':name,'accepted':True,
                't_on_dbfs':q.t_on_dbfs,'t_on_linear':q.t_on_linear,'hop':q.hop_size,'valid':q.valid,
                'band':q.monitored_band_hz})
        except Exception as e:rows.append({'case':name,'accepted':False,'error':type(e).__name__})
    return rows
record('threshold_policy_remaining_invariants',direct_policy_invariants)
def invalid_calibration_inputs():
    rows=[]
    inputs=[('real_input',np.ones(20000,dtype=float)),
            ('nonfinite_input',np.full(20000,np.nan+0j,np.complex64))]
    for label,x in inputs:
        c=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
        c.begin_calibration()
        try:c.update_calibration(x);rows.append({'case':label,'accepted':True,'committed_samples':c._cal_samples_collected,
                'committed_windows':len(c._cal_window_powers)})
        except Exception as e:rows.append({'case':label,'accepted':False,'error':type(e).__name__})
    return rows
record('calibration_input_validation_before_mutation',invalid_calibration_inputs)
def sampling_support():
    c=EmpiricalNoiseCalibrator(window_size=16,hop_size=1,pfa_on=.01,pfa_off=.1)
    rng=np.random.default_rng(888);x=((rng.normal(size=115)+1j*rng.normal(size=115))*.01).astype(np.complex64)
    d=c.calibrate(x);q=c.create_policy(d)
    return {'samples':d['num_samples'],'overlapping_windows':d['num_windows'],'support_flag':q.supported_pfa,'valid':q.valid,
            'declared_pfa':q.pfa_on,'record_fields':list(d),'policy_fields':[f.name for f in dataclasses.fields(q)]}
record('support_from_100_dependent_windows',sampling_support)
def controlled_worker_admission():
    w=h.FeatureAnalysisWorker(synchronous=False,max_queue_size=2)
    admitted=threading.Event();release=threading.Event();stop_done=threading.Event();errors=[]
    original_put=w._queue.put_nowait
    def gated_put(item):admitted.set();release.wait(2);return original_put(item)
    w._queue.put_nowait=gated_put
    def submit():
        try:w.submit(h.sidecar())
        except Exception as e:errors.append(type(e).__name__)
    def stop():
        try:w.shutdown(timeout=1)
        except Exception as e:errors.append(type(e).__name__)
        finally:stop_done.set()
    t1=threading.Thread(target=submit);t1.start();assert admitted.wait(2)
    t2=threading.Thread(target=stop);t2.start();time.sleep(.025)
    premature=stop_done.is_set();release.set();t1.join(2);t2.join(2)
    return {'shutdown_completed_before_admission_release':premature,'errors':errors,'completed':w.completed_count,
            'unfinished':w._queue.unfinished_tasks,'queued':w._queue.qsize(),'state':w.state.value}
record('preserved_atomic_worker_admission',controlled_worker_admission)
def input_output_overload():
    w=h.FeatureAnalysisWorker(synchronous=False,max_queue_size=1,max_results=1)
    entered=threading.Event();release=threading.Event();orig=w._process_one
    def gate(sc):entered.set();release.wait(2);return orig(sc)
    w._process_one=gate
    w.submit(h.sidecar());assert entered.wait(2);w.submit(h.sidecar());r=w.submit(h.sidecar())
    release.set();w.shutdown()
    result={'input_drop':w.dropped_overload_count,'output_drop':w.dropped_result_overload_count,'completed':w.completed_count,
            'history_size':len(w.processed_results),'result_queue':w._result_queue.qsize(),'unfinished':w._queue.unfinished_tasks}
    return result
record('positive_input_output_overload_bounded',input_output_overload)
def configured_gaussian_warmup():
    rng=np.random.default_rng(2608)
    train=((rng.normal(size=700000)+1j*rng.normal(size=700000))*math.sqrt(1e-4/2)).astype(np.complex64)
    c=EmpiricalNoiseCalibrator(window_size=64,hop_size=64)
    policy=c.create_policy(train,monitored_band_hz=(90000,110000),settling_samples=4096)
    rng=np.random.default_rng(919)
    quiet=((rng.normal(size=4096)+1j*rng.normal(size=4096))*math.sqrt(1e-4/2)).astype(np.complex64)
    tone=(.01*np.exp(2j*np.pi*100000*np.arange(8192)/1e6)).astype(np.complex64)
    rows=[]
    for label,est in [('matched_fixed',GaussianNoiseEstimator(window_size=64,calibrated_noise_power_linear=policy.mean_power_linear)),
                      ('raw_warmup',GaussianNoiseEstimator(window_size=64,warmup_samples=4096))]:
        pipe=h.p(window_size=64,hop_size=64,detector_mode='configured_band',monitored_band_hz=(90000,110000),
                 threshold_policy=policy,noise_estimator=est,pre_trigger_samples=0,post_signal_samples=0)
        events=pipe.process_block(h.b(quiet));reference_after_prefix=est.noise_power_linear
        events+=pipe.process_block(h.b(tone,1,4096));events+=pipe.flush();pipe.shutdown()
        rows.append({'route':label,'reference_after_prefix':reference_after_prefix,'matched_policy_reference':policy.mean_power_linear,
                     'events':[h.summ(q) for q in events]})
    return rows
record('configured_gaussian_warmup_reference',configured_gaussian_warmup)
def gaussian_recalibration_no_budget():
    g=GaussianNoiseEstimator(window_size=16,calibrated_noise_power_linear=1e-4)
    g.recalibrate()
    consumed=g.update_samples(np.full(128,.01,np.complex64))
    return {'state':g.state.value,'target':g.target_warmup_samples,'trained':g.samples_trained,'consumed':consumed}
record('zero_budget_gaussian_recalibration',gaussian_recalibration_no_budget)
if __name__=='__main__':
    Path(__file__).with_suffix('.json').write_text(json.dumps(RESULTS,indent=2))
    print(json.dumps(RESULTS,indent=2))

