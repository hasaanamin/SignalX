"""Fresh runtime and reliability checks for examples(2).zip; no implementation edits."""
from pathlib import Path
import os,sys,json,math,dataclasses,threading,time,tracemalloc
import numpy as np
from scipy import signal,linalg
SRC=Path(os.environ.get('SIGNAL_PROCESSING_SRC',Path(__file__).resolve().parents[1]/'')).resolve()
os.environ['SIGNAL_PROCESSING_SRC']=str(SRC);sys.path.insert(0,str(SRC))
import signal_processing2_deep_probes as h
import signal_processing2_extended_probes as ex
import new_mechanism_probes as n
from signal_processing.detection import EnergyDetector,GaussianNoiseEstimator,AdaptiveRobustNoiseEstimator,EmpiricalNoiseCalibrator
from signal_processing.features.bandwidth import BandwidthFeatureExtractor,BandwidthFeatureConfig
from signal_processing.features import FeatureAggregator
from signal_processing.features.aggregation import FeatureRegistry
from signal_processing.runtime import FeatureAnalysisWorker
from signal_processing.core import NoiseEstimate,StreamInfo,IQBlock

def retained_history():
    p=h.p(window_size=256,hop_size=64,pre_trigger_samples=0,post_signal_samples=0)
    seq=0;start=0;rows=[];tracemalloc.start()
    for target in (65536,131072,262144):
        while start<target:
            x=np.full(min(1024,target-start),.003,np.complex64)
            p.process_block(h.b(x,seq,start));start+=len(x);seq+=1
        current,peak=tracemalloc.get_traced_memory()
        rows.append({'received':start,'retained_hops':len(p._pending_noise_iq),'retained_iq_samples':sum(len(x) for x in p._pending_noise_iq),'active_pending_samples':len(p._pending_noise_buf),'ring_capacity':p.ring_buffer.capacity,'traced_current_bytes':current,'traced_peak_bytes':peak})
    tracemalloc.stop();p.flush();p.shutdown();return rows

def origin_causality():
    rows=[]
    for origin in (0,1000):
        x=np.full(64,.003,np.complex64);x[17:]=.1
        est=AdaptiveRobustNoiseEstimator(window_size=17,initial_noise_power_linear=1e-4,adaptation_window_size=1,time_constant_s=1e-6)
        p=h.p(window_size=17,hop_size=8,noise_estimator=est,pre_trigger_samples=0,post_signal_samples=0)
        windows=[];admitted=[];old=p.detector.process_windows;update=est.update_samples
        def measure(b,ref):
            ms=old(b,ref);windows.extend({'end':m.end_sample,'on':m.above_on_threshold,'reference':m.noise_power_dbfs} for m in ms);return ms
        def consume(x):admitted.append({'end':windows[-1]['end'],'count':len(x),'active_samples':int(np.count_nonzero(abs(x)>.01))});return update(x)
        p.detector.process_windows=measure;est.update_samples=consume
        ev=p.process_block(h.b(x,0,origin));ev+=p.flush();p.shutdown()
        rows.append({'origin':origin,'events':[h.summ(s) for s in ev],'admitted':admitted,'measurements':windows,'final_reference':est.noise_power_linear,'admission_counter':p._noise_admission_sample})
    return rows

def gap_complete_window():
    rows=[]
    for end in ('EOF','gap'):
        p=h.p(window_size=17,hop_size=8,pre_trigger_samples=0,post_signal_samples=0)
        x=np.full(17,.5,np.complex64);ev=p.process_block(h.b(x))
        if end=='gap':ev+=p.process_block(h.b(np.zeros(32,np.complex64),1,49,dropped_samples_before=32))
        ev+=p.flush();p.shutdown();rows.append({'termination':end,'events':[h.summ(s) for s in ev]})
    return rows

def startup_age_before_flush():
    est=AdaptiveRobustNoiseEstimator(window_size=256,adaptation_window_size=64,initial_noise_power_linear=1e-4,max_staleness_samples=16)
    p=h.p(window_size=256,hop_size=64,noise_estimator=est,pre_trigger_samples=0)
    error=None
    try:p.process_block(h.b(np.full(192,.003,np.complex64)))
    except Exception as e:error=type(e).__name__
    before={'error':error,'age':est.staleness_samples,'estimator_state':est.state.value,'pipeline_state':p.state.value,'pending':len(p._pending_noise_buf),'ring':p.ring_buffer.available_range()}
    p.flush();after={'age':est.staleness_samples,'state':est.state.value};p.shutdown();return {'before_flush':before,'after_flush':after}

def frozen_residual_age():
    e=AdaptiveRobustNoiseEstimator(window_size=16,adaptation_window_size=5,initial_noise_power_linear=1e-4,max_staleness_samples=100)
    e.update_samples(np.full(4,.003,np.complex64));before=e.staleness_samples
    e.freeze(True);e.advance_stream_samples(10);frozen=e.staleness_samples;e.freeze(False)
    e.update_samples(np.full(1,.1,np.complex64))
    return {'age_before':before,'age_frozen':frozen,'age_after_one_post_active_sample':e.staleness_samples,'accepted':e.total_accepted_samples,'reference':e.noise_power_linear,'residual':len(e._adaptation_residual)}

def midblock_policy_failure():
    c=EmpiricalNoiseCalibrator(window_size=16,hop_size=8,pfa_on=.01,pfa_off=.1)
    q=c.create_policy(n.noise(20000),settling_samples=0)
    est=AdaptiveRobustNoiseEstimator(window_size=16,adaptation_window_size=16,initial_noise_power_linear=q.mean_power_linear,time_constant_s=1e-6)
    p=h.p(window_size=16,hop_size=8,pfa_on=.01,pfa_off=.1,threshold_policy=q,noise_estimator=est,pre_trigger_samples=0)
    before=p.ring_buffer.available_range();err=None
    try:p.process_block(h.b(np.full(64,.003,np.complex64)))
    except Exception as e:err=type(e).__name__
    out={'error':err,'state':p.state.value,'ring_before':before,'ring_after':p.ring_buffer.available_range(),'reference':est.noise_power_linear,'previous_end':p._previous_block.end_sample if p._previous_block else None}
    p.shutdown();return out

def calibrated_policy():
    return EmpiricalNoiseCalibrator(window_size=64,hop_size=64).create_policy(n.noise(700000,2608),monitored_band_hz=(90000,110000),settling_samples=4096)

def matched_reference_variability():
    q=calibrated_policy();rows=[]
    for seed in range(20):
        e=GaussianNoiseEstimator(window_size=64,warmup_samples=4096)
        p=h.p(window_size=64,hop_size=64,detector_mode='configured_band',monitored_band_hz=(90000,110000),threshold_policy=q,noise_estimator=e,pre_trigger_samples=0)
        err=None
        try:
            p.process_block(h.b(n.noise(4096,seed)));p.process_block(h.b(n.noise(64,100+seed),1,4096))
        except Exception as exn:err=type(exn).__name__
        rows.append({'seed':seed,'reference':e.noise_power_linear,'ratio_to_policy':e.noise_power_linear/q.mean_power_linear,'error':err,'pipeline_state':p.state.value});p.shutdown()
    return {'cases':rows,'rejected_same_domain':sum(r['error'] is not None for r in rows),'total':len(rows)}

def effective_gamma_tails():
    # Independent stationary complex-Gaussian windows generated from the exact
    # covariance eigenvalues. No filtered-window overlap or saved observations.
    cases=[]
    for N,band in [(64,(90000,110000)),(256,(90000,110000)),(64,(99500,100500))]:
        d=EnergyDetector(window_size=N,hop_size=N,mode='configured_band',monitored_band_hz=band)
        # Impulse horizon longer than the detector's capped approximation; convergence is recorded.
        impulse=np.zeros(65536);impulse[0]=1;hh=signal.sosfilt(d._sos,impulse)
        fftn=1<<int((2*len(hh)-1).bit_length());spec=np.fft.rfft(hh,n=fftn)
        autocorr=np.fft.irfft(spec*np.conj(spec),n=fftn)[:N]
        cov=linalg.toeplitz(autocorr/autocorr[0]);vals=np.linalg.eigvalsh(cov)/N;vals=np.maximum(vals,0)
        keep=vals>1e-12;weights=vals[keep];threshold=d._unit_q_on
        rng=np.random.default_rng(1000+N+int(band[0]));trials=1000000;hits=0
        for start in range(0,trials,20000):
            powers=np.sum(rng.exponential(size=(min(20000,trials-start),len(weights)))*weights,axis=1)
            hits+=int(np.count_nonzero(powers>=threshold))
        from scipy.stats import beta
        lower=float(beta.ppf(.025,hits,trials-hits+1)) if hits else 0.
        upper=float(beta.ppf(.975,hits+1,trials-hits)) if hits<trials else 1.
        cases.append({'N':N,'band':band,'model_n_eff':d._n_eff,'covariance_n_eff':float(1/np.sum(vals**2)),'threshold_over_true_mean':threshold,'independent_windows':trials,'hits':hits,'rate':hits/trials,'95pct_binomial_interval':[lower,upper],'target':1e-4,'retained_eigenvalues':len(weights),'omitted_normalized_energy':float(vals[~keep].sum()),'impulse_last_1024_energy_fraction':float(np.sum(hh[-1024:]**2)/np.sum(hh**2))})
    return cases

def configured_actual_N_and_large_N():
    rows=[]
    for N in (256,8192):
        try:
            d=EnergyDetector(window_size=N,hop_size=min(N,64),mode='configured_band',monitored_band_hz=(90000,110000))
            m=d.process(h.b(np.ones(1,np.complex64)),NoiseEstimate(h.S.stream_id,0,0))
            rows.append({'configured_N':N,'accepted':True,'actual_N':1,'unit_threshold':10**(m.on_threshold_dbfs/10),'correct_single_sample_exponential_threshold':-math.log(1e-4)})
        except Exception as e:rows.append({'configured_N':N,'accepted':False,'error':type(e).__name__,'message':str(e)})
    return rows

def callback_reset_between_sinks():
    w=FeatureAnalysisWorker(synchronous=True);calls=[]
    def first(sc):calls.append({'sink':'first','id':sc.event.event_id,'generation':w.current_generation});w.reset()
    def second(sc):calls.append({'sink':'second','id':sc.event.event_id,'generation':w.current_generation})
    w.add_result_sink(first);w.add_result_sink(second);w.submit(h.sidecar());w.shutdown()
    return {'callbacks':calls,'generation':w.current_generation,'discarded':w.discarded_old_generation_count}

def overwritten_capture_epoch():
    sc=dataclasses.replace(h.sidecar(),stream_epoch=42);w=FeatureAnalysisWorker(synchronous=True)
    r=w.submit(sc);w.shutdown();return {'input_epoch':sc.stream_epoch,'result_epoch':r.stream_epoch}

def cancel_second_call_liveness():
    w=FeatureAnalysisWorker(synchronous=False,max_queue_size=4);entered=threading.Event();release=threading.Event();processed=threading.Event();orig=w._process_one
    def gate(sc):entered.set();release.wait(2);r=orig(sc);processed.set();return r
    w._process_one=gate;w.submit(h.sidecar());assert entered.wait(1)
    errors=[]
    try:w.shutdown(drain=True,timeout=.01)
    except Exception as e:errors.append(type(e).__name__)
    cancel=w.cancel
    def cancel_release():cancel();release.set()
    w.cancel=cancel_release
    try:w.shutdown(drain=False,timeout=.2)
    except Exception as e:errors.append(type(e).__name__)
    before={'errors':errors,'job_processed':processed.is_set(),'state':w.state.value,'thread_alive':w._thread.is_alive(),'queue':w._queue.qsize(),'unfinished':w._queue.unfinished_tasks,'sentinel_flag':w._sentinel_enqueued}
    w.shutdown(timeout=1);return before

def support_101_windows():
    c=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
    q=c.create_policy(n.noise(1616),settling_samples=0)
    return {'windows':q.training_windows,'supported':q.supported_pfa,'valid':q.valid,'fields':[f.name for f in dataclasses.fields(q)],'zero_hit_95pct_upper':1-.05**(1/101)}

def finalize_failure_recovery():
    c=EmpiricalNoiseCalibrator(window_size=16,max_samples=1000);c.begin_calibration();c.update_calibration(n.noise(8));errs=[]
    try:c.finalize_calibration()
    except Exception as e:errs.append(type(e).__name__)
    try:c.update_calibration(n.noise(8,33))
    except Exception as e:errs.append(type(e).__name__)
    return {'errors':errs,'sealed':c._finalized,'samples':c._cal_samples_collected}

def bandwidth_one_extra_bin():
    f=np.arange(-1000,1001,dtype=float);excess=100/(1+f*f);mask=abs(f)<=12
    cfg=BandwidthFeatureConfig(mode='emitter',roi_hz=(-11,11),noise_floor_dbfs_per_hz=0)
    result=BandwidthFeatureExtractor(cfg).extract(ex.SpectralContext(f[mask],1+excess[mask]))
    return {'unobserved_known_energy_fraction':float(excess[~mask].sum()/excess.sum()),'observed_outside_ROI_fraction':float(excess[(abs(f)>11)&mask].sum()/excess[mask].sum()),'features':{k:{'value':v.value,'valid':v.valid,'reason':v.reason,'interpretation':v.interpretation} for k,v in result.features.items()}}

def feature_policy_remaining_collisions():
    N=4096;rng=np.random.default_rng(6);rows=[]
    def make_context(two):
        x=.1*np.exp(2j*np.pi*.0625*np.arange(N))
        if two:x+=.09*np.exp(-2j*np.pi*.0625*np.arange(N))
        x+=(rng.normal(size=N)+1j*rng.normal(size=N))*.001
        return h.SignalContext(h.SignalEvent('policy',h.S,x.astype(np.complex64),0,N,0,N,0),h.AnalysisConfig(psd_mode='welch',psd_nperseg=256))
    for label,ctx,configs in [('multimodal',make_context(True),[BandwidthFeatureConfig(mode='emitter',noise_floor_dbfs_per_hz=-100,check_edge_clipping=False,multimodal_ratio_threshold=t) for t in (.7,.9)]),('auto_noise',make_context(False),[BandwidthFeatureConfig(mode='emitter',roi_hz=(-80000,80000),auto_noise_estimation=a) for a in (True,False)])]:
        values=[]
        for cfg in configs:
            out=FeatureAggregator(FeatureRegistry([BandwidthFeatureExtractor(cfg)])).extract_analysis_result(ctx)
            values.append({'config':dataclasses.asdict(cfg),'policy_id':out.policy.policy_id,'features':{k:{'value':v.value,'valid':v.valid,'reason':v.reason,'interpretation':v.interpretation} for k,v in out.feature_vector.features.items()}})
        rows.append({'case':label,'results':values})
    return rows

def mutable_versions_record():
    N=64;x=np.exp(2j*np.pi*.1*np.arange(N)).astype(np.complex64)
    ctx=h.SignalContext(h.SignalEvent('v',h.S,x,0,N,0,N,0));out=h.FeatureAggregator(h.build_default_feature_registry()).extract_analysis_result(ctx)
    before=dict(out.policy.extractor_versions);pid=out.policy.policy_id
    try:out.policy.extractor_versions['amplitude']='invented';allowed=True
    except Exception:allowed=False
    return {'mutation_allowed':allowed,'policy_id_unchanged':out.policy.policy_id==pid,'versions_before':before,'versions_after':dict(out.policy.extractor_versions)}

def half_db_policy_contradiction():
    q=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1).create_policy(n.noise(20000),settling_samples=0)
    try:r=dataclasses.replace(q,t_on_dbfs=q.t_on_dbfs+.49,t_off_dbfs=q.t_off_dbfs+.49);ok=True
    except Exception:ok=False
    return {'accepted':ok,'power_discrepancy_fraction':10**(.49/10)-1}

def quality_calibration_trust():
    rows=[]
    for trust in (True,False):
        sc=dataclasses.replace(h.sidecar(),noise_valid=trust);w=FeatureAnalysisWorker(synchronous=True,skip_incomplete_quality=False)
        r=w.submit(sc);w.shutdown();rows.append({'noise_valid':trust,'status':r.feature_status,'quality':dataclasses.asdict(r.analysis_result.capture_quality)})
    return rows

def wrong_input_Fs():
    d=EnergyDetector(window_size=64,hop_size=64,mode='configured_band',monitored_band_hz=(90000,110000),sample_rate_hz=1e6)
    other=StreamInfo(h.S.stream_id,h.S.source_id,2e6)
    tone=(.5*np.exp(2j*np.pi*200000*np.arange(256)/other.sample_rate_hz)).astype(np.complex64)
    hits=0
    try:
        ms=d.process_windows(IQBlock(other,0,0,None,tone),NoiseEstimate(other.stream_id,0,-40));ok=True;hits=sum(m.above_on_threshold for m in ms)
    except Exception:ok=False
    return {'configured_Fs':1e6,'input_Fs':2e6,'physical_tone_Hz':200000,'configured_band_Hz':(90000,110000),'accepted':ok,'ON_windows':hits}

def other_buffer_entrypoints():
    from signal_processing.buffering import CircularIQBuffer
    rows=[]
    for value in (-1,True,1.5,float('nan')):
        buf=CircularIQBuffer(16);before=buf.stream_generation
        try:buf.reset(value);ok=True;err=None
        except Exception as e:ok=False;err=type(e).__name__
        rows.append({'API':'reset','value':repr(value),'accepted':ok,'error':err,'generation_before':before,'generation_after':buf.stream_generation,'range_after':buf.available_range()})
    buf=CircularIQBuffer(16);buf.write(0,np.arange(4,dtype=np.complex64))
    try:x=buf.read_range(.9,2.9);rows.append({'API':'read_range','accepted':True,'actual_samples':x.real.tolist()})
    except Exception as e:rows.append({'API':'read_range','accepted':False,'error':type(e).__name__})
    return rows

PROBES=[retained_history,origin_causality,gap_complete_window,startup_age_before_flush,frozen_residual_age,midblock_policy_failure,matched_reference_variability,effective_gamma_tails,configured_actual_N_and_large_N,callback_reset_between_sinks,overwritten_capture_epoch,cancel_second_call_liveness,support_101_windows,finalize_failure_recovery,bandwidth_one_extra_bin,feature_policy_remaining_collisions,mutable_versions_record,half_db_policy_contradiction,quality_calibration_trust,wrong_input_Fs,other_buffer_entrypoints]
if __name__=='__main__':
    out={'source':str(SRC)}
    for fn in PROBES:
        try:out[fn.__name__]={'executed':True,'observed':fn()}
        except Exception as e:out[fn.__name__]={'executed':False,'error':type(e).__name__,'message':str(e)}
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=True));print(json.dumps(out,indent=2,allow_nan=True))
