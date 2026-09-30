"""Fresh mechanism probes for pyproject.zip, source is never edited."""
from pathlib import Path
import os,sys,json,math,dataclasses,hashlib,threading,time,importlib.util
import numpy as np
from scipy import signal
SRC=Path(os.environ.get('SIGNAL_PROCESSING_SRC',Path(__file__).resolve().parents[1]/'pyproject_review')).resolve()
os.environ['SIGNAL_PROCESSING_SRC']=str(SRC)
sys.path.insert(0,str(SRC))
import signal_processing2_deep_probes as h
import signal_processing2_extended_probes as ex
from signal_processing.detection import EnergyDetector,GaussianNoiseEstimator,AdaptiveRobustNoiseEstimator,EmpiricalNoiseCalibrator
from signal_processing.core import StreamInfo,IQBlock,NoiseEstimate
from signal_processing.buffering import CircularIQBuffer
from signal_processing.runtime import FeatureAnalysisWorker
from signal_processing.features import FeatureAggregator
from signal_processing.features.aggregation import FeatureRegistry
from signal_processing.features.bandwidth import BandwidthFeatureExtractor,BandwidthFeatureConfig

def noise(n,seed=919,p=1e-4):
    r=np.random.default_rng(seed)
    return ((r.normal(size=n)+1j*r.normal(size=n))*math.sqrt(p/2)).astype(np.complex64)

def nonmultiple_eof():
    rows=[]
    for n,hop in [(15,8),(17,8),(31,16),(64,64)]:
        x=np.full(n,.5,np.complex64)
        r=h.run(x,n,window_size=n,hop_size=hop,pre_trigger_samples=0,post_signal_samples=0)
        d=EnergyDetector(window_size=n,hop_size=hop)
        m=d.process_windows(h.b(x),h.NoiseEstimate(h.S.stream_id,0,-40))
        rows.append({'N':n,'H':hop,'pipeline_events':r['events'],'pipeline_measurements':r['measurements'],'direct_windows':len(m),'direct_ON':sum(a.above_on_threshold for a in m)})
    return rows

def future_admission():
    x=np.full(64,.003,np.complex64);x[17:]=.1
    rows=[]
    for adapt in (False,True):
        est=AdaptiveRobustNoiseEstimator(window_size=17,initial_noise_power_linear=1e-4,adaptation_window_size=1,time_constant_s=1e-6) if adapt else GaussianNoiseEstimator(window_size=17,calibrated_noise_power_linear=1e-4)
        p=h.p(window_size=17,hop_size=8,noise_estimator=est,pre_trigger_samples=0,post_signal_samples=0)
        admitted=[];windows=[]
        original=p.detector.process_windows
        def wrapped(b,ref):
            ms=original(b,ref)
            windows.extend({'start':m.start_sample,'end':m.end_sample,'reference':m.noise_power_dbfs,'on':m.above_on_threshold} for m in ms)
            return ms
        p.detector.process_windows=wrapped
        if adapt:
            upd=est.update_samples
            def checked(a):
                admitted.append({'measurement_horizon':windows[-1]['end'] if windows else None,'samples':len(a),'active_samples':int(np.count_nonzero(abs(a)>.01))});return upd(a)
            est.update_samples=checked
        events=p.process_block(h.b(x));events+=p.flush();p.shutdown()
        rows.append({'adaptive':adapt,'events':[h.summ(s) for s in events],'admitted':admitted,'measurements':windows,'reference':est.noise_power_linear})
    return rows

def quiet_without_successful_update():
    rows=[]
    for n,hop,m,length in [(16,8,64,48),(256,64,64,192),(16,8,5,24)]:
        est=AdaptiveRobustNoiseEstimator(window_size=n,initial_noise_power_linear=1e-4,adaptation_window_size=m,max_staleness_samples=16)
        r=h.run(np.full(length,.003,np.complex64),length,window_size=n,hop_size=hop,noise_estimator=est,pre_trigger_samples=0)
        rows.append({'N':n,'H':hop,'M':m,'samples':length,'accepted':r['accepted'],'age':r['age'],'state':est.state.value,'error':r['error'],'residual':len(est._adaptation_residual),'measurements':len(r['measurements'])})
    return rows

def configured_warmup_partition():
    x=noise(4096);x=np.r_[x,(.01*np.exp(2j*np.pi*100000*np.arange(8192)/1e6)).astype(np.complex64)]
    rows=[]
    for bs in (64,257,len(x)):
        est=GaussianNoiseEstimator(window_size=64,warmup_samples=4096)
        p=h.p(window_size=64,hop_size=64,detector_mode='configured_band',monitored_band_hz=(90000,110000),noise_estimator=est,pre_trigger_samples=0,post_signal_samples=0)
        ms=[];calls=[];orig=p.detector.process_windows;prep=p.detector._prepare_detection_iq
        def wrapped(b,ref):
            a=orig(b,ref);ms.extend((m.start_sample,m.end_sample,m.signal_power_dbfs) for m in a);return a
        def preparing(iq):calls.append(len(iq));return prep(iq)
        p.detector.process_windows=wrapped;p.detector._prepare_detection_iq=preparing
        events=[]
        for seq,start in enumerate(range(0,len(x),bs)):events.extend(p.process_block(h.b(x[start:start+bs],seq,start)))
        events+=p.flush();p.shutdown()
        rows.append({'block':bs,'trained':est.samples_trained,'reference':est.noise_power_linear,'processed_by_measurement_path':sum(calls),'input_samples':len(x),'phase':p.detector._phase_accumulator,'events':[h.summ(s) for s in events],'measurements':ms})
    base=rows[0]['measurements']
    for row in rows:
        row['maximum_power_difference_db']=max(abs(a[2]-b[2]) for a,b in zip(base,row['measurements'])) if row['measurements'] else None
        row['windows']=len(row.pop('measurements'))
    return rows

def configured_warmup_future_poison():
    quiet=noise(4096)
    rows=[]
    for tail_amp in (0.,1.):
        future=(tail_amp*np.exp(2j*np.pi*100000*np.arange(8192)/1e6)).astype(np.complex64)
        x=np.r_[quiet,np.zeros(64,np.complex64),future]
        p=h.p(window_size=64,hop_size=64,detector_mode='configured_band',monitored_band_hz=(90000,110000),noise_estimator=GaussianNoiseEstimator(window_size=64,warmup_samples=4096),pre_trigger_samples=0,post_signal_samples=0)
        seen=[];orig=p.detector.process_windows
        def wrapped(b,ref):
            ms=orig(b,ref);seen.extend((m.end_sample,m.signal_power_dbfs) for m in ms);return ms
        p.detector.process_windows=wrapped
        p.process_block(h.b(x));p.flush();p.shutdown()
        rows.append({'tail_amplitude':tail_amp,'reference':p.noise_estimator.noise_power_linear,'first_measurement':seen[0]})
    return rows

def near_equal_reference_mismatch():
    c=EmpiricalNoiseCalibrator(window_size=64,hop_size=64)
    q=c.create_policy(noise(700000,2608),monitored_band_hz=(90000,110000),settling_samples=4096)
    rows=[]
    for factor in (1.,2.,4.):
        d=EnergyDetector(window_size=64,hop_size=64,mode='configured_band',monitored_band_hz=(90000,110000),threshold_policy=q)
        try:
            ms=d.process_windows(h.b(noise(128)),NoiseEstimate(h.S.stream_id,0,10*math.log10(q.mean_power_linear*factor)))
            rows.append({'factor':factor,'accepted':True,'threshold':ms[0].on_threshold_dbfs})
        except Exception as e:rows.append({'factor':factor,'accepted':False,'error':type(e).__name__})
    return {'reference_record_fields':[f.name for f in dataclasses.fields(NoiseEstimate)],'cases':rows}

def filtered_gamma_false_alarm():
    r=np.random.default_rng(2608)
    train=((r.normal(size=2000000)+1j*r.normal(size=2000000))*math.sqrt(1e-4/2)).astype(np.complex64)
    test=((r.normal(size=2004096)+1j*r.normal(size=2004096))*math.sqrt(1e-4/2)).astype(np.complex64)
    c=EmpiricalNoiseCalibrator(window_size=64,hop_size=64,max_samples=2000000)
    q=c.create_policy(train,monitored_band_hz=(90000,110000),settling_samples=4096)
    rows=[]
    for label,pol in [('implicit_gamma',None),('matched_empirical',q)]:
        d=EnergyDetector(window_size=64,hop_size=64,mode='configured_band',monitored_band_hz=(90000,110000),threshold_policy=pol)
        ms=[m for m in d.process_windows(h.b(test),NoiseEstimate(h.S.stream_id,0,10*math.log10(q.mean_power_linear))) if m.start_sample>=4096]
        hits=sum(m.above_on_threshold for m in ms)
        rows.append({'path':label,'windows':len(ms),'hits':hits,'rate':hits/len(ms),'requested_pfa':1e-4})
    return rows

def bandwidth_roi_bypass():
    f=np.arange(-11,12,dtype=float);p=1+100/(1+f*f)
    rows=[]
    for roi in (None,(-11.,11.),(-1000.,1000.)):
        cfg=BandwidthFeatureConfig(mode='emitter',roi_hz=roi,noise_floor_dbfs_per_hz=0)
        fs=BandwidthFeatureExtractor(cfg).extract(ex.SpectralContext(f,p))
        rows.append({'roi':roi,'features':{k:{'value':v.value,'valid':v.valid,'reason':v.reason} for k,v in fs.features.items()}})
    return rows

def bandwidth_policy_collision():
    n=4096;r=np.random.default_rng(6)
    x=(.1*np.exp(2j*np.pi*.07*np.arange(n))+(r.normal(size=n)+1j*r.normal(size=n))*.004).astype(np.complex64)
    ctx=h.SignalContext(h.SignalEvent('policy',h.S,x,0,n,0,n,0),h.AnalysisConfig(psd_mode='welch',psd_nperseg=256));rows=[]
    for check in (True,False):
        reg=FeatureRegistry([BandwidthFeatureExtractor(BandwidthFeatureConfig(mode='emitter',noise_floor_dbfs_per_hz=-100,check_edge_clipping=check))])
        a=FeatureAggregator(reg).extract_analysis_result(ctx)
        rows.append({'check_edge_clipping':check,'policy_id':a.policy.policy_id,'policy_fields':dataclasses.asdict(a.policy),'features':{k:{'value':v.value,'valid':v.valid,'reason':v.reason} for k,v in a.feature_vector.features.items()}})
    return rows

def calibration_lifecycle_and_shape():
    rows=[]
    for kind in ('matrix','finalize_then_update','negative_frequency_band'):
        c=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1,max_samples=10000)
        c.begin_calibration()
        try:
            if kind=='matrix':c.update_calibration(np.ones((20,2),np.complex64))
            elif kind=='finalize_then_update':
                c.update_calibration(noise(1600));a=c.finalize_calibration();c.update_calibration(noise(1600,111));b=c.finalize_calibration();rows.append({'case':kind,'accepted':True,'first_samples':a['num_samples'],'second_samples':b['num_samples']});continue
            else:c.begin_calibration(monitored_band_hz=(-110000,-90000))
            rows.append({'case':kind,'accepted':True,'samples':c._cal_samples_collected,'windows':len(c._cal_window_powers)})
        except Exception as e:rows.append({'case':kind,'accepted':False,'error':type(e).__name__,'message':str(e),'samples':getattr(c,'_cal_samples_collected',None),'windows':len(getattr(c,'_cal_window_powers',[]))})
    return rows

def support_without_reliability():
    c=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
    q=c.create_policy(noise(1600,22),settling_samples=0)
    return {'valid':q.valid,'supported_pfa':q.supported_pfa,'training_windows':q.training_windows,'policy_fields':[f.name for f in dataclasses.fields(q)],'zero_hit_95pct_upper_for_100_independent':1-.05**(1/100)}

def direct_buffer_contracts():
    rows=[]
    for kind in ('bool_capacity','negative_origin','fractional_origin','nan_iq'):
        try:
            r=CircularIQBuffer(True if kind=='bool_capacity' else 16)
            if kind!='bool_capacity':r.write(-1 if kind=='negative_origin' else 1.5 if kind=='fractional_origin' else 0,np.full(4,np.nan+0j if kind=='nan_iq' else 1+0j,np.complex64))
            rows.append({'case':kind,'accepted':True,'range':r.available_range()})
        except Exception as e:rows.append({'case':kind,'accepted':False,'error':type(e).__name__})
    return rows

def shutdown_cancel_retry():
    w=FeatureAnalysisWorker(synchronous=False,max_queue_size=4)
    started=threading.Event();release=threading.Event();orig=w._process_one
    def gate(sc):started.set();release.wait(2);return orig(sc)
    w._process_one=gate;w.submit(h.sidecar());assert started.wait(1)
    errors=[]
    try:w.shutdown(drain=True,timeout=.01)
    except Exception as e:errors.append(type(e).__name__)
    sentinel_before=w._queue.qsize()
    try:w.shutdown(drain=False,timeout=.01)
    except Exception as e:errors.append(type(e).__name__)
    sentinel_after=w._queue.qsize();release.set();time.sleep(.03)
    try:w.shutdown(timeout=.03)
    except Exception as e:errors.append(type(e).__name__)
    result={'errors':errors,'queued_sentinel_before_cancel':sentinel_before,'queue_after_cancel':sentinel_after,'sentinel_flag':w._sentinel_enqueued,'state':w.state.value,'thread_alive':w._thread.is_alive(),'unfinished':w._queue.unfinished_tasks}
    if w._thread.is_alive():
        with w._lock:w._sentinel_enqueued=False
        w.shutdown(timeout=1)
    return result

def known_unknown_quality():
    rows=[]
    for label,kwargs,start in [('EOF',{},None),('known_gap',{'dropped_samples_before':31},100),('unknown_loss',{'overflow':True},100)]:
        p=h.p(pre_trigger_samples=0,post_signal_samples=0)
        p.process_block(h.b(np.full(69,.5,np.complex64)))
        ev=p.flush() if start is None else p.process_block(h.b(np.zeros(16,np.complex64),1,start,**kwargs))
        p.shutdown();sc=ev[0];w=FeatureAnalysisWorker(synchronous=True,skip_incomplete_quality=False);res=w.submit(sc);w.shutdown()
        rows.append({'cause':label,'sidecar_known':sc.known_lost_samples,'sidecar_unknown':sc.unknown_loss,'termination':sc.termination_reason,'analysis_quality':dataclasses.asdict(res.analysis_result.capture_quality),'feature_status':res.feature_status})
    return rows

def metadata_failure_commit():
    p=h.p(noise_estimator=GaussianNoiseEstimator(window_size=16),pre_trigger_samples=0,post_signal_samples=0)
    s=StreamInfo(h.S.stream_id,h.S.source_id,1e6,center_frequency_hz=100e6)
    try:p.process_block(IQBlock(s,0,0,None,np.ones(16,np.complex64)))
    except Exception as e:error=type(e).__name__
    else:error=None
    r={'error':error,'state':p.state.value,'adopted_center_after_error':p._stream_info.center_frequency_hz,'ring':p.ring_buffer.available_range()};p.shutdown();return r

def reset_callback_race():
    w=FeatureAnalysisWorker(synchronous=False,max_queue_size=4)
    ready=threading.Event();release=threading.Event();delivered=threading.Event();calls=[];orig=w._deliver_sink
    def hold(sc):ready.set();release.wait(2);return orig(sc)
    def sink(sc):calls.append({'id':sc.event.event_id,'generation_at_callback':w.current_generation});delivered.set()
    w._deliver_sink=hold;w.add_result_sink(sink);w.submit(h.sidecar());assert ready.wait(1)
    before=w.current_generation;w.reset();after=w.current_generation;release.set();delivered.wait(.05);w.shutdown()
    return {'generation_before':before,'generation_after_reset':after,'callbacks':calls,'history_after_reset':len(w.processed_results),'discarded_count':w.discarded_old_generation_count}

def signed_band_contracts():
    band=(-110000,-90000);r=[]
    try:d=EnergyDetector(mode='configured_band',monitored_band_hz=band);r.append({'api':'EnergyDetector','accepted':True})
    except Exception as e:r.append({'api':'EnergyDetector','accepted':False,'error':type(e).__name__})
    try:EmpiricalNoiseCalibrator(window_size=16).begin_calibration(monitored_band_hz=band);r.append({'api':'EmpiricalNoiseCalibrator','accepted':True})
    except Exception as e:r.append({'api':'EmpiricalNoiseCalibrator','accepted':False,'error':type(e).__name__})
    # Exercise a genuinely negative-band calibration; relabeling a full-band
    # reference now correctly violates the immutable domain contract.
    q=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1).create_policy(noise(20000),monitored_band_hz=band,settling_samples=4096)
    try:dataclasses.replace(q,monitored_band_hz=band);r.append({'api':'ThresholdPolicy','accepted':True})
    except Exception as e:r.append({'api':'ThresholdPolicy','accepted':False,'error':type(e).__name__})
    return r

def wide_band_raw_reference():
    x=noise(700000,123);band=(50000,450000)
    c=EmpiricalNoiseCalibrator(window_size=64,hop_size=64)
    q=c.create_policy(x,monitored_band_hz=band,settling_samples=4096)
    raw_mean=float(np.mean(abs(x.astype(np.complex128))**2));rows=[]
    for label,value in [('matched_filtered',q.mean_power_linear),('raw_full_band',raw_mean)]:
        d=EnergyDetector(window_size=64,hop_size=64,mode='configured_band',monitored_band_hz=band,threshold_policy=q)
        try:
            d.process_windows(h.b(noise(128,111)),NoiseEstimate(h.S.stream_id,0,10*math.log10(value)))
            rows.append({'domain':label,'accepted':True,'power':value,'ratio_to_policy':value/q.mean_power_linear})
        except Exception as e:rows.append({'domain':label,'accepted':False,'error':type(e).__name__})
    return rows

def pipeline_policy_failure_commit():
    band=(90000,110000)
    q=EmpiricalNoiseCalibrator(window_size=64,hop_size=64).create_policy(noise(700000),monitored_band_hz=band,settling_samples=4096)
    p=h.p(window_size=64,hop_size=64,detector_mode='configured_band',monitored_band_hz=band,threshold_policy=q,noise_estimator=GaussianNoiseEstimator(window_size=64,calibrated_noise_power_linear=1e-4),pre_trigger_samples=0)
    before=p.ring_buffer.available_range()
    try:p.process_block(h.b(np.ones(64,np.complex64)))
    except Exception as e:error=type(e).__name__
    else:error=None
    out={'error':error,'state':p.state.value,'ring_before':before,'ring_after':p.ring_buffer.available_range(),'previous_block_committed':p._previous_block is not None,'phase_after':p.detector._phase_accumulator};p.shutdown();return out

def calibration_allocation_cap():
    from unittest import mock
    c=EmpiricalNoiseCalibrator(window_size=16,max_samples=64);c.begin_calibration(monitored_band_hz=(90000,110000),settling_samples=8)
    x=np.ones(100000,np.complex64);sizes=[];orig=np.isfinite
    def observe(a,*args,**kw):sizes.append(int(np.size(a)));return orig(a,*args,**kw)
    with mock.patch.object(np,'isfinite',side_effect=observe):c.update_calibration(x)
    return {'budget_plus_settling':72,'finite_validation_input_sizes':sizes,'retained':c._cal_samples_collected}

def gap_during_training():
    est=GaussianNoiseEstimator(window_size=16,warmup_samples=64)
    p=h.p(noise_estimator=est,pre_trigger_samples=0,post_signal_samples=0)
    p.process_block(h.b(np.full(32,.01,np.complex64)))
    before=est.samples_trained;events=p.process_block(h.b(np.full(512,.5,np.complex64),1,64,dropped_samples_before=32));events+=p.flush();p.shutdown()
    return {'authorized_quiet_before_gap':before,'reference_after_signal':est.noise_power_linear,'trained_after_gap':est.samples_trained,'state':est.state.value,'events':[h.summ(s) for s in events]}

PROBES=[nonmultiple_eof,future_admission,quiet_without_successful_update,configured_warmup_partition,configured_warmup_future_poison,near_equal_reference_mismatch,filtered_gamma_false_alarm,bandwidth_roi_bypass,bandwidth_policy_collision,calibration_lifecycle_and_shape,support_without_reliability,direct_buffer_contracts,shutdown_cancel_retry,known_unknown_quality,metadata_failure_commit,reset_callback_race,signed_band_contracts,wide_band_raw_reference,pipeline_policy_failure_commit,calibration_allocation_cap,gap_during_training]
if __name__=='__main__':
    result={'source':str(SRC)}
    for fn in PROBES:
        try:result[fn.__name__]={'executed':True,'observed':fn()}
        except Exception as e:result[fn.__name__]={'executed':False,'error':type(e).__name__,'message':str(e)}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=True));print(json.dumps(result,indent=2,allow_nan=True))
