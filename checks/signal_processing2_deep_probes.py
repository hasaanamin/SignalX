"""Independent runtime probes for signal_processing(2).zip; reviewed source is unchanged."""
from pathlib import Path
import os, sys, json, math, threading, time, dataclasses, hashlib, importlib.util
import numpy as np
from scipy import signal
ROOT=Path(__file__).parent
SRC=Path(os.environ.get('SIGNAL_PROCESSING_SRC', ROOT.parent if (ROOT.parent/'signal_processing').exists() else ROOT/'signal_processing2_review')).resolve()
sys.path.insert(0,str(SRC))
from signal_processing.core import IQBlock, NoiseEstimate, SignalEvent, StreamInfo, PSDResult
from signal_processing.detection import EnergyDetector, GaussianNoiseEstimator, AdaptiveRobustNoiseEstimator, EmpiricalNoiseCalibrator
from signal_processing.detection.noise_estimation import ThresholdPolicy
from signal_processing.runtime import StreamingCapturePipeline, FeatureAnalysisWorker
from signal_processing.capture.sidecar import CaptureSidecar
from signal_processing.context import SignalContext, AnalysisConfig
from signal_processing.features import FeatureAggregator, build_default_feature_registry
from signal_processing.features.aggregation import FeatureRegistry
from signal_processing.features.bandwidth import BandwidthFeatureExtractor, BandwidthFeatureConfig
S=StreamInfo('audit2','source',1e6)
RESULTS={}
def record(name,fn):
    if __name__ != '__main__':return
    try:RESULTS[name]={'executed':True,'observed':fn()}
    except Exception as e:RESULTS[name]={'executed':False,'exception':type(e).__name__,'message':str(e)}
def b(x,seq=0,start=0,stream=S,**kw):return IQBlock(stream,seq,start,kw.pop('timestamp_ns',None),np.asarray(x,np.complex64),**kw)
def p(**kw):
    a=dict(stream_info=S,ring_capacity=4096,window_size=16,hop_size=8,start_persistence=1,end_persistence=1,
           pre_trigger_samples=32,pre_signal_margin_samples=0,post_signal_samples=32,min_post_trigger_samples=0,
           max_event_samples=10000,noise_estimator=GaussianNoiseEstimator(window_size=16,calibrated_noise_power_linear=1e-4),
           enable_feature_analysis=False)
    a.update(kw);return StreamingCapturePipeline(**a)
def summ(sc):
    e=sc.event
    return {'start':e.event_start_sample,'end':e.event_end_sample,'ds':e.detected_start_sample,'de':e.detected_end_sample,
            'trigger':e.trigger_sample,'complete':e.complete,'termination':sc.termination_reason,
            'boundary_uncertainty':sc.boundary_uncertainty_samples,'iq_sha256':hashlib.sha256(e.iq.tobytes()).hexdigest()}
def run(x,bs,**kw):
    pipe=p(**kw);out=[];err=None;ms=[]
    original=pipe.detector.process_windows
    def measured(block,ref):
        values=original(block,ref);ms.extend((m.end_sample,m.noise_power_dbfs,m.above_on_threshold,m.below_off_threshold) for m in values)
        return values
    pipe.detector.process_windows=measured
    try:
        k=0;seq=0
        while k<len(x):
            n=bs[seq%len(bs)] if isinstance(bs,list) else bs
            out.extend(pipe.process_block(b(x[k:k+n],seq,k)));k+=n;seq+=1
        out.extend(pipe.flush())
    except Exception as e:err=type(e).__name__+': '+str(e)
    finally:pipe.shutdown()
    est=pipe.noise_estimator
    return {'block_size':bs,'events':[summ(q) for q in out],'error':err,'state':pipe.state.value,
            'reference':est.noise_power_linear,'age':getattr(est,'staleness_samples',None),
            'accepted':getattr(est,'total_accepted_samples',None),'pending':sum(len(z) for z in pipe._pending_noise_iq),
            'measurements':ms}
def adaptive_misaligned():
    x=np.full(2048,.003,np.complex64);x[512:768]=.01
    rows=[]
    for bs in (1,7,11,16,31,64,2048,[3,17,5,61,2,23]):
        est=AdaptiveRobustNoiseEstimator(window_size=16,initial_noise_power_linear=1e-4,time_constant_s=1e-4,adaptation_window_size=64)
        r=run(x,bs,noise_estimator=est)
        r['noise_timeline_sha256']=hashlib.sha256(repr(r.pop('measurements')).encode()).hexdigest();rows.append(r)
    return rows
record('adaptive_misaligned_partition',adaptive_misaligned)
def age_after_burst():
    x=np.full(512,.003,np.complex64);x[128:192]=.5
    rows=[]
    for bs in (1,8,16,31,64,512):
        est=AdaptiveRobustNoiseEstimator(window_size=16,initial_noise_power_linear=1e-4,time_constant_s=1e-4,
                                         adaptation_window_size=16,max_staleness_samples=1024)
        r=run(x,bs,noise_estimator=est,pre_trigger_samples=0,post_signal_samples=0)
        r.pop('measurements');rows.append(r)
    return rows
record('noise_age_after_burst_then_quiet',age_after_burst)
def staleness_partition():
    rows=[]
    for bs in (8,32,256):
        est=AdaptiveRobustNoiseEstimator(window_size=16,initial_noise_power_linear=1e-4,adaptation_window_size=16,
                                         max_staleness_samples=32)
        r=run(np.full(256,.5,np.complex64),bs,noise_estimator=est,pre_trigger_samples=0,post_signal_samples=0)
        r['measurements_processed']=len(r.pop('measurements'));rows.append(r)
    return rows
record('stale_prefix_partition',staleness_partition)
def sidecar():
    e=SignalEvent('e',S,np.ones(64,np.complex64),0,64,0,64,0)
    return CaptureSidecar(stream_epoch=0,event_id='e',config_version='audit',detector_mode='full_band',monitored_band_hz=None,
        first_on_sample=0,confirmation_sample=16,end_confirmation_sample=64,requested_leading_samples=0,actual_leading_samples=0,
        requested_trailing_samples=0,actual_trailing_samples=0,boundary_uncertainty_samples=8,noise_provenance='external',
        noise_valid=True,termination_reason='normal',is_truncated=False,known_lost_samples=0,unknown_loss=False,
        feature_status='pending',feature_reason=None,event=e)
def sync_sink():
    calls=[];w=FeatureAnalysisWorker(synchronous=True,result_sink=lambda sc:calls.append(sc.event_id))
    result=w.submit(sidecar());w.shutdown()
    return {'returned_status':result.feature_status,'callbacks':calls,'recorded_results':len(w.processed_results),'completed':w.completed_count}
record('synchronous_sink_delivery',sync_sink)
def throwing_sink():
    def sink(sc):raise ValueError('deliberate failure')
    w=FeatureAnalysisWorker(synchronous=False,result_sink=sink);w.submit(sidecar());w.shutdown()
    return {'completed':w.completed_count,'sink_errors':w.sink_error_count,'unfinished':w._queue.unfinished_tasks}
record('asynchronous_sink_error_observable',throwing_sink)
def repeated_shutdown():
    entered=threading.Event();release=threading.Event();w=FeatureAnalysisWorker(synchronous=False,max_queue_size=4)
    original=w._process_one
    def gated(sc):entered.set();release.wait(3);return original(sc)
    w._process_one=gated;w.submit(sidecar());assert entered.wait(2)
    errors=[]
    for _ in range(2):
        try:w.shutdown(timeout=.025)
        except TimeoutError:errors.append('TimeoutError')
    before={'queued':w._queue.qsize(),'unfinished':w._queue.unfinished_tasks}
    release.set();w._thread.join(2);w.shutdown(timeout=1)
    return {'timeouts':errors,'before_release':before,'state':w.state.value,'thread_alive':w._thread.is_alive(),
            'queued_after_close':w._queue.qsize(),'unfinished_after_close':w._queue.unfinished_tasks,'completed':w.completed_count}
record('repeated_shutdown_sentinels',repeated_shutdown)
def overload_sink():
    entered=threading.Event();release=threading.Event();calls=[]
    w=FeatureAnalysisWorker(synchronous=False,max_queue_size=1,result_sink=lambda sc:calls.append(sc.feature_status))
    orig=w._process_one
    def slow(sc):entered.set();release.wait(2);return orig(sc)
    w._process_one=slow;w.submit(sidecar());assert entered.wait(2)
    w.submit(sidecar());rejected=w.submit(sidecar());release.set();w.shutdown()
    return {'rejected_status':rejected.feature_status,'callbacks':calls,'input_drops':w.dropped_overload_count,
            'completed':w.completed_count,'unfinished':w._queue.unfinished_tasks}
record('overload_sink_delivery',overload_sink)
def numeric_config_gaps():
    from signal_processing.sources import ArrayIQSource
    from signal_processing.segmentation import HysteresisSegmenter
    rows=[]
    constructors=[
        ('detector_Fs_0',lambda:EnergyDetector(sample_rate_hz=0)),
        ('detector_Fs_nan',lambda:EnergyDetector(sample_rate_hz=float('nan'))),
        ('detector_Fs_bool',lambda:EnergyDetector(sample_rate_hz=True)),
        ('source_block_bool',lambda:ArrayIQSource(np.ones(64,np.complex64),S,block_size=True)),
        ('segmenter_max_bool',lambda:HysteresisSegmenter(max_event_samples=True)),
        ('calibrator_band_outside_nyquist',lambda:EmpiricalNoiseCalibrator().begin_calibration(monitored_band_hz=(900000,1000000))),
    ]
    for label,ctor in constructors:
        try:ctor();rows.append({'case':label,'accepted':True})
        except Exception as e:rows.append({'case':label,'accepted':False,'error':type(e).__name__})
    return rows
record('remaining_public_validation',numeric_config_gaps)
def calibration_streaming_cap():
    cal=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1,max_samples=64)
    cal.begin_calibration(monitored_band_hz=(90000,110000),settling_samples=8)
    seen=[];orig=signal.sosfilt
    def spy(sos,x,*args,**kw):seen.append(len(x));return orig(sos,x,*args,**kw)
    signal.sosfilt=spy
    try:cal.update_calibration(np.full(100000,.01,np.complex64));d=cal.finalize_calibration()
    finally:signal.sosfilt=orig
    return {'cap':64,'settling':8,'training_samples':d['num_samples'],'actual_samples_sent_to_filter':seen,
            'retained_filter_phase':cal._cal_phase_acc}
record('streamed_calibration_caps_before_filter',calibration_streaming_cap)
def policy_missing_identity():
    rng=np.random.default_rng(17);x=((rng.normal(size=20000)+1j*rng.normal(size=20000))*.01).astype(np.complex64)
    c=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
    d=c.calibrate(x);d.pop('hop_size');d.pop('pfa_off')
    rows=[]
    for label,c2 in [('changed_H',EmpiricalNoiseCalibrator(window_size=16,hop_size=1,pfa_on=.01,pfa_off=.1)),
                     ('changed_OFF_pfa',EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.2))]:
        try:pol=c2.create_policy(d);rows.append({'case':label,'accepted':True,'policy_H':pol.hop_size,'policy_pfa_off':pol.pfa_off})
        except Exception as e:rows.append({'case':label,'accepted':False,'error':type(e).__name__})
    return rows
record('calibration_missing_identity_relabeling',policy_missing_identity)
def threshold_reference_identity():
    rng=np.random.default_rng(2608)
    x=((rng.normal(size=700000)+1j*rng.normal(size=700000))*math.sqrt(1e-4/2)).astype(np.complex64)
    c=EmpiricalNoiseCalibrator(window_size=64,hop_size=64)
    pol=c.create_policy(x,monitored_band_hz=(90000,110000),settling_samples=4096)
    rows=[]
    tone=(.01*np.exp(2j*np.pi*100000*np.arange(8192)/1e6)).astype(np.complex64)
    for label,power in [('correct_filtered',pol.mean_power_linear),('raw_mismatch',1e-4)]:
        det=EnergyDetector(window_size=64,hop_size=64,mode='configured_band',monitored_band_hz=(90000,110000),threshold_policy=pol)
        ref=NoiseEstimate(S.stream_id,0,10*math.log10(power))
        try:
            ms=[m for m in det.process_windows(b(tone),ref) if m.start_sample>=4096]
            rows.append({'reference':label,'accepted':True,'post_settling_windows':len(ms),'ON_windows':sum(m.above_on_threshold for m in ms)})
        except Exception as e:
            rows.append({'reference':label,'accepted':False,'error':type(e).__name__})
    return rows
record('untyped_reference_domain_still_accepted',threshold_reference_identity)
def quality_loss():
    pipe=p(pre_trigger_samples=0,post_signal_samples=0)
    old=pipe.process_block(b(np.full(69,.5,np.complex64)));out=pipe.process_block(b(np.zeros(16,np.complex64),1,69,overflow=True))
    pipe.shutdown()
    sc=(old+out)[0];w=FeatureAnalysisWorker(synchronous=True,skip_incomplete_quality=False)
    result=w.submit(sc);w.shutdown()
    return {'unknown_loss':sc.unknown_loss,'known_loss':sc.known_lost_samples,'event_dropped':sc.event.dropped_samples,
            'capture_range':[sc.event.event_start_sample,sc.event.event_end_sample],
            'feature_status':result.feature_status,'reported_continuous':result.analysis_result.capture_quality.is_continuous,
            'warnings':result.analysis_result.capture_quality.warnings,'recorded_boundary_uncertainty':sc.boundary_uncertainty_samples}
record('unknown_overflow_quality_provenance',quality_loss)
def emitter_policy_collisions():
    rng=np.random.default_rng(6)
    n=4096;x=(.1*np.exp(2j*np.pi*.07*np.arange(n))+(rng.normal(size=n)+1j*rng.normal(size=n))*.004).astype(np.complex64)
    ctx=SignalContext(SignalEvent('policy',S,x,0,n,0,n,0),AnalysisConfig(psd_mode='welch',psd_nperseg=256))
    rows=[]
    for minimum in (3.,30.):
        reg=FeatureRegistry([BandwidthFeatureExtractor(BandwidthFeatureConfig(mode='emitter',noise_floor_dbfs_per_hz=-100,min_snr_db=minimum))])
        r=FeatureAggregator(reg).extract_analysis_result(ctx)
        rows.append({'minimum_snr_db':minimum,'policy_id':r.policy.policy_id,'features':{k:{'valid':v.valid,'value':v.value,'reason':v.reason} for k,v in r.feature_vector.features.items()}})
    return rows
record('bandwidth_minimum_SNR_policy_collision',emitter_policy_collisions)
def amplitude_policy_collision():
    from signal_processing.features.amplitude import AmplitudeFeatureConfig,AmplitudeFeatureExtractor
    x=np.array([1,2,3,5],np.complex64);ctx=SignalContext(SignalEvent('amplitude',S,x,0,4,0,4,0))
    rows=[]
    for ddof in (0,1):
        reg=FeatureRegistry([AmplitudeFeatureExtractor(AmplitudeFeatureConfig(ddof=ddof))])
        r=FeatureAggregator(reg).extract_analysis_result(ctx)
        rows.append({'ddof':ddof,'policy_id':r.policy.policy_id,'std':r.feature_vector.features['amplitude.std'].value})
    return rows
record('amplitude_ddof_policy_collision',amplitude_policy_collision)
def center_adoption_before_validation():
    pipe=p();foreign=StreamInfo('foreign','different-source',1e6,100e6)
    error=None
    try:pipe.process_block(b(np.ones(32),stream=foreign))
    except Exception as e:error=type(e).__name__
    adopted=pipe._stream_info.center_frequency_hz
    ok=None
    try:pipe.process_block(b(np.ones(32)));ok=True
    except Exception as e:ok=type(e).__name__+': '+str(e)
    pipe.shutdown()
    return {'first_error':error,'center_after_rejected_foreign_block':adopted,'valid_original_block_accepted':ok}
record('center_adoption_mutates_rejected_stream',center_adoption_before_validation)
if __name__ == '__main__':
    Path(__file__).with_suffix('.json').write_text(json.dumps(RESULTS,indent=2))
    print(json.dumps(RESULTS,indent=2))

