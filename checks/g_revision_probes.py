"""Exact 2.2.0 fixtures; G08 helper migrated to authoritative immediate rejection.

Original, including its obsolete mandatory-rejection-callback wait, is frozen in
audit_verification/g_revision/auditor_originals/checks/v220_runtime_probes.py.
"""
from pathlib import Path
import sys,os,json,math,threading,dataclasses
import numpy as np
SRC=Path(os.environ['SIGNAL_PROCESSING_SRC']).resolve();sys.path.insert(0,str(SRC))
from signal_processing.core import StreamInfo,IQBlock,NoiseEstimate,DetectionMeasurement
from signal_processing.detection import EnergyDetector,GaussianNoiseEstimator,AdaptiveRobustNoiseEstimator,EmpiricalNoiseCalibrator
from signal_processing.runtime import StreamingCapturePipeline,FeatureAnalysisWorker
from signal_processing.features import FeatureAggregator
from signal_processing.features.aggregation import build_default_feature_registry
from signal_processing.context import SignalContext
S=StreamInfo('v220-independent','synthetic',1e6)
def block(x,seq=0,start=0):return IQBlock(S,seq,start,None,np.asarray(x,np.complex64))
def pipeline(**kwargs):
    args=dict(stream_info=S,ring_capacity=4096,window_size=16,hop_size=8,start_persistence=1,end_persistence=1,
              pre_trigger_samples=0,pre_signal_margin_samples=0,post_signal_samples=0,min_post_trigger_samples=0,
              enable_feature_analysis=False,noise_estimator=GaussianNoiseEstimator(window_size=16,calibrated_noise_power_linear=1e-4))
    args.update(kwargs);return StreamingCapturePipeline(**args)
def sidecar():
    p=pipeline();x=np.full(128,.003,np.complex64);x[:32]=.5
    r=p.process_block(block(x));r+=p.flush();p.shutdown();return r[0]

def scaled_snr():
    rows=[]
    for scale in (1.,1e-4,1e-8):
        d=EnergyDetector(window_size=16,hop_size=8);mean=1e-4*scale**2
        m=d.process(block(np.full(16,.02*scale)),NoiseEstimate(S.stream_id,0,10*math.log10(mean)))
        rows.append(dict(scale=scale,on=m.above_on_threshold,snr=m.snr_estimate_db,
                         physical_snr=10*math.log10((m.signal_power_linear-m.noise_power_linear)/m.noise_power_linear)))
    d=EnergyDetector(window_size=16,hop_size=8)
    m=d.process(block(np.zeros(16)),NoiseEstimate(S.stream_id,0,-200))
    return dict(scales=rows,zero_signal_snr=m.snr_estimate_db,zero_signal_below_off=m.below_off_threshold)

def dynamic_range_partitions():
    x=np.full(64,1.2e-8,np.complex64);x[0]=1
    oracle=[float(np.mean(np.abs(x[i:i+16].astype(np.complex128))**2)) for i in range(0,49,8)]
    rows=[]
    for size in (1,8,16,64):
        d=EnergyDetector(window_size=16,hop_size=8);ms=[]
        for seq,start in enumerate(range(0,64,size)):
            ms.extend(d.process_windows(block(x[start:start+size],seq,start),NoiseEstimate(S.stream_id,seq,-160)))
        rows.append(dict(partition=size,starts=[m.start_sample for m in ms],
                         powers=[m.signal_power_linear for m in ms],on=[m.above_on_threshold for m in ms],
                         max_quiet_relative_error=max(abs(m.signal_power_linear/oracle[i]-1) for i,m in enumerate(ms) if i)))
    return dict(oracle_powers=oracle,rows=rows)

def sync_shutdown_inflight():
    sc=sidecar();entered=threading.Event();release=threading.Event();callbacks=[];outputs=[]
    aggregate=FeatureAggregator(build_default_feature_registry())
    def analyse(ctx):
        entered.set()
        if not release.wait(3):raise RuntimeError('probe release timed out')
        return aggregate.extract_analysis_result(ctx)
    w=FeatureAnalysisWorker(synchronous=True,analysis_function=analyse,result_sink=lambda r:callbacks.append((r.feature_status,w.state.value)))
    t=threading.Thread(target=lambda:outputs.append(w.submit(sc)));t.start();assert entered.wait(2)
    error=None
    try:w.shutdown(drain=True,timeout=.02)
    except Exception as exc:error=type(exc).__name__
    before=dict(shutdown_error=error,state=w.state.value,submit_thread_alive=t.is_alive(),callbacks=len(callbacks),outputs=len(outputs))
    release.set();t.join(2);w.shutdown(timeout=1)
    return dict(before_release=before,after_callbacks=callbacks,after_outputs=len(outputs),thread_alive=t.is_alive())

def frozen_warmup():
    e=GaussianNoiseEstimator(window_size=16,warmup_samples=16);e.freeze(True)
    p=pipeline(noise_estimator=e);errors=[]
    try:p.process_block(block(np.full(64,.003)))
    except Exception as exc:errors.append(dict(phase='frozen',type=type(exc).__name__,detail=str(exc)))
    before=dict(trained=e.samples_trained,detector_input_end=p.detector._input_end,ring_range=p.ring_buffer.available_range(),state=p.state.value)
    e.freeze(False)
    try:p.process_block(block(np.full(64,.003),1,64))
    except Exception as exc:errors.append(dict(phase='resume',type=type(exc).__name__,detail=str(exc)))
    after=dict(trained=e.samples_trained,state=p.state.value,ring_range=p.ring_buffer.available_range())
    p.shutdown();return dict(before=before,after=after,errors=errors)

def training_dynamic_range():
    rng=np.random.default_rng(2233);x=((rng.normal(size=2048)+1j*rng.normal(size=2048))*1e-8).astype(np.complex64);x[0]=1
    rows=[]
    oracle=np.array([np.mean(abs(x[i:i+16].astype(np.complex128))**2) for i in range(0,len(x)-15,8)])
    for size in (16,256,2048):
        c=EmpiricalNoiseCalibrator(window_size=16,hop_size=8,pfa_on=.01,pfa_off=.1)
        c.begin_calibration()
        for start in range(0,len(x),size):c.update_calibration(x[start:start+size])
        values=np.array(c._cal_window_powers)
        result=c.finalize_calibration()
        rows.append(dict(partition=size,windows=len(values),nonpositive_quiet_windows=int(np.count_nonzero(values[1:]<=0)),
                         maximum_quiet_relative_error=float(np.max(abs(values[1:]/oracle[1:]-1))),
                         t_on=result['empirical_t_on_linear'],t_off=result['empirical_t_off_linear']))
    return dict(rows=rows,oracle_on=float(np.quantile(oracle,.99)),oracle_off=float(np.quantile(oracle,.9)))

def backend_event_mismatch():
    sc=sidecar();aggregate=FeatureAggregator(build_default_feature_registry())
    cached=aggregate.extract_analysis_result(SignalContext(sc.event))
    other_event=dataclasses.replace(sc.event,event_id='other-valid-event',iq=np.full_like(sc.event.iq,.001))
    other=dataclasses.replace(sc,event=other_event,event_id=other_event.event_id)
    w=FeatureAnalysisWorker(synchronous=True,analysis_function=lambda ctx:cached)
    r=w.submit(other);w.shutdown()
    return dict(submitted_id=other.event_id,status=r.feature_status,vector_id=r.features.event_id if r.features else None,
                analysis_vector_id=r.analysis_result.feature_vector.event_id if r.analysis_result else None,
                submitted_mean_amplitude=float(np.mean(abs(other.event.iq))),
                reported_mean_amplitude=r.features.features['amplitude.mean'].value if r.features else None)

def polling_reset_positive():
    # Serial public reset invalidates the old result/reservation; the next
    # generation must admit work and release its result reservation correctly.
    sc=sidecar();w=FeatureAnalysisWorker(synchronous=False,max_results=2)
    w.submit(sc);w._queue.join()
    # Ordinary public reset should invalidate the old result and reservation.
    w.reset();r=w.submit(sc);w._queue.join();out=w.get_result(block=False)
    record=dict(submit_status=r.feature_status,result_status=out.feature_status,reserved=w._polling_reserved,
                current_generation=w.current_generation,result_generation=out.processing_generation)
    w.shutdown();return record

def timestamp_consistency():
    rows=[]
    for warmup in (False,True):
        for jump in (0,1_000_000_000):
            est=GaussianNoiseEstimator(window_size=16,warmup_samples=16) if warmup else GaussianNoiseEstimator(window_size=16,calibrated_noise_power_linear=1e-4)
            p=pipeline(noise_estimator=est)
            p.process_block(IQBlock(S,0,0,1_000_000_000,np.full(64,.003,np.complex64)))
            previous_timestamp=p._previous_block.timestamp_ns
            x=np.full(128,.003,np.complex64);x[:32]=.5;error=None;events=[]
            try:events=p.process_block(IQBlock(S,1,64,1_000_064_000+jump,x))
            except Exception as exc:error=type(exc).__name__
            rows.append(dict(warmup=warmup,jump_ns=jump,previous_timestamp=previous_timestamp,error=error,
                             events=[dict(start=sc.event.event_start_sample,timestamp=sc.event.timestamp_ns,
                                          first_anchor_expected=1_000_000_000+sc.event.event_start_sample*1000,
                                          complete=sc.event.complete) for sc in events]))
            p.shutdown()
    return rows

def async_overload_sink_blocking():
    sc=sidecar();entered=threading.Event();release_analysis=threading.Event()
    sink_entered=threading.Event();release_sink=threading.Event();records=[];third_output=[]
    aggregate=FeatureAggregator(build_default_feature_registry())
    def analyse(ctx):
        entered.set()
        if not release_analysis.wait(3):raise RuntimeError('analysis release missing')
        return aggregate.extract_analysis_result(ctx)
    def sink(result):
        records.append(dict(status=result.feature_status,thread=threading.get_ident()))
        if result.feature_status=='dropped_overload':
            sink_entered.set()
            if not release_sink.wait(3):raise RuntimeError('sink release missing')
    w=FeatureAnalysisWorker(synchronous=False,max_queue_size=1,result_delivery='callbacks',analysis_function=analyse,result_sink=sink)
    first=w.submit(sc);assert entered.wait(2);second=w.submit(sc)
    returned=threading.Event()
    def submit_third():
        third_output.append(w.submit(sc));returned.set()
    caller=threading.Thread(target=submit_third);caller.start()
    did_return=returned.wait(2)
    before=dict(submitter_blocked=caller.is_alive(),third_returned=len(third_output),
                rejected_sink_on_submitter=any(r['status']=='dropped_overload' and r['thread']==caller.ident for r in records),
                normal_worker_thread=w._thread.ident,rejected_sink_thread=None)
    release_sink.set();caller.join(2);release_analysis.set();w.shutdown(timeout=3)
    assert did_return and not sink_entered.is_set(), 'Rejection must return without invoking a sink'
    return dict(before_release=before,first_status=first.feature_status,second_status=second.feature_status,
                rejected_status=third_output[0].feature_status,completed=w.completed_count,
                unfinished=w._queue.unfinished_tasks,sink_errors=w.sink_error_count)

PROBES=[scaled_snr,dynamic_range_partitions,sync_shutdown_inflight,frozen_warmup,training_dynamic_range,backend_event_mismatch,polling_reset_positive,timestamp_consistency,async_overload_sink_blocking]
if __name__=='__main__':
    result={'source':str(SRC)}
    for f in PROBES:
        try:result[f.__name__]=f()
        except Exception as exc:result[f.__name__]={'probe_error':type(exc).__name__,'detail':str(exc)}
    out=Path(os.environ.get('V220_PROBE_OUTPUT',str(Path(__file__).with_suffix('.json'))));out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
