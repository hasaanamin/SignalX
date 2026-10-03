"""Independent probes for examples2_repaired.zip. Reviewed implementation is unchanged."""
from pathlib import Path
import os, sys, math, json, dataclasses, threading
import numpy as np
SRC=Path(os.environ['SIGNAL_PROCESSING_SRC']).resolve()
sys.path.insert(0,str(SRC))
from signal_processing.core import IQBlock, StreamInfo, NoiseEstimate
from signal_processing.detection import EnergyDetector, GaussianNoiseEstimator, AdaptiveRobustNoiseEstimator, EmpiricalNoiseCalibrator
from signal_processing.runtime import StreamingCapturePipeline
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.detection.exceptions import PolicyCompatibilityError

S=StreamInfo('repair-audit','synthetic',1e6)
def b(x,seq=0,start=0,**kw):
    return IQBlock(S,seq,start,None,np.asarray(x,dtype=np.complex64),**kw)
def pipeline(**kw):
    args=dict(stream_info=S,ring_capacity=4096,window_size=16,hop_size=8,
              start_persistence=1,end_persistence=1,pre_trigger_samples=0,
              pre_signal_margin_samples=0,post_signal_samples=0,min_post_trigger_samples=0,
              enable_feature_analysis=False,noise_estimator=GaussianNoiseEstimator(window_size=16,calibrated_noise_power_linear=1e-4))
    args.update(kw)
    return StreamingCapturePipeline(**args)
def noise(count,seed=448,p=1e-4):
    rng=np.random.default_rng(seed)
    return ((rng.normal(size=count)+1j*rng.normal(size=count))*math.sqrt(p/2)).astype(np.complex64)
def summary(sc):
    return dict(event_id=sc.event_id,start=sc.event.event_start_sample,end=sc.event.event_end_sample,
                status=sc.feature_status,termination=sc.termination_reason,capture_epoch=sc.stream_epoch,
                processing_generation=sc.processing_generation)

def async_gap_delivery():
    """Pause a real worker before analysis; submit a completed old-prefix capture at a gap."""
    rows=[]
    for sync in (True,False):
        p=pipeline(enable_feature_analysis=True,synchronous_analysis=sync,skip_incomplete_quality=False)
        entered=threading.Event(); release=threading.Event(); callbacks=[]
        original=p.worker._process_one
        if not sync:
            def held(sc):
                entered.set()
                if not release.wait(5): raise TimeoutError('probe release missing')
                return original(sc)
            p.worker._process_one=held
        p.worker.add_result_sink(lambda sc:callbacks.append(summary(sc)))
        p.process_block(b(np.full(64,.5)))
        returned=p.process_block(b(np.full(64,.003),1,96,dropped_samples_before=32))
        if not sync:
            # Whether the worker began before generation advanced does not matter:
            # the old job is discarded before analysis or before result publication.
            release.set()
        p.flush(); p.shutdown()
        rows.append(dict(synchronous=sync,returned=[summary(sc) for sc in returned],
                         callbacks=callbacks,stored=[summary(sc) for sc in p.worker.processed_results],
                         discarded=p.worker.discarded_old_generation_count,
                         unfinished=p.worker._queue.unfinished_tasks))
    return rows

def partial_block_emissions():
    """A completed real capture before a later policy failure must remain recoverable."""
    q=EmpiricalNoiseCalibrator(window_size=16,hop_size=8,pfa_on=.01,pfa_off=.1).create_policy(noise(20000),settling_samples=0)
    e=AdaptiveRobustNoiseEstimator(window_size=16,initial_noise_power_linear=q.mean_power_linear,
                                  adaptation_window_size=16,time_constant_s=1e-6)
    p=pipeline(pfa_on=.01,pfa_off=.1,threshold_policy=q,noise_estimator=e)
    internally_dispatched=[]; dispatch=p._dispatch_event
    def observed(sc):
        internally_dispatched.append(summary(sc)); return dispatch(sc)
    p._dispatch_event=observed
    iq=np.full(128,.003,np.complex64);iq[:32]=.5
    error=None; returned=[]
    try: returned=p.process_block(b(iq))
    except Exception as exc:
        error=type(exc).__name__
        returned=list(getattr(exc, 'completed_captures', ()))
    after=p.flush();p.shutdown()
    return dict(error=error,state=p.state.value,consumed=p.consumed_end_sample,received=p.received_end_sample,
                internally_dispatched=internally_dispatched,returned=[summary(sc) for sc in returned],
                recovered=[summary(sc) for sc in after],worker_stored=[summary(sc) for sc in p.worker.processed_results])

def held_out_evidence_binding():
    q=EmpiricalNoiseCalibrator(window_size=16,hop_size=8,pfa_on=.01,pfa_off=.1).create_policy(noise(20000),settling_samples=0)
    # Validate a fixed conservative policy against genuine independent Gaussian powers.
    q=dataclasses.replace(q,t_on_linear=q.mean_power_linear*4,t_off_linear=q.mean_power_linear*3,
                          t_on_dbfs=10*math.log10(q.mean_power_linear*4),t_off_dbfs=10*math.log10(q.mean_power_linear*3))
    rng=np.random.default_rng(887)
    powers=rng.gamma(16,q.mean_power_linear/16,10000)
    supported=q.validate_held_out(powers,reference_domain=q.reference_domain,
                                independence_assumption='independent_held_out_windows',evidence_id='real-independent-powers-887')
    new_on,new_off=q.mean_power_linear*.5,q.mean_power_linear*.4
    try:
        changed=dataclasses.replace(supported,t_on_linear=new_on,t_off_linear=new_off,
                                    t_on_dbfs=10*math.log10(new_on),t_off_dbfs=10*math.log10(new_off))
    except ValueError as exc:
        return dict(original_supported=supported.supported_pfa,original_count=supported.validation_exceedances_on,
                    original_upper=supported.pfa_upper_on,changed_supported=False,replacement_rejected=type(exc).__name__,
                    actual_changed_exceedances=int(np.count_nonzero(powers>=new_on)),validation_windows=len(powers),
                    detector_accepts_changed_evidence=False)
    accepted=False;error=None
    try:
        d=EnergyDetector(window_size=16,hop_size=8,pfa_on=.01,pfa_off=.1,threshold_policy=changed,require_supported_pfa=True)
        d.process(b(np.full(16,math.sqrt(q.mean_power_linear))),NoiseEstimate(S.stream_id,0,10*math.log10(q.mean_power_linear),reference_domain=q.reference_domain))
        accepted=True
    except Exception as exc:error=type(exc).__name__
    return dict(original_supported=supported.supported_pfa,original_count=supported.validation_exceedances_on,
                original_upper=supported.pfa_upper_on,changed_supported=changed.supported_pfa,
                retained_count=changed.validation_exceedances_on,actual_changed_exceedances=int(np.count_nonzero(powers>=changed.t_on_linear)),
                validation_windows=len(powers),detector_accepts_changed_evidence=accepted,error=error)

def detector_discontinuity():
    d=EnergyDetector(window_size=16,hop_size=8)
    ref=NoiseEstimate(S.stream_id,0,-40)
    first=d.process_windows(b(np.full(8,.2)),ref)
    error=None
    try:second=d.process_windows(b(np.full(8,.3),1,100,dropped_samples_before=92),ref)
    except StreamConsistencyError as exc:second=[];error=type(exc).__name__
    return dict(first_count=len(first),second=[dict(start=m.start_sample,end=m.end_sample,power=m.signal_power_dbfs) for m in second],received_spans=[[0,8],[100,108]],error=error)

def analytic_certified_scale():
    q=EmpiricalNoiseCalibrator(window_size=16,hop_size=8,pfa_on=.01,pfa_off=.1).create_policy(noise(20000),settling_samples=0)
    # Analytic policy is a public supported mode; validate its literal thresholds.
    from scipy.stats import gamma
    on=gamma.isf(.01,a=16,scale=q.mean_power_linear/16);off=gamma.isf(.1,a=16,scale=q.mean_power_linear/16)
    q=dataclasses.replace(q,mode='analytic_gamma',t_on_linear=on,t_off_linear=off,
                          t_on_dbfs=10*math.log10(on),t_off_dbfs=10*math.log10(off))
    # An independent held-out domain with smaller actual mean supports fixed thresholds.
    rng=np.random.default_rng(712)
    q=q.validate_held_out(rng.gamma(16,q.mean_power_linear/32,10000),reference_domain=q.reference_domain,
                         independence_assumption='independent_held_out_windows',evidence_id='real-independent-powers-712')
    d=EnergyDetector(window_size=16,hop_size=8,pfa_on=.01,pfa_off=.1,threshold_policy=q,require_supported_pfa=True)
    ref=NoiseEstimate(S.stream_id,0,10*math.log10(q.mean_power_linear*.1),reference_domain=q.reference_domain)
    try:m=d.process(b(noise(16)),ref)
    except PolicyCompatibilityError as exc:
        return dict(supported=q.supported_pfa,rejected=True,error=type(exc).__name__,calibrated_on=q.t_on_linear)
    return dict(supported=q.supported_pfa,rejected=False,calibrated_on=q.t_on_linear,runtime_on=10**(m.on_threshold_dbfs/10),ratio=10**(m.on_threshold_dbfs/10)/q.t_on_linear)

def runtime_conversion_floor():
    rows=[]
    for scale in (1.,1e-8):
        mean=1e-4*scale*scale
        d=EnergyDetector(window_size=16,hop_size=8)
        m=d.process(b(np.full(16,.02*scale)),NoiseEstimate(S.stream_id,0,10*math.log10(mean)))
        rows.append(dict(scale=scale,physical_snr_ratio=4.,on=m.above_on_threshold,off=m.below_off_threshold,
                         measured_dbfs=m.signal_power_dbfs,on_threshold_dbfs=m.on_threshold_dbfs,
                         effective_threshold_over_noise=10**(m.on_threshold_dbfs/10)/mean))
    return rows

def benchmark_failing_sink():
    import benchmarks.sustained_evaluation as evaluation
    original=evaluation._pipeline
    def configured(*args,**kwargs):
        p=original(*args,**kwargs)
        def failed(sc): raise RuntimeError('deliberate external sink failure')
        p.worker.add_result_sink(failed)
        return p
    evaluation._pipeline=configured
    try:
        r=evaluation.bounded_acquisition(.6,sample_rate_hz=10000,block_size=256,
                                         features=True,queue_capacity=4)
    finally:evaluation._pipeline=original
    return {k:r[k] for k in ('passed','recovered','errors','worker_sink_errors','worker_failed',
                             'worker_completed','deadline_misses','dropped_samples','overload_drops',
                             'result_overload_drops','worker_discarded_generation','worker_state','worker_unfinished')}

def gaussian_warmup_partition_oracle():
    rows=[]; reference=None
    for part in (1,3,8,17,128):
        e=GaussianNoiseEstimator(window_size=17,warmup_samples=19,warmup_settling_samples=9)
        p=pipeline(window_size=17,hop_size=8,noise_estimator=e)
        x=np.full(128,.003,np.complex64);x[:9]=.6;x[28:60]=.5
        events=[]
        for seq,start in enumerate(range(0,len(x),part)):
            events.extend(p.process_block(b(x[start:start+part],seq,10**12+start)))
        events.extend(p.flush());p.shutdown()
        sig=[(sc.event.event_start_sample-10**12,sc.event.event_end_sample-10**12,
              sc.event.detected_start_sample-10**12,sc.event.detected_end_sample-10**12,
              sc.event.iq.tobytes()) for sc in events]
        if reference is None:reference=sig
        rows.append(dict(partition=part,same_as_reference=sig==reference,training_samples=e.samples_trained,
                         trained_power=e.noise_power_linear,
                         domain_matches=e.reference_domain==p.detector.reference_domain,
                         events=[summary(sc) for sc in events],
                         exact_iq=all(np.array_equal(sc.event.iq,x[sc.event.event_start_sample-10**12:sc.event.event_end_sample-10**12]) for sc in events)))
    return rows

def continuous_filtered_covariance_oracle():
    from scipy import signal, linalg
    rows=[]
    for n,width in ((64,20000),(256,20000),(64,1000),(8192,1000),(64,10)):
        d=EnergyDetector(window_size=n,hop_size=min(n,64),mode='configured_band',
                         monitored_band_hz=(100000-width/2,100000+width/2))
        record=dict(n=n,width_hz=width,method=d._covariance_method,threshold_model=d.threshold_model,
                    impulse_length=d._impulse_length,tail_fraction=d._impulse_tail_fraction,
                    effective_n=d._n_eff,unit_on=d._unit_q_on,unit_off=d._unit_q_off)
        if n<=256 and width>=1000:
            impulse=np.zeros(131072);impulse[0]=1
            h=signal.sosfilt(d._sos,impulse)
            rho=signal.fftconvolve(h,h[::-1],mode='full')[len(h)-1:][:n]
            rho/=rho[0]
            cov=linalg.toeplitz(rho)
            weights=np.linalg.eigvalsh(cov)/n
            var=np.sum(weights**2);w_max=np.max(weights)
            t=-math.log(1e-4)
            exact_eigenvalue_bound=1+math.sqrt(2*var*t)+w_max*t
            record.update(independent_covariance_neff=1/var,
                          neff_relative_error=abs(d._n_eff*var-1),
                          eigenvalue_bound=exact_eigenvalue_bound,
                          computed_at_least_exact_bound=bool(d._unit_q_on+1e-10>=exact_eigenvalue_bound))
        rows.append(record)
    return rows

PROBES=[async_gap_delivery,partial_block_emissions,held_out_evidence_binding,detector_discontinuity,analytic_certified_scale,runtime_conversion_floor,benchmark_failing_sink,gaussian_warmup_partition_oracle,continuous_filtered_covariance_oracle]
if __name__=='__main__':
    results={'source':str(SRC)}
    for probe in PROBES:
        if probe.__name__=='benchmark_failing_sink' and os.environ.get('AUDIT_PROBE_SKIP_BENCHMARK'): continue
        try:results[probe.__name__]=probe()
        except Exception as exc:results[probe.__name__]={'probe_error':type(exc).__name__,'detail':str(exc)}
    Path(os.environ.get('AUDIT_PROBE_OUTPUT',str(Path(__file__).with_suffix('.json')))).write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
