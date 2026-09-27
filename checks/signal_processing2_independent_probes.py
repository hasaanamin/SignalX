"""Independent, current-API audit probes; no changes to the implementation.

Run from any working directory with Python, NumPy and SciPy installed.
Every probe records observations rather than treating harness/API errors as defects.
"""
from pathlib import Path
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
import warnings

import numpy as np
from scipy import signal, stats

import os

ROOT = Path(__file__).resolve().parent
SRC = Path(os.environ.get('SIGNAL_PROCESSING_SRC', ROOT.parent if (ROOT.parent / 'signal_processing').exists() else ROOT / 'signal_processing2_review')).resolve()
sys.path.insert(0, str(SRC))
from signal_processing.core import IQBlock, NoiseEstimate, SignalEvent, StreamInfo
from signal_processing.runtime import FeatureAnalysisWorker, StreamingCapturePipeline
from signal_processing.detection import (AdaptiveRobustNoiseEstimator, EmpiricalNoiseCalibrator,
                                        EnergyDetector, GaussianNoiseEstimator)
from signal_processing.detection.noise_estimation import ThresholdPolicy
from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.features import FeatureAggregator, build_default_feature_registry
from signal_processing.features.aggregation import FeatureRegistry
from signal_processing.features.bandwidth import BandwidthFeatureConfig, BandwidthFeatureExtractor

STREAM = StreamInfo("independent", "audit-generator", 1e6)
RESULTS = {}


def case(name, fn):
    try:
        RESULTS[name] = fn()
    except Exception as exc:
        RESULTS[name] = {"harness_or_probe_exception": type(exc).__name__, "message": str(exc)}


def block(x, seq=0, start=0, info=STREAM, **kw):
    return IQBlock(stream=info, sequence_number=seq, start_sample=start,
                   timestamp_ns=kw.pop("timestamp_ns", None), iq=np.asarray(x, np.complex64), **kw)


def reference(power):
    return NoiseEstimate(stream_id=STREAM.stream_id, block_sequence_number=0,
                         noise_power_dbfs=10 * math.log10(power), valid=True)


def pipe(**kw):
    args = dict(stream_info=STREAM, ring_capacity=4096, window_size=16, hop_size=8,
                start_persistence=1, end_persistence=1, pre_trigger_samples=32,
                pre_signal_margin_samples=0, post_signal_samples=32,
                min_post_trigger_samples=0, max_event_samples=10000,
                noise_estimator=GaussianNoiseEstimator(window_size=16,
                                                       calibrated_noise_power_linear=1e-4))
    args.update(kw)
    return StreamingCapturePipeline(**args)


def summary(sc):
    return dict(start=sc.event.event_start_sample, end=sc.event.event_end_sample,
                detected_start=sc.event.detected_start_sample, detected_end=sc.event.detected_end_sample,
                trigger=sc.event.trigger_sample, complete=sc.event.complete,
                termination=sc.termination_reason, noise_valid=sc.noise_valid,
                samples=sc.event.num_samples)


def feed(x, bs, **kw):
    p = pipe(**kw)
    p.worker._process_one = lambda sc: sc
    out = []
    error = None
    try:
        for seq, start in enumerate(range(0, len(x), bs)):
            out.extend(p.process_block(block(x[start:start+bs], seq, start)))
        out.extend(p.flush())
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        p.shutdown()
    return p, out, error


def warmup_partition():
    rng = np.random.default_rng(321)
    x = ((rng.normal(size=512) + 1j*rng.normal(size=512)) * math.sqrt(1e-4/2)).astype(np.complex64)
    x[128:192] += .5
    rows = []
    for bs in (1, 16, 64, 512):
        n = GaussianNoiseEstimator(window_size=16, warmup_samples=32)
        p, out, error = feed(x, bs, noise_estimator=n)
        rows.append(dict(block_size=bs, error=error, reference_power=n.noise_power_linear,
                         training_samples=n._warmup_samples_collected, events=[summary(s) for s in out],
                         expected_training_power=float(np.mean(np.abs(x[:32].astype(np.complex128))**2))))
    return rows


def adaptive_pipeline_partition():
    x = np.full(2048, .003, np.complex64)
    x[512:768] = .01
    rows = []
    for bs in (1, 16, 64, 2048):
        n = AdaptiveRobustNoiseEstimator(window_size=16, initial_noise_power_linear=1e-4,
                                        time_constant_s=1e-4, adaptation_window_size=64)
        p, out, error = feed(x, bs, noise_estimator=n)
        rows.append(dict(block_size=bs, error=error, reference_power=n.noise_power_linear,
                         accepted_noise_samples=n._total_accepted_samples,
                         events=[summary(s) for s in out]))
    return rows


def adaptation_before_full_window():
    rows = []
    x = np.full(512, .1, np.complex64)
    for bs in (64, 512):
        n = AdaptiveRobustNoiseEstimator(window_size=256, initial_noise_power_linear=1e-4,
                                        time_constant_s=1e-6, adaptation_window_size=64)
        p, out, error = feed(x, bs, noise_estimator=n, window_size=256, hop_size=64,
                              pre_trigger_samples=0)
        rows.append(dict(block_size=bs, error=error, reference_power=n.noise_power_linear,
                         accepted_noise_samples=n._total_accepted_samples, events=[summary(s) for s in out]))
    return rows


def gap_reference_and_residual():
    n = AdaptiveRobustNoiseEstimator(window_size=16, initial_noise_power_linear=1e-4,
                                    time_constant_s=1e-4)
    p = pipe(noise_estimator=n)
    try:
        p.process_block(block(np.full(64, .003)))
        before = n.noise_power_linear
        p.process_block(block(np.full(5, .003), 1, 64))
        residual_before = len(n._adaptation_residual)
        p.process_block(block(np.full(5, .003), 2, 100, dropped_samples_before=31))
        return dict(initial_reference=1e-4, learned_before_gap=before, reference_after_gap=n.noise_power_linear,
                    valid_after_gap=n.get_estimate().valid, residual_before_gap=residual_before,
                    residual_after_gap=len(n._adaptation_residual), worker_generation=p.worker._current_generation,
                    ring_generation=p.ring_buffer.stream_generation)
    finally:
        p.shutdown()


def stale_failure_atomicity_and_recovery():
    n = AdaptiveRobustNoiseEstimator(window_size=16, initial_noise_power_linear=1e-4,
                                    max_staleness_samples=16)
    p = pipe(noise_estimator=n, pre_trigger_samples=0)
    p.worker._process_one = lambda sc: sc
    try:
        p.process_block(block(np.full(32, .5)))
        before = p.ring_buffer.available_range()
        error = None
        try:
            p.process_block(block(np.zeros(32), 1, 32))
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        n.freeze(False)
        n.update_samples(np.full(64, .003, np.complex64))
        return dict(error=error, ring_before=before, ring_after=p.ring_buffer.available_range(),
                    pipeline_state=p.state.value, previous_block_start=p._previous_block.start_sample,
                    state_after_clean_update=n.state.value, accepted_samples=n._total_accepted_samples,
                    age_after_update=n._samples_since_last_accepted)
    finally:
        p.shutdown()


def configured_band_calibration():
    rng = np.random.default_rng(2608)
    def noise(n):
        return ((rng.normal(size=n)+1j*rng.normal(size=n))*math.sqrt(1e-4/2)).astype(np.complex64)
    train = noise(2_000_000)
    heldout = noise(2_004_096)
    cal = EmpiricalNoiseCalibrator(window_size=64, hop_size=64)
    policy = cal.create_policy(train, monitored_band_hz=(90000, 110000), settling_samples=4096)
    rows = []
    for label, power, use_policy in (("gamma_matched_reference", policy.mean_power_linear, False),
                                     ("empirical_matched_reference", policy.mean_power_linear, True),
                                     ("empirical_raw_reference_used_by_supplied_test", 1e-4, True)):
        d = EnergyDetector(window_size=64, hop_size=64, mode="configured_band",
                           monitored_band_hz=(90000, 110000), threshold_policy=policy if use_policy else None)
        ms = [m for m in d.process_windows(block(heldout), reference(power)) if m.start_sample >= 4096]
        hits = sum(m.above_on_threshold for m in ms)
        rows.append(dict(label=label, reference_power=power, windows=len(ms), threshold_hits=hits,
                         observed_window_exceedance_fraction=hits/len(ms), threshold_linear=10**(ms[-1].on_threshold_dbfs/10),
                         iid_binomial_interval_descriptive_only=stats.binomtest(hits, len(ms)).proportion_ci()._asdict()))
    # 0 dB CW SNR in the raw input; after filtering, emitter SNR is much higher.
    tone = (math.sqrt(1e-4)*np.exp(2j*np.pi*100000*np.arange(8192)/1e6)).astype(np.complex64)
    tone_rows = []
    for label, power in (("matched_reference",policy.mean_power_linear),("raw_reference",1e-4)):
        d = EnergyDetector(window_size=64,hop_size=64,mode="configured_band",
                           monitored_band_hz=(90000,110000),threshold_policy=policy)
        ms=[m for m in d.process_windows(block(tone),reference(power)) if m.start_sample>=4096]
        tone_rows.append(dict(label=label,full_amplitude_windows=len(ms),triggering_windows=sum(m.above_on_threshold for m in ms)))
    return dict(training_windows=policy.training_windows,matched_mean=policy.mean_power_linear,
                raw_to_filtered_ratio=1e-4/policy.mean_power_linear,requested_window_pfa=1e-4,
                noise_results=rows,tone_results=tone_rows,
                confidence_note="Intervals are descriptive IID binomial intervals; adjacent filtered windows may be correlated.")


def policy_binding():
    rng = np.random.default_rng(190)
    x=((rng.normal(size=160000)+1j*rng.normal(size=160000))*.01).astype(np.complex64)
    cal=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
    policy=cal.create_policy(x,settling_samples=0,monitored_band_hz=(90000,110000))
    rows=[]
    for label,kw in (("different_sample_rate",dict(sample_rate_hz=2e6)),
                     ("different_filter_order",dict(filter_order=2)),
                     ("different_hop",dict(hop_size=1)),
                     ("different_pfa",dict(pfa_on=1e-6,pfa_off=1e-3)),
                     ("filtered_policy_on_full_band",dict(mode="full_band",monitored_band_hz=None))):
        opts=dict(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1,mode="configured_band",
                  monitored_band_hz=(90000,110000),threshold_policy=policy)
        opts.update(kw)
        try:
            EnergyDetector(**opts)
            rows.append(dict(mismatch=label,accepted=True))
        except Exception as exc:
            rows.append(dict(mismatch=label,accepted=False,error=type(exc).__name__))
    # Same object marked valid with nonsensical threshold fields is admitted too.
    bad=dataclasses.replace(policy,t_on_linear=float("nan"),mean_power_linear=-1.0)
    admitted=False
    try:
        d=EnergyDetector(window_size=16,hop_size=16,mode="configured_band",
                         monitored_band_hz=(90000,110000),threshold_policy=bad)
        admitted=True
        ms=d.process_windows(block(np.full(32,.5)),reference(1e-4))
        invalid_policy_result=dict(accepted=True,triggering_windows=sum(m.above_on_threshold for m in ms),
                                   on_thresholds=[m.on_threshold_dbfs for m in ms])
    except Exception as exc:
        invalid_policy_result=dict(constructor_accepted=admitted,measurement_error=type(exc).__name__)
    return dict(mismatches=rows,invalid_numeric_policy=invalid_policy_result)


def calibration_dictionary_misbinding():
    rng=np.random.default_rng(71)
    x=((rng.normal(size=160000)+1j*rng.normal(size=160000))*.01).astype(np.complex64)
    a=EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
    result=a.calibrate(x)
    b=EmpiricalNoiseCalibrator(window_size=256,hop_size=256,pfa_on=1e-4,pfa_off=.01)
    rebound=b.create_policy(result)
    d=EnergyDetector(window_size=256,hop_size=256,threshold_policy=rebound)
    return dict(original_N=16,original_H=16,original_pfa=.01,claimed_N=rebound.window_size,
                claimed_H=rebound.hop_size,claimed_pfa=rebound.pfa_on,marked_valid=rebound.valid,
                detector_accepts=True,policy_training_windows=rebound.training_windows,
                original_empirical_threshold=result["empirical_t_on_linear"],rebound_threshold=rebound.t_on_linear)


def configured_band_adaptation_units():
    rng=np.random.default_rng(28)
    x=((rng.normal(size=32000)+1j*rng.normal(size=32000))*math.sqrt(1e-4/2)).astype(np.complex64)
    cal=EmpiricalNoiseCalibrator(window_size=64,hop_size=64,pfa_on=.01,pfa_off=.1)
    train=((rng.normal(size=640000)+1j*rng.normal(size=640000))*math.sqrt(1e-4/2)).astype(np.complex64)
    policy=cal.create_policy(train,monitored_band_hz=(90000,110000))
    n=AdaptiveRobustNoiseEstimator(window_size=64,initial_noise_power_linear=policy.mean_power_linear,
                                    time_constant_s=1e-4)
    p,out,error=feed(x,64,noise_estimator=n,window_size=64,hop_size=64,
                   pfa_on=.01,pfa_off=.1,detector_mode="configured_band",
                   monitored_band_hz=(90000,110000),threshold_policy=policy,pre_trigger_samples=0)
    return dict(matched_initial_reference=policy.mean_power_linear,reference_after_raw_idle_training=n.noise_power_linear,
                expected_raw_mean=1e-4,accepted_samples=n._total_accepted_samples,events=len(out),error=error)


def single_block_policy_wrong_n():
    rng=np.random.default_rng(27)
    x=((rng.normal(size=160000)+1j*rng.normal(size=160000))*.01).astype(np.complex64)
    c=EmpiricalNoiseCalibrator(window_size=256,hop_size=256,pfa_on=.01,pfa_off=.1)
    policy=c.create_policy(x,settling_samples=0)
    d=EnergyDetector(window_size=256,hop_size=256,pfa_on=.01,pfa_off=.1,threshold_policy=policy)
    m=d.process(block(np.array([.02],np.complex64)),reference(policy.mean_power_linear))
    threshold=10**(m.on_threshold_dbfs/10)
    return dict(configured_N=256,actual_samples=1,threshold_linear=threshold,
                requested_pfa=.01,actual_one_sample_gaussian_pfa=math.exp(-threshold/policy.mean_power_linear))


def warmup_timestamp():
    n=GaussianNoiseEstimator(window_size=16,warmup_samples=32)
    p=pipe(noise_estimator=n,pre_trigger_samples=0)
    p.worker._process_one=lambda s:s
    try:
        p.process_block(block(np.full(32,.01),timestamp_ns=1_000_000_000))
        p.process_block(block(np.full(64,.5),1,32))
        out=p.flush()
        return dict(timestamp_after_timestamped_warmup=out[0].event.timestamp_ns,
                    expected_timestamp_ns=1_000_000_000+out[0].event.event_start_sample*1000)
    finally:p.shutdown()


def worker_barrier():
    w=FeatureAnalysisWorker(synchronous=False)
    entered=threading.Event();release=threading.Event();finished=threading.Event()
    def process(s):
        entered.set()
        release.wait(2)
        finished.set()
        return s
    w._process_one=process
    marker=object()
    try:
        w.submit(marker)
        if not entered.wait(1):raise RuntimeError("Worker did not start probe job")
        w.reset()
        release.set()
        if not finished.wait(1):raise RuntimeError("Worker did not finish probe job")
        w.shutdown(timeout=2)
        return dict(old_inflight_result_suppressed=len(w.processed_results)==0,
                    delivered_result=w.get_result(block=False) is not None,
                    queue_unfinished_tasks=w._queue.unfinished_tasks,state=w.state.value)
    finally:
        release.set()
        if w.state.value != "CLOSED":w.shutdown(timeout=2)


def worker_sink_reentrancy():
    code='''import sys, threading
sys.path.insert(0, sys.argv[1])
from signal_processing.runtime import FeatureAnalysisWorker
entered=threading.Event(); finished=threading.Event()
w=None
def sink(s):
    entered.set()
    w.processed_results
    finished.set()
w=FeatureAnalysisWorker(synchronous=False,result_sink=sink)
w._process_one=lambda s:s
w.submit(object())
assert entered.wait(2)
print("callback_entered",flush=True)
print("callback_completed",finished.wait(.25),flush=True)
print("worker_alive",w._thread.is_alive(),flush=True)
'''
    r=subprocess.run([sys.executable,"-c",code,str(SRC)],capture_output=True,text=True,timeout=5)
    return dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr,
                note="A result callback reads the public processed_results property; callback runs under the same non-reentrant lock.")


def worker_controlled_atomic_admission():
    w=FeatureAnalysisWorker(synchronous=False)
    w._process_one=lambda s:s
    original=w._queue.put_nowait
    admission_entered=threading.Event();release=threading.Event();closed=threading.Event()
    errors=[];accepted=[]
    def put(item):
        admission_entered.set()
        if not release.wait(2):raise RuntimeError("Probe staging timed out")
        return original(item)
    w._queue.put_nowait=put
    def submit():
        try:accepted.append(w.submit(object()))
        except Exception as exc:errors.append(str(exc))
    def close():
        try:w.shutdown(timeout=2)
        except Exception as exc:errors.append(str(exc))
        finally:closed.set()
    a=threading.Thread(target=submit);b=threading.Thread(target=close)
    a.start()
    if not admission_entered.wait(1):raise RuntimeError("Admission was not staged")
    b.start()
    closed_before_admission_release=closed.wait(.05)
    release.set();a.join(2);b.join(2)
    return dict(closed_before_admission_release=closed_before_admission_release,
                accepted=len(accepted),results=len(w.processed_results),errors=errors,
                queued=w._queue.qsize(),unfinished=w._queue.unfinished_tasks,state=w.state.value)


def worker_shutdown_timeout_budget():
    w=FeatureAnalysisWorker(synchronous=False,max_queue_size=1)
    entered=threading.Event();release=threading.Event()
    def process(s):
        entered.set();release.wait(2);return s
    w._process_one=process
    w.submit(object())
    if not entered.wait(1):raise RuntimeError("Worker did not start timeout probe")
    w.submit(object())
    t=time.monotonic();error=None
    try:w.shutdown(drain=True,timeout=.1)
    except Exception as exc:error=type(exc).__name__
    elapsed=time.monotonic()-t
    release.set();w.shutdown(timeout=2)
    return dict(requested_timeout_s=.1,observed_elapsed_s=elapsed,error=error,
                final_state=w.state.value,unfinished=w._queue.unfinished_tasks)


def staleness_after_fresh_idle_block():
    n=AdaptiveRobustNoiseEstimator(window_size=16,initial_noise_power_linear=1e-4,
                                    max_staleness_samples=64,time_constant_s=.001)
    p=pipe(noise_estimator=n,pre_trigger_samples=0)
    try:
        p.process_block(block(np.full(128,.003,np.complex64)))
        return dict(accepted_clean_noise_samples=n._total_accepted_samples,
                    reported_age_after_fresh_block=n._samples_since_last_accepted,
                    configured_maximum_age=64,state=n.state.value)
    finally:p.shutdown()


def narrow_filter_delay():
    rows=[]
    for width in (20000,1000,100,10):
        with warnings.catch_warnings(record=True) as caught:
            d=EnergyDetector(window_size=64,hop_size=64,mode="configured_band",
                             monitored_band_hz=(100000-width/2,100000+width/2),sample_rate_hz=1e6)
        # Direct differentiation of each SOS section's phase at DC. Summing
        # section delays avoids expanding a narrow filter into a poorly
        # conditioned high-order denominator polynomial.
        dc_delay=sum((s[1]+2*s[2])/sum(s[:3])-(s[4]+2*s[5])/sum(s[3:]) for s in d._sos)
        rows.append(dict(bandwidth_hz=width,reported_filter_delay=d.filter_delay_samples,
                         sos_section_dc_delay=float(dc_delay),warnings=[str(w.message) for w in caught]))
    return rows


class PSDContext:
    def __init__(self,p,f=None):
        self.p=np.asarray(p,float)
        self.f=np.arange(len(p),dtype=float) if f is None else np.asarray(f,float)
    def psd(self):
        class PSD:pass
        s=PSD();s.frequency_offset_hz=self.f;s.power_linear=self.p
        return s


def bwsummary(fset):
    return {k:dict(value=v.value,valid=v.valid,reason=v.reason) for k,v in fset.features.items()}


def bandwidth_checks():
    rows={}
    # Large excess signal with a monotone lobe clipped at the Nyquist/observation edge;
    # no explicit ROI is set, so source spectrum boundaries still matter.
    p=np.array([101.,81.,51.,11.,1.,1.,1.,1.,1.])
    config=BandwidthFeatureConfig(mode="emitter",noise_floor_dbfs_per_hz=0,min_snr_db=0)
    rows["observation_edge_without_explicit_roi"]=bwsummary(BandwidthFeatureExtractor(config).extract(PSDContext(p)))
    # Broad equal plateau: no distinct dominant lobe, although a 3-dB interval exists mathematically.
    p=np.ones(41);p[10:31]=101
    config=BandwidthFeatureConfig(mode="emitter",roi_hz=(0,40),noise_floor_dbfs_per_hz=0)
    rows["flat_wideband_emitter_plateau"]=bwsummary(BandwidthFeatureExtractor(config).extract(PSDContext(p)))
    f=np.arange(-10,11,dtype=float)
    excess=100*np.exp(-f*f/8)
    config=BandwidthFeatureConfig(mode="emitter",roi_hz=(-3,3),noise_floor_dbfs_per_hz=0)
    rows["occupied_tail_clipped_below_half_power"]=dict(
        actual_excess_fraction_outside_roi=float(excess[np.abs(f)>3].sum()/excess.sum()),
        features=bwsummary(BandwidthFeatureExtractor(config).extract(PSDContext(1+excess,f))))
    # Explicit raw-PSD mode still noise-subtracts if a noise floor is configured.
    p=np.ones(100);p[45:55]=11
    no_ref=BandwidthFeatureExtractor(BandwidthFeatureConfig(mode="raw_psd")).extract(PSDContext(p))
    with_ref=BandwidthFeatureExtractor(BandwidthFeatureConfig(mode="raw_psd",noise_floor_dbfs_per_hz=0)).extract(PSDContext(p))
    rows["raw_psd_reference_changes_descriptor"]=dict(no_reference=bwsummary(no_ref),with_reference=bwsummary(with_ref))
    return rows


def custom_extractor_policy_collision():
    x=np.exp(2j*np.pi*.1*np.arange(128)).astype(np.complex64)
    evt=SignalEvent(event_id="policy-probe",stream=STREAM,iq=x,event_start_sample=0,event_end_sample=128,
                    detected_start_sample=0,detected_end_sample=128,trigger_sample=0)
    context=SignalContext(event=evt)
    rows=[]
    for mode in ("raw_psd","emitter"):
        ex=BandwidthFeatureExtractor(BandwidthFeatureConfig(mode=mode))
        aggregator=FeatureAggregator(FeatureRegistry([ex]))
        result=aggregator.extract_analysis_result(context)
        rows.append(dict(actual_extractor_mode=mode,recorded_mode=result.policy.bandwidth_mode,
                         policy_id=result.policy.policy_id,features=bwsummary(ex.extract(context))))
    return rows


def strict_validation():
    rows=[]
    for key,value in (("adaptation_window_size",0),("adaptation_window_size",-4),
                      ("adaptation_window_size",True),("adaptation_window_size",1.5),
                      ("max_staleness_samples",-1),("max_staleness_samples",True)):
        try:
            n=AdaptiveRobustNoiseEstimator(initial_noise_power_linear=1e-4,**{key:value})
            error=None
            try:n.update_samples(np.ones(64,np.complex64))
            except Exception as exc:error=type(exc).__name__+": "+str(exc)
            rows.append(dict(parameter=key,value=value,constructor_accepted=True,update_error=error))
        except Exception as exc:
            rows.append(dict(parameter=key,value=value,constructor_accepted=False,error=type(exc).__name__))
    for floor in (0,-1,float("nan"),True):
        try:
            EnergyDetector(conversion_floor_linear=floor)
            rows.append(dict(parameter="detector.conversion_floor_linear",value=floor,constructor_accepted=True))
        except Exception as exc:
            rows.append(dict(parameter="detector.conversion_floor_linear",value=floor,constructor_accepted=False,error=type(exc).__name__))
    return rows


def direct_detector_invalid_noise_mutation():
    d=EnergyDetector(window_size=16,hop_size=8,mode="configured_band",monitored_band_hz=(90000,110000))
    before_phase=d._phase_accumulator;before_zi=d._sos_zi.copy()
    invalid=NoiseEstimate(stream_id=STREAM.stream_id,block_sequence_number=0,noise_power_dbfs=float("nan"),valid=False,reason="Probe: no calibrated reference")
    error=None
    try:d.process_windows(block(np.ones(25)),invalid)
    except Exception as exc:error=type(exc).__name__
    return dict(error=error,phase_mutated=d._phase_accumulator!=before_phase,filter_state_mutated=not np.array_equal(d._sos_zi,before_zi))


def metadata_first_center():
    p=pipe()
    try:
        info=StreamInfo(STREAM.stream_id,STREAM.source_id,1e6,center_frequency_hz=100e6)
        p.process_block(block(np.zeros(32),info=info))
        return dict(configured_center=None,incoming_center=100e6,accepted=True)
    finally:p.shutdown()


def quality_and_timing():
    p=pipe(skip_incomplete_quality=False,pre_trigger_samples=0)
    try:
        p.process_block(block(np.full(64,.5)))
        s=p.flush()[0]
        q=s.analysis_result.capture_quality
        return dict(termination=s.termination_reason,complete=s.event.complete,dropped=s.event.dropped_samples,
                    feature_status=s.feature_status,reported_continuity=q.is_continuous,
                    actual_samples_contiguous=True,quality_warnings=q.warnings)
    finally:p.shutdown()


def eof_unwindowed_tail():
    rows=[]
    x=np.full(69,.5,np.complex64)
    for bs in (1,16,69):
        p,out,error=feed(x,bs,pre_trigger_samples=0)
        s=out[-1]
        rows.append(dict(block_size=bs,error=error,actual_input_samples=len(x),ring_end=p.ring_buffer.available_range()[1],
                         captured_end=s.event.event_end_sample,captured_samples=s.event.num_samples,
                         omitted_available_tail=len(x)-s.event.event_end_sample,termination=s.termination_reason))
    return rows


if __name__=="__main__":
    cases=(warmup_partition,adaptive_pipeline_partition,adaptation_before_full_window,
           gap_reference_and_residual,stale_failure_atomicity_and_recovery,configured_band_calibration,
           policy_binding,calibration_dictionary_misbinding,configured_band_adaptation_units,
           single_block_policy_wrong_n,warmup_timestamp,worker_barrier,worker_sink_reentrancy,
           bandwidth_checks,custom_extractor_policy_collision,strict_validation,
           direct_detector_invalid_noise_mutation,metadata_first_center,quality_and_timing,eof_unwindowed_tail,
           worker_controlled_atomic_admission,worker_shutdown_timeout_budget,staleness_after_fresh_idle_block,
           narrow_filter_delay)
    for fn in cases:
        case(fn.__name__,fn)
    destination=Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(RESULTS,indent=2,default=str))
    print(json.dumps(RESULTS,indent=2,default=str))
