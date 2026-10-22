"""Hardware-facing replay diagnostic, separate from the synthetic RNG schedule.

Same IQ, 1 MS/s, 1024-sample blocks, N256/H64, ring4096, queue16, real
on-demand features. RNG is offline, not on the acquisition CPU. Deadline expiry
is STILL enforced at one block interval and reported, never removed from pass.
"""
import argparse
import collections
import dataclasses
import json
from pathlib import Path
import threading
import time



def prepare(path,seconds):
    from benchmarks.process_source import _ScheduledIQ
    if path.exists(): raise FileExistsError(path)
    total=round(seconds*1e6)
    generator=_ScheduledIQ(1024)
    with path.open('wb') as output:
        for start in range(0,total,1024): generator.generate(start,min(1024,total-start)).tofile(output)
    return dict(samples=total,scope='offline seed-730 IQ preparation, not a timing result')


def run(path, *, scheduled=True):
    import numpy as np
    from signal_processing.core import StreamInfo
    from signal_processing.detection import AdaptiveRobustNoiseEstimator, POSITIVE_SNR_IID
    from signal_processing.runtime import StreamingCapturePipeline, ProcessSourceRunner, ReplayFileFactory
    from benchmarks.audit_pipeline_runtime import environment
    stream=StreamInfo('bounded-acquisition','scheduled-recording',1e6)
    total=path.stat().st_size//8
    before=time.perf_counter()
    noise=AdaptiveRobustNoiseEstimator(window_size=256,initial_noise_power_linear=1e-4,
                                       time_constant_s=.5,adaptation_window_size=1024)
    pipe=StreamingCapturePipeline(stream,noise_estimator=noise,window_size=256,hop_size=64,
        ring_capacity=4096,start_persistence=2,end_persistence=2,pre_trigger_samples=256,
        pre_signal_margin_samples=0,post_signal_samples=0,min_post_trigger_samples=0,
        max_event_samples=4096,fixed_quiet_batch=True,synchronous_analysis=False,
        feature_analysis_mode='on_demand',analysis_backend='process_pool',process_workers=1,
        max_worker_queue_size=4,result_delivery='callbacks',result_order='completion',
        detection_profile_id=POSITIVE_SNR_IID.profile_id,boundary_uncertainty_samples=256)
    factory=ReplayFileFactory(str(path.resolve()),stream,1024,scheduled,.001024)
    runner=ProcessSourceRunner(factory,stream,capacity=16,max_block_samples=1024)
    lock=threading.Lock()
    pending={}
    outcomes=collections.Counter()
    duplicates=0
    last_terminal=-1
    requested=raw=unrequested=0
    errors=[]
    latency=collections.deque(maxlen=65536)
    processing=collections.deque(maxlen=65536)
    maximum=0.
    last_classification=0.
    def terminal(sc):
        nonlocal duplicates,last_terminal
        key=(sc.event_id,sc.stream_epoch)
        index=int(sc.event_id.rsplit('_',1)[1])
        with lock:
            if key not in pending or index<=last_terminal: duplicates+=1
            pending.pop(key,None)
            last_terminal=max(last_terminal,index)
            outcomes[sc.feature_status]+=1
    pipe.worker.add_result_sink(terminal)
    def observe(captures):
        nonlocal raw,requested,unrequested,last_classification
        for sc in captures:
            raw+=1
            position=sc.event.event_end_sample/1e6
            purpose='classification' if position-last_classification>=2 else 'ui'
            if purpose=='classification': last_classification=position
            key=(sc.event_id,sc.stream_epoch)
            with lock:
                if len(pending)>8: raise RuntimeError('Bounded terminal ledger exhausted')
                pending[key]=True
            result=pipe.request_features(sc,purpose=purpose)
            if result.feature_status=='skipped':
                with lock: pending.pop(key)
                unrequested+=1
            else:
                requested+=1
                if result.feature_status!='pending': terminal(result)
    initialization=time.perf_counter()-before
    source_start=time.perf_counter()
    runner.start()
    source_startup=time.perf_counter()-source_start
    started=time.perf_counter()
    try:
        while (lease:=runner.get(timeout=5)) is not None:
            try:
                tick=time.perf_counter()
                captures=pipe.process_block(lease.envelope.block)
                observe(captures)
                finished=time.perf_counter_ns()
                processing.append((time.perf_counter()-tick)*1000)
                if lease.envelope.scheduled_monotonic_ns is not None:
                    lag=max(0.,(finished-lease.envelope.scheduled_monotonic_ns)/1e6)
                    latency.append(lag); maximum=max(maximum,lag)
            except Exception:
                runner.ack(lease,outcome='failed')
                raise
            runner.ack(lease)
        observe(pipe.flush())
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        observe(getattr(exc,'completed_captures',()))
    finally:
        for shutdown in (lambda:runner.shutdown(drain=False,timeout=5),lambda:pipe.shutdown(timeout=10)):
            try: shutdown()
            except Exception as exc: errors.append(f'cleanup: {type(exc).__name__}: {exc}')
    source=runner.source_snapshot()
    rs=runner.snapshot()
    balanced=(rs.accounting_balanced and source.total_samples==source.read_samples+source.unread_samples
        and source.read_samples==source.returned_samples+source.expired_samples+source.pending_samples
        and source.returned_samples==rs.read_samples)
    eligible=total-source.expired_samples
    delivery=(not pending and not duplicates and sum(outcomes.values())==requested
        and not outcomes['failed'] and not outcomes['dropped_overload'] and not pipe.worker.sink_error_count
        and not pipe.worker.discarded_old_generation_count and not pipe.worker.cancelled_count)
    recovered=(rs.state=='CLOSED' and pipe.worker.state.value=='CLOSED' and not pipe.worker.unfinished_work
               and not rs.outstanding_deliveries)
    processing_ok=(not errors and balanced and delivery and recovered and eligible>0
        and rs.processed_samples==eligible and not rs.queue_dropped_samples and not rs.failed_samples)
    lossless=processing_ok and rs.processed_samples==total
    return dict(scope=__doc__,scheduled=scheduled,environment=environment(),
        source_backend='spawn_process_bounded_acknowledged_slots',input=str(path),
        initialization_seconds=initialization,source_startup_seconds=source_startup,
        elapsed_seconds=time.perf_counter()-started,scheduled_samples=total,consumed_samples=rs.processed_samples,
        acquisition_fraction=rs.processed_samples/total,source_eligible_samples=eligible,
        processing_acquisition_fraction=rs.processed_samples/eligible if eligible else None,
        producer_deadline_dropped_samples=source.expired_samples,queue_dropped_samples=rs.queue_dropped_samples,
        sample_accounting_balanced=balanced,latency_ms_percentiles=np.percentile(latency,[50,95,99,100]).tolist() if latency else [],
        max_commit_latency_ms=maximum,processing_ms_percentiles=np.percentile(processing,[50,95,99,100]).tolist() if processing else [],
        raw_captures=raw,requested_captures=requested,unrequested_captures=unrequested,outcomes=dict(outcomes),
        unresolved=len(pending),duplicate_results=duplicates,feature_requests=dataclasses.asdict(pipe.feature_request_snapshot),
        delivery_passed=delivery,recovered=recovered,unfinished=pipe.worker.unfinished_work,
        runner=dataclasses.asdict(rs),source=dataclasses.asdict(source),noise_age_samples=noise.staleness_samples,
        retention=pipe.retention_snapshot(),processing_scope_passed=processing_ok,acquisition_passed=lossless,
        sub10ms_processing_passed=bool(scheduled and processing_ok and maximum<10),
        sub10ms_lossless_passed=bool(scheduled and lossless and maximum<10),errors=errors)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','run'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--unpaced',action='store_true')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=prepare(args.input,args.seconds) if args.mode=='prepare' else run(args.input,scheduled=not args.unpaced)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(int(result.get('sub10ms_lossless_passed') is False))
