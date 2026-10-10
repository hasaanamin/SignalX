"""Same scheduled IQ workload, direct bounded source-to-capture process handoff."""
import argparse
import collections
import gc
import heapq
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')


def run(seconds, *, slow_sink=0., adaptive=False, pin_primary_host=False, affinity_layout='split', priority='normal', profile='compatibility', fail_sink=False, feature_processes=4, result_order='submission', physical_affinity=False, high_qos=False):
    import numpy as np
    from signal_processing.core import StreamInfo
    from signal_processing.runtime import FeatureAnalysisWorker
    from signal_processing.runtime.capture_stream import CaptureStreamProcess
    from benchmarks.process_source import ProcessScheduledSource
    stream=StreamInfo('bounded-acquisition','scheduled-synthetic',1e6)
    total=int(seconds*1e6)
    start_init=time.perf_counter()
    worker=FeatureAnalysisWorker(synchronous=False,analysis_backend='process_pool',max_queue_size=4,
                                result_delivery='callbacks',skip_incomplete_quality=True,
                                process_workers=feature_processes,result_order=result_order)
    source=capture=None
    qos=None
    errors=[]
    lock=threading.Lock()
    balances={}
    terminal_due={}
    terminal_latencies=collections.deque(maxlen=65536)
    max_terminal_latency=0.
    slowest_terminal=[]
    outcomes=collections.Counter()
    raw_count=0
    delivered_counter=0
    duplicate=0
    # Event IDs are the capture engine's monotonically numbered identities.
    # A bounded set tracks completion holes; settled IDs need no retained IQ.
    settled=0
    complete=set()
    def terminal(sc):
        nonlocal delivered_counter,duplicate,settled,max_terminal_latency
        index=int(sc.event_id.rsplit('_',1)[1])
        key=(sc.event_id,sc.stream_epoch)
        with lock:
            if len(complete)>=512:raise RuntimeError('Bounded terminal identity ledger exhausted')
            delivered_counter+=1
            outcomes[sc.feature_status]+=1
            if index<=settled or index in complete:
                duplicate+=1
            complete.add(index)
            while settled+1 in complete:
                settled+=1;complete.remove(settled)
            balances[key]=balances.get(key,0)-1
            if not balances[key]:del balances[key]
            due=terminal_due.pop(key,None)
            if due is not None:
                latency=1000*(time.perf_counter()-due)
                terminal_latencies.append(latency)
                max_terminal_latency=max(max_terminal_latency,latency)
                item=(latency,key)
                if len(slowest_terminal)<16:heapq.heappush(slowest_terminal,item)
                elif item>slowest_terminal[0]:heapq.heapreplace(slowest_terminal,item)
        if slow_sink:time.sleep(slow_sink)
        if fail_sink:raise RuntimeError('Injected required-sink failure')
    worker.add_result_sink(terminal)
    latencies=collections.deque(maxlen=65536)
    processing=collections.deque(maxlen=65536)
    consumed=misses=blocks=0
    capture_cpu=0.
    snapshot={}
    peak_worker=0
    delivery_backpressure_seconds=0.
    max_latency=0.
    latency_first=collections.deque(maxlen=1024)
    slowest_blocks=[]
    started=None
    init=None
    options=noise={}
    affinity_map=None
    original_affinity=None
    gc_times={str(g):dict(count=0,total_seconds=0.,max_seconds=0.) for g in range(3)}
    gc_start=0.
    def gc_probe(phase,info):
        nonlocal gc_start
        if phase=='start':gc_start=time.perf_counter()
        elif started is not None:
            elapsed=time.perf_counter()-gc_start
            entry=gc_times[str(info['generation'])]
            entry['count']+=1;entry['total_seconds']+=elapsed
            entry['max_seconds']=max(entry['max_seconds'],elapsed)
    gc.callbacks.append(gc_probe)
    try:
        source=ProcessScheduledSource(stream,total,1024,16)
        noise=(dict(window_size=64,initial_noise_power_linear=1e-4,time_constant_s=.5,adaptation_window_size=1024)
               if adaptive else dict(window_size=64,calibrated_noise_power_linear=1e-4))
        options=dict(window_size=64,hop_size=32,ring_capacity=4096,
                start_persistence=2,end_persistence=2,pre_trigger_samples=0,pre_signal_margin_samples=0,
                post_signal_samples=0,min_post_trigger_samples=0,max_event_samples=4096,fixed_quiet_batch=True)
        if profile=='positive_snr':
            from signal_processing.detection import POSITIVE_SNR_IID
            if not adaptive:raise ValueError('positive_snr requires adaptive reference')
            options.update(window_size=256,hop_size=64,pre_trigger_samples=256,
                detection_profile_id=POSITIVE_SNR_IID.profile_id,boundary_uncertainty_samples=256)
            noise['window_size']=256
        capture=CaptureStreamProcess(source._queue,stream,noise_mode='adaptive' if adaptive else 'fixed',
            noise_options=noise,pipeline_options=options)
        if pin_primary_host:
            import psutil
            original_affinity=psutil.Process().cpu_affinity()
            if not set(range(12)).issubset(original_affinity):
                raise ValueError('This experiment requires the declared 12-logical-CPU i5-1235U host')
            feature_cpus=((2,5,6,7) if affinity_layout=='split' else (5,6,7,8))[:feature_processes]
            affinity_map={os.getpid():[3 if affinity_layout=='split' else 2],source._process.pid:[4],capture._process.pid:[0]}
            affinity_map.update({pid:[cpu] for pid,cpu in zip(worker._process_backend.pids,feature_cpus,strict=True)})
            if physical_affinity:
                affinity_map[capture.pid]=[0,1]
                affinity_map[os.getpid()]=[2,3]
                if affinity_layout=='split':affinity_map[worker.backend_pids[0]]=[2,3]
            if affinity_layout=='analysis_p':
                # i5-1235U: give measured DSP work the two physical P cores;
                # source and parent codec/delivery run on distinct E cores.
                affinity_map={os.getpid():[5],source._process.pid:[4],capture.pid:[2,3]}
                placements=([0,1],[6],[7],[8])
                affinity_map.update({pid:cpus for pid,cpus in zip(worker.backend_pids,
                    placements[:feature_processes],strict=True)})
            for pid,cpus in affinity_map.items():psutil.Process(pid).cpu_affinity(cpus)
        if priority=='above_normal':
            import psutil
            if os.name!='nt':raise ValueError('above_normal is a Windows process priority experiment')
            for pid in (source._process.pid,capture._process.pid):
                psutil.Process(pid).nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        if high_qos:
            from benchmarks.windows_qos import OwnedHighQoS
            qos=OwnedHighQoS([os.getpid(),source._process.pid,capture.pid,*worker.backend_pids])
        init=time.perf_counter()-start_init
        started=time.perf_counter();source.start(started)
        while not capture.done:
            try: record=capture.get(.05)
            except queue.Empty:
                if not source.is_alive() and source._queue.empty() and time.perf_counter()-started>seconds+2:
                    raise RuntimeError('Source ended without terminal capture record')
                continue
            try:
                snapshot=record['snapshot']
                if record['sequence']>=0:
                    # Processing progress belongs to the capture process, not to
                    # successful submission/delivery by this downstream pump.
                    n=record['num_samples']
                    if record['error']:
                        n=min(n,max(0,snapshot['consumed_end_sample']-record['start_sample']))
                    consumed+=n;blocks+=1
                    latency=time.perf_counter()-record['arrival']
                    latencies.append(latency*1000)
                    max_latency=max(max_latency,latency*1000)
                    item=(latency*1000,record['sequence'],record['processing_seconds']*1000)
                    if len(slowest_blocks)<16:heapq.heappush(slowest_blocks,item)
                    elif item>slowest_blocks[0]:heapq.heapreplace(slowest_blocks,item)
                    if blocks<=1024:latency_first.append(latency*1000)
                    processing.append(record['processing_seconds']*1000)
                    capture_cpu+=record['cpu_seconds']
                    misses+=latency>1024/1e6
                for sc in record['captures']:
                    raw_count+=1
                    with lock:
                        if len(balances)>=512:raise RuntimeError('Bounded pending identity ledger exhausted')
                        key=(sc.event_id,sc.stream_epoch)
                        balances[key]=balances.get(key,0)+1
                        if not balances[key]:del balances[key]
                        terminal_due[key]=started+sc.event.event_end_sample/stream.sample_rate_hz
                for sc in record['captures']:
                    before_capacity=time.perf_counter()
                    # This is the downstream completion pump, not the source or
                    # capture executor. Its outstanding completion slot remains
                    # leased while waiting; both upstream buffers stay bounded.
                    if not worker.wait_for_capacity(2.):
                        raise TimeoutError('Required analysis delivery capacity did not recover')
                    delivery_backpressure_seconds+=time.perf_counter()-before_capacity
                    submitted=worker.submit(sc)
                    if submitted.feature_status!='pending':terminal(submitted)
                if record['error']:raise RuntimeError(record['error'])
                peak_worker=max(peak_worker,worker.unfinished_work)
            finally:capture.task_done()
        source.done=True
    except Exception as exc:errors.append(f'{type(exc).__name__}: {exc}')
    finally:
        if source:source.stop.set();source.join(2)
        if capture:
            try:capture.close()
            except Exception as exc:errors.append(f'capture cleanup: {exc}')
        try:worker.shutdown(timeout=max(5,6*slow_sink))
        except Exception as exc:errors.append(f'feature cleanup: {exc}')
        if original_affinity is not None:
            import psutil
            psutil.Process().cpu_affinity(original_affinity)
        if qos is not None:
            try:qos.close()
            except Exception as exc:errors.append(f'QoS restoration: {exc}')
        gc.callbacks.remove(gc_probe)
    stats={}
    if source:
        try:stats=source.finish()
        except Exception as exc:errors.append(f'source final accounting: {type(exc).__name__}: {exc}')
    errors+=stats.pop('errors',[])
    input_remaining=source._queue.qsize() if source else None
    input_unfinished=source._queue.unfinished_tasks if source else None
    if source:
        try:source.close()
        except Exception as exc:errors.append(f'source cleanup: {type(exc).__name__}: {exc}')
    loss=total-consumed
    accounting=(loss==stats.get('producer_deadline_dropped_samples',0)+stats.get('queue_dropped_samples',0))
    delivery_ok=(not balances and delivered_counter==raw_count and not duplicate and
        not outcomes['failed'] and not outcomes['dropped_overload'] and not worker.sink_error_count and not worker.wakeup_errors
        and not worker.discarded_old_generation_count and not worker.cancelled_count)
    recovery=(worker.state.value=='CLOSED' and worker.unfinished_work==0 and capture is not None and capture.drained
              and input_remaining==0 and input_unfinished==0)
    return dict(**stats,errors=errors,backend='direct_capture_process_and_feature_process_pool',
        feature_processes=feature_processes,result_order=result_order,
        high_qos=qos.observations if qos else None,
        initialization_seconds=init,scheduled_samples=total,consumed_samples=consumed,lost_samples=loss,
        acquisition_fraction=consumed/total,sample_accounting_balanced=accounting,
        blocks=blocks,deadline_misses=misses,capture_cpu_seconds=capture_cpu,
        elapsed_seconds=time.perf_counter()-started if started else None,
        latency_ms_percentiles=np.percentile(latencies,[50,95,99,100]).tolist() if latencies else [],
        processing_ms_percentiles=np.percentile(processing,[50,95,99,100]).tolist() if processing else [],
        max_capture_delivery_latency_ms=max_latency,
        terminal_latency_ms_percentiles=np.percentile(terminal_latencies,[50,95,99,100]).tolist() if terminal_latencies else [],
        max_terminal_latency_ms=max_terminal_latency,
        slowest_blocks=sorted(slowest_blocks,reverse=True),slowest_terminal=sorted(slowest_terminal,reverse=True),
        first_1024_latency_mean_ms=float(np.mean(latency_first)) if latency_first else None,
        last_1024_latency_mean_ms=float(np.mean(list(latencies)[-1024:])) if latencies else None,
        raw_captures=raw_count,terminal_results=delivered_counter,outcomes=dict(outcomes),
        duplicate_results=duplicate,unresolved=len(balances),delivery_passed=delivery_ok,
        sink_errors=worker.sink_error_count,failed_analyses=worker.failed_count,
        wakeup_errors=worker.wakeup_errors,
        discarded_generations=worker.discarded_old_generation_count,cancelled_jobs=worker.cancelled_count,
        recovered=recovery,unfinished=worker.unfinished_work,worker_queue_peak=peak_worker,
        retained_results=len(worker.processed_results),snapshot=snapshot,
        child_feature_seconds=worker._process_backend.child_analysis_seconds,
        child_feature_cpu_seconds=worker._process_backend.child_cpu_seconds,
        parent_feature_cpu_seconds=worker.parent_worker_cpu_seconds,
        delivery_backpressure_seconds=delivery_backpressure_seconds,
        acquisition_over_90=consumed/total>.9,
        strict_passed=not errors and accounting and loss==0 and misses==0 and delivery_ok and recovery,
        candidate_passed=not errors and accounting and consumed/total>.9 and not stats.get('queue_dropped_samples',0) and delivery_ok and recovery,
        adaptive=adaptive,profile=profile,pipeline_options=options,noise_options=noise,
        affinity_map=affinity_map,acquisition_process_priority=priority,parent_gc=gc_times,
        quantile_scope='last 65536 blocks; counts and losses cover entire run')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=5);p.add_argument('--output',required=True)
    p.add_argument('--slow-sink',type=float,default=0);p.add_argument('--adaptive',action='store_true')
    p.add_argument('--pin-primary-host',action='store_true')
    p.add_argument('--affinity-layout',choices=('split','separate','analysis_p'),default='split')
    p.add_argument('--priority',choices=('normal','above_normal'),default='normal')
    p.add_argument('--profile',choices=('compatibility','positive_snr'),default='compatibility')
    p.add_argument('--fail-sink',action='store_true')
    p.add_argument('--feature-processes',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--result-order',choices=('submission','completion'),default='submission')
    p.add_argument('--physical-affinity',action='store_true')
    p.add_argument('--high-qos',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=run(a.seconds,slow_sink=a.slow_sink,adaptive=a.adaptive,pin_primary_host=a.pin_primary_host,affinity_layout=a.affinity_layout,priority=a.priority,profile=a.profile,fail_sink=a.fail_sink,feature_processes=a.feature_processes,result_order=a.result_order,physical_affinity=a.physical_affinity,high_qos=a.high_qos)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));raise SystemExit(not result['candidate_passed'])
