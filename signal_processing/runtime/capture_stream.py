"""Bounded source-to-capture process boundary, with no per-block parent RPC.

The supplied SPSC SharedIQTransport has exactly one consumer: this child.
Input blocks retain their original size and coordinates. Completion records are
bounded by a semaphore and acknowledged by the parent, never by queue.empty().
Feature execution and arbitrary callbacks belong to the parent application.
"""
import dataclasses
import gc
import heapq
import multiprocessing
import pickle
import queue
import struct
import time

import numpy as np

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import GaussianNoiseEstimator, AdaptiveRobustNoiseEstimator
from .pipeline import StreamingCapturePipeline
from .isolated_capture import _encode_sidecar, _decode_sidecar, _snapshot
from ._completion_slots import CompletionSlots


# Empty OPEN progress has a fixed scalar envelope: avoid rebuilding/pickling
# nested dictionaries ~1,000 times/s. Captures/faults/EOF retain the exact full
# record. Input blocks, progress frequency, bounds and acknowledgements do not
# change. The magic cannot be a pickle protocol-5 header.
_PROGRESS = struct.Struct('<4sdqqqqqqqddd')


def _stream_capture(connection, source, slots, stop, stream, mode, noise_options, options, progress_protocol='compact', completions=None, progress_group=1, gc_observation=False, prepare_gc=False):
    pipe = None
    published = 0
    pending_progress = []
    publications=0
    send_seconds=send_maximum=slot_seconds=slot_maximum=0.
    slow_commits=[]
    previous_post=0.
    previous_end=None
    gc_started=0.
    gc_times={str(k):dict(count=0,seconds=0.,maximum_seconds=0.) for k in range(3)}
    def observe_gc(phase,info):
        nonlocal gc_started
        if phase=='start': gc_started=time.perf_counter()
        else:
            value=gc_times[str(info['generation'])]
            elapsed=time.perf_counter()-gc_started
            value['count']+=1;value['seconds']+=elapsed
            value['maximum_seconds']=max(value['maximum_seconds'],elapsed)
    try:
        stream = StreamInfo(**stream)
        estimator = (GaussianNoiseEstimator if mode == 'fixed' else AdaptiveRobustNoiseEstimator)(**noise_options)
        pipe = StreamingCapturePipeline(stream,noise_estimator=estimator,enable_feature_analysis=False,**options)
        if prepare_gc:
            # This is an owned, fresh spawn interpreter. Finish collection of
            # import/configuration graphs BEFORE readiness; automatic collection
            # remains enabled throughout acquisition. No parent/global GC change.
            gc.collect()
        if gc_observation: gc.callbacks.append(observe_gc)
        connection.send(('ready',))
        while not stop.is_set():
            get_started=time.perf_counter() if gc_observation else None
            try:
                record = source.get(timeout=.05)
            except queue.Empty:
                continue
            cpu = time.thread_time()
            before = time.perf_counter()
            error = None
            try:
                if record is None:
                    events = pipe.flush()
                    arrival,seq,start,data,lost = 0.,-1,0,b'',0
                else:
                    arrival,seq,start,data,lost = record
                    block = IQBlock(stream,seq,start,None,np.frombuffer(data,np.complex64),dropped_samples_before=lost)
                    events = pipe.process_block(block)
            except Exception as exc:
                events = list(getattr(exc,'completed_captures',()))
                error = (type(exc).__name__, str(exc))
            finished = time.perf_counter()
            if gc_observation and record is not None and arrival>0:
                publication=getattr(source,'last_publication_monotonic',None)
                item=((finished-arrival)*1000,seq,
                    (publication-arrival)*1000 if publication is not None else None,
                    (before-publication)*1000 if publication is not None else None,
                    (before-get_started)*1000,(finished-before)*1000,
                    (get_started-previous_end)*1000 if previous_end is not None else None,
                    previous_post*1000)
                if len(slow_commits)<32: heapq.heappush(slow_commits,item)
                elif item>slow_commits[0]: heapq.heapreplace(slow_commits,item)
            snapshot = _snapshot(pipe)
            if gc_observation and (record is None or error):
                snapshot['capture_gc']={**gc_times,'prepared_before_ready':prepare_gc,
                                        'automatic_enabled':gc.isenabled()}
                snapshot['capture_transport'] = dict(publications=publications,
                    send_seconds=send_seconds,maximum_send_seconds=send_maximum,
                    slot_wait_seconds=slot_seconds,maximum_slot_wait_seconds=slot_maximum)
                snapshot['slow_commits'] = dict(columns=['commit_ms','sequence','source_publication_lateness_ms',
                    'publication_to_processing_ms','get_wait_ms','processing_ms','between_iterations_ms','previous_postprocessing_ms'],
                    rows=sorted(slow_commits,reverse=True))
            records = list(map(_encode_sidecar, events))
            processing_cpu = time.thread_time()-cpu
            coordinates = (seq, start, len(data)//8, snapshot['consumed_end_sample'],
                snapshot['received_end_sample'], snapshot['retained_noise_samples'], snapshot['retained_history_samples'])
            compact = (progress_protocol == 'compact' and seq >= 0 and not records and error is None
                and snapshot['state'] == 'OPEN' and all(type(v) is int and 0 <= v < 2**63 for v in coordinates))
            current = ('completed', arrival,seq,start,len(data)//8,
                records,snapshot,error,finished-before,processing_cpu,finished)
            if progress_group > 1:
                if seq >= 0:
                    consumed = len(data)//8 if error is None else min(len(data)//8,
                        max(0,snapshot['consumed_end_sample']-start))
                    pending_progress.append((arrival,seq,start,consumed,finished-before,processing_cpu,finished))
                # Input completion is distinct from parent delivery. No raw IQ
                # is retained in this finite <=4-entry telemetry group.
                source.task_done()
                if not records and error is None and record is not None and len(pending_progress)<progress_group:
                    if gc_observation:
                        previous_end=time.perf_counter();previous_post=previous_end-finished
                    continue
                message = pickle.dumps(('group',current,tuple(pending_progress)),protocol=5)
                pending_progress.clear()
                compact = False
            elif compact:
                message = _PROGRESS.pack(b'IQP1', arrival, seq, start, len(data)//8,
                    snapshot['consumed_end_sample'], snapshot['received_end_sample'],
                    snapshot['retained_noise_samples'], snapshot['retained_history_samples'],
                    finished-before, processing_cpu, finished)
            else:
                message = pickle.dumps(current, protocol=5)
            # A finite completion slot includes transport and parent delivery.
            publication_start=time.perf_counter() if gc_observation else None
            while not slots.acquire(timeout=.05):
                if stop.is_set():
                    return
            if publication_start is not None:
                wait=time.perf_counter()-publication_start
                slot_seconds+=wait;slot_maximum=max(slot_maximum,wait)
            # Processing has finished and a bounded completion is reserved.
            # A published EOF therefore also proves all input work was acked;
            # result delivery still has its separate parent-owned slot lease.
            if progress_group == 1:
                source.task_done()
            send_start=time.perf_counter() if gc_observation else None
            if completions is not None:
                header = message if compact else None
                shared_payload=(not compact and completions.payload_size and len(message)<=completions.payload_size)
                completions.publish(published, header,payload=message if shared_payload else None)
                published += 1
                if header is None and not shared_payload:
                    connection.send_bytes(message)
            else:
                connection.send_bytes(message)
            if send_start is not None:
                elapsed=time.perf_counter()-send_start
                send_seconds+=elapsed;send_maximum=max(send_maximum,elapsed)
                publications+=1
                previous_end=time.perf_counter();previous_post=previous_end-finished
            if record is None or error:
                break
    except (EOFError,BrokenPipeError,OSError):
        pass
    except Exception as exc:
        try:
            connection.send(('failed',type(exc).__name__,str(exc)))
        except (EOFError,BrokenPipeError,OSError):
            pass
    finally:
        if gc_observation and observe_gc in gc.callbacks: gc.callbacks.remove(observe_gc)
        if pipe is not None:
            pipe.shutdown()
        connection.close()


class CaptureStreamProcess:
    """Ready-before-acquisition child consuming one bounded shared source.

    EOF is the source's explicit None record. Each get() requires one task_done()
    after processing its raw captures. Do not read that source in the parent.
    A dead source without EOF must be detected/stopped by its owning session.

    progress_group=1 preserves per-block replies. An explicit 2..4 coalesces
    ONLY empty scalar telemetry: captures/faults/EOF publish immediately. Each
    returned progress tuple retains (arrival, sequence, start, committed_count,
    processing_seconds, cpu_seconds, finished) for every block, including an
    EOF tail. No input blocks/windows/IQ are merged. Requires an independently
    advancing producer, not a put-one/wait-for-one RPC loop. Grouped telemetry
    receipt delay is not detector completion latency. Capacity remains finite.

    completion_backend='shared_payload' additionally reserves finite full-message
    byte slots (default 256 KiB each). It avoids a small OS pipe imposing an
    earlier byte-capacity bottleneck for large captures, but is NOT universally
    faster; 'pipe' remains the compatibility/default path. Oversize records use
    the existing signalled pipe fallback, never disappear. Each read copies the
    bytes before constructing immutable events; task_done retains slot capacity.
    gc_observation adds bounded diagnostic clocks/counters; prepare_gc optionally
    collects import graphs before readiness WITHOUT disabling automatic GC.
    """
    def __init__(self, source, stream, *, noise_mode='fixed', noise_options=None,
                 pipeline_options=None, completion_capacity=16, progress_protocol='compact', completion_backend='pipe', progress_group=1, gc_observation=False, prepare_gc=False,max_completion_bytes=262144):
        if type(gc_observation) is not bool or type(prepare_gc) is not bool:
            raise ValueError('GC readiness/observation controls must be bool')
        if noise_mode not in ('fixed','adaptive') or type(completion_capacity) is not int or completion_capacity<1:
            raise ValueError('Invalid capture mode or completion capacity')
        if progress_protocol not in ('compact', 'pickle'):
            raise ValueError('Progress protocol must be compact or pickle')
        if completion_backend not in ('shared', 'pipe','shared_payload'):
            raise ValueError('Completion backend must be shared, shared_payload or pipe')
        if (type(max_completion_bytes) is not int or not 1024<=max_completion_bytes<=16*1024*1024
                or (completion_backend=='shared_payload' and completion_capacity*max_completion_bytes>256*1024*1024)):
            raise ValueError('Invalid bounded full-completion slot size (1 KiB..16 MiB, pool <=256 MiB)')
        if type(progress_group) is not int or not 1 <= progress_group <= 4:
            raise ValueError('Progress group must be an integer from 1 to 4')
        ctx = multiprocessing.get_context('spawn')
        self._connection,child = ctx.Pipe()
        self._slots=ctx.Semaphore(completion_capacity)
        self._completions = (CompletionSlots(ctx, completion_capacity, _PROGRESS.size,
                                            max_completion_bytes if completion_backend=='shared_payload' else 0)
                             if completion_backend in ('shared','shared_payload') else None)
        self._read_index = 0
        self._pending_pipe = False
        self._stop=ctx.Event()
        self._closed=False
        self._aborted=False
        self._failed=False
        self._outstanding=0
        self.done=False
        self._pid=None
        self._process=ctx.Process(target=_stream_capture,args=(child,source,self._slots,self._stop,
            dataclasses.asdict(stream),noise_mode,dict(noise_options or {}),dict(pipeline_options or {}),progress_protocol,self._completions,progress_group,gc_observation,prepare_gc),
            name='RF-capture-stream',daemon=True)
        try:
            self._process.start();self._pid=self._process.pid;child.close()
            if not self._connection.poll(20) or self._connection.recv()!=('ready',):
                raise RuntimeError('Capture stream startup failed')
        except BaseException:
            child.close();self.close();raise

    def get(self, timeout=.05):
        if self._closed:raise RuntimeError('Capture stream is closed')
        if self.done:
            raise queue.Empty
        deadline = None if timeout is None else time.monotonic()+max(0.,timeout)
        payload = None
        if self._completions is not None and not self._pending_pipe:
            if not self._completions.ready.acquire(timeout=timeout):
                if not self._process.is_alive():
                    self._failed=True
                    raise RuntimeError('Capture stream died without EOF acknowledgement')
                raise queue.Empty
            try:
                payload = self._completions.read(self._read_index)
            except ValueError as exc:
                self._failed=True
                raise RuntimeError('Invalid capture completion slot; not a successful EOF') from exc
            self._read_index += 1
            self._pending_pipe = payload is None
        remaining = None if deadline is None else max(0.,deadline-time.monotonic())
        if payload is None and not self._connection.poll(remaining):
            if not self._process.is_alive():
                self._failed=True
                raise RuntimeError('Capture stream died without EOF acknowledgement')
            raise queue.Empty
        try:
            if payload is None:
                payload = self._connection.recv_bytes()
                self._pending_pipe = False
            if payload.startswith(b'IQP1'):
                _,arrival,seq,start,n,consumed,received,noise,history,wall,cpu,finished = _PROGRESS.unpack(payload)
                message = ('completed',arrival,seq,start,n,[],dict(state='OPEN',consumed_end_sample=consumed,
                    received_end_sample=received,retained_noise_samples=noise,retained_history_samples=history),
                    None,wall,cpu,finished)
            else:
                message = pickle.loads(payload)
            progress = None
            if message[0] == 'group':
                _, message, progress = message
        except (EOFError,OSError,struct.error,pickle.UnpicklingError) as exc:
            self._failed=True
            raise RuntimeError('Capture stream lost; unacknowledged progress is unknown') from exc
        if message[0]!='completed':
            self._failed=True
            raise RuntimeError(f'Capture stream failure: {message}')
        _,arrival,seq,start,n,records,snapshot,error,wall,cpu,finished=message
        self._outstanding+=1
        self.done=seq==-1
        self._failed = self._failed or error is not None
        return dict(arrival=arrival,sequence=seq,start_sample=start,num_samples=n,
            captures=list(map(_decode_sidecar,records)),snapshot=snapshot,error=error,
            processing_seconds=wall,cpu_seconds=cpu,finished=finished,progress=progress)

    @property
    def pid(self):
        return self._pid

    @property
    def unfinished_completions(self):
        """Parent-owned completions awaiting acknowledgement (not input jobs)."""
        return self._outstanding

    @property
    def drained(self):
        """True only after explicit EOF and every parent completion acknowledged."""
        return self.done and not self._outstanding and not self._failed and not self._aborted

    def task_done(self):
        if self._outstanding<1:
            raise ValueError('No capture completion to acknowledge')
        self._outstanding-=1;self._slots.release()

    def close(self, timeout=5):
        """Release owned resources; early close is an explicit abort, not a drain.

        A caller must consume EOF and acknowledge its completions for drained to
        be true. On failure, unread child progress is unknown, not synthetic RF
        loss or successful terminal feature delivery. Source ownership is separate.
        """
        if self._closed:
            return
        self._stop.set()
        self._aborted = not self.drained
        if self._process.pid is not None:
            self._process.join(timeout)
            if self._process.is_alive():
                self._aborted=True
                self._process.terminate();self._process.join(timeout)
            if self._process.is_alive():
                raise TimeoutError('Capture stream process did not exit')
        self._connection.close();self._process.close();self._closed=True
        self._completions = None
