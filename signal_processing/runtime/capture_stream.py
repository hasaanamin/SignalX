"""Bounded source-to-capture process boundary, with no per-block parent RPC.

The supplied SPSC SharedIQTransport has exactly one consumer: this child.
Input blocks retain their original size and coordinates. Completion records are
bounded by a semaphore and acknowledged by the parent, never by queue.empty().
Feature execution and arbitrary callbacks belong to the parent application.
"""
import dataclasses
import multiprocessing
import queue
import time

import numpy as np

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import GaussianNoiseEstimator, AdaptiveRobustNoiseEstimator
from .pipeline import StreamingCapturePipeline
from .isolated_capture import _encode_sidecar, _decode_sidecar, _snapshot


def _stream_capture(connection, source, slots, stop, stream, mode, noise_options, options):
    pipe = None
    try:
        stream = StreamInfo(**stream)
        estimator = (GaussianNoiseEstimator if mode == 'fixed' else AdaptiveRobustNoiseEstimator)(**noise_options)
        pipe = StreamingCapturePipeline(stream,noise_estimator=estimator,enable_feature_analysis=False,**options)
        connection.send(('ready',))
        while not stop.is_set():
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
            message = ('completed', arrival,seq,start,len(data)//8,
                       list(map(_encode_sidecar,events)), _snapshot(pipe),error,
                       finished-before,time.thread_time()-cpu,finished)
            # A finite completion slot includes transport and parent delivery.
            while not slots.acquire(timeout=.05):
                if stop.is_set():
                    return
            connection.send(message)
            source.task_done()
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
        if pipe is not None:
            pipe.shutdown()
        connection.close()


class CaptureStreamProcess:
    """Ready-before-acquisition child consuming one bounded shared source.

    EOF is the source's explicit None record. Each get() requires one task_done()
    after processing its raw captures. Do not read that source in the parent.
    A dead source without EOF must be detected/stopped by its owning session.
    """
    def __init__(self, source, stream, *, noise_mode='fixed', noise_options=None,
                 pipeline_options=None, completion_capacity=16):
        if noise_mode not in ('fixed','adaptive') or type(completion_capacity) is not int or completion_capacity<1:
            raise ValueError('Invalid capture mode or completion capacity')
        ctx = multiprocessing.get_context('spawn')
        self._connection,child = ctx.Pipe()
        self._slots=ctx.Semaphore(completion_capacity)
        self._stop=ctx.Event()
        self._closed=False
        self._aborted=False
        self._failed=False
        self._outstanding=0
        self.done=False
        self._pid=None
        self._process=ctx.Process(target=_stream_capture,args=(child,source,self._slots,self._stop,
            dataclasses.asdict(stream),noise_mode,dict(noise_options or {}),dict(pipeline_options or {})),
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
        if not self._connection.poll(timeout):
            if not self._process.is_alive():
                self._failed=True
                raise RuntimeError('Capture stream died without EOF acknowledgement')
            raise queue.Empty
        try:message=self._connection.recv()
        except (EOFError,OSError) as exc:
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
            processing_seconds=wall,cpu_seconds=cpu,finished=finished)

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
