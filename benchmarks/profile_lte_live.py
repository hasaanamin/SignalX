"""Live stall attribution; instrumentation is not timing qualification."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TracedSourceFactory:
    delegate: object
    trace: str

    def __call__(self):
        source = self.delegate()
        original_read,original_close = source.read_envelope,source.close
        origin = None
        starts,collections,slow_reads = {},[],[]
        waits,reads = [],[]
        original_wait = source._clock.wait_until_ns
        original_block = source._source.read_block
        def wait(deadline,cancellation):
            before = time.perf_counter_ns()
            try:
                return original_wait(deadline,cancellation)
            finally:
                end = time.perf_counter_ns()
                if (len(waits) < 4 or end-deadline > 2_000_000) and len(waits) < 1024:
                    waits.append(((before-(origin or before))/1e9,(deadline-before)/1e6,(end-deadline)/1e6))
        def block():
            before = time.perf_counter_ns()
            value = original_block()
            end = time.perf_counter_ns()
            if (len(reads) < 4 or end-before > 2_000_000) and len(reads) < 1024:
                reads.append(((before-(origin or before))/1e9,(end-before)/1e6,
                              value.sequence_number if value is not None else None))
            return value
        source._clock.wait_until_ns,source._source.read_block = wait,block
        processors = {}
        import ctypes
        processor = ctypes.windll.kernel32.GetCurrentProcessorNumber if sys.platform == 'win32' else lambda: -1
        def collection(phase,info):
            if origin is None:
                return
            generation = info['generation']
            if phase == 'start':
                starts[generation] = time.perf_counter_ns()
            elif len(collections) < 1024:
                end = time.perf_counter_ns()
                start = starts.pop(generation,end)
                collections.append(((start-origin)/1e9,(end-start)/1e6,generation,info['collected']))
        def read(timeout=None):
            nonlocal origin
            start = time.perf_counter_ns()
            if origin is None:
                origin = start
            try:
                return original_read(timeout=timeout)
            finally:
                cpu = processor()
                processors[cpu] = processors.get(cpu,0)+1
                elapsed = (time.perf_counter_ns()-start)/1e6
                if elapsed > 12 and len(slow_reads) < 1024:
                    slow_reads.append(((start-origin)/1e9,elapsed))
        def close():
            try:
                original_close()
            finally:
                gc.callbacks.remove(collection)
                Path(self.trace).write_text(json.dumps(dict(origin_perf_ns=origin,
                    gc_collections=collections,slow_reads=slow_reads,processors=processors,
                    waits_start_remaining_overshoot=waits,block_reads_start_ms_sequence=reads),indent=2)+'\n',encoding='utf8')
        gc.callbacks.append(collection)
        source.read_envelope,source.close = read,close
        return source


def main():
    import numpy as np
    from signal_processing.lte.broadcast import LTEBroadcastReceiver
    from signal_processing.lte.cli import main as cli
    from signal_processing.runtime import ProcessSourceRunner
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--trace-blocks',type=int,default=20000)
    parser.add_argument('--source-trace',type=Path)
    parser.add_argument('--no-dsp-mmcss',action='store_true')
    parser.add_argument('--trace-source-cpu',type=int)
    parser.add_argument('--trace-spin-wait',action='store_true')
    args,remaining = parser.parse_known_args()
    if args.trace.exists():
        raise FileExistsError(args.trace)
    trace = np.empty((args.trace_blocks,14))
    count,origin = 0,None
    collections,writes,starts = [],[],{}
    original_begin = ProcessSourceRunner.begin
    original_runner_init = ProcessSourceRunner.__init__
    original_get = ProcessSourceRunner.get
    if args.trace_spin_wait:
        from signal_processing.sources import SourceReadTimeout
        def spin_get(self,timeout=None,**options):
            deadline = None if timeout is None else time.monotonic()+timeout
            while True:
                try:
                    return original_get(self,timeout=0,**options)
                except SourceReadTimeout:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise
        ProcessSourceRunner.get = spin_get
    if args.trace_source_cpu is not None:
        def runner_init(self,*a,**kw):
            kw['acquisition_cpus'] = [args.trace_source_cpu]
            original_runner_init(self,*a,**kw)
        ProcessSourceRunner.__init__ = runner_init
    original_process = LTEBroadcastReceiver.process_envelope
    original_stdout = sys.stdout
    original_init = LTEBroadcastReceiver.__init__
    original_prepare = LTEBroadcastReceiver._prepare_blind_powers
    import signal_processing.lte.cli as cli_module
    original_factory = cli_module.ReplayFileFactory
    import signal_processing.runtime._performance as scheduling_module
    original_realtime_scope = scheduling_module.realtime_scope
    if args.no_dsp_mmcss:
        scheduling_module.realtime_scope = lambda enabled=True,**kwargs: original_realtime_scope(False,**kwargs)
    if args.source_trace is not None:
        if args.source_trace.exists():
            raise FileExistsError(args.source_trace)
        cli_module.ReplayFileFactory = lambda *a,**kw: TracedSourceFactory(original_factory(*a,**kw),str(args.source_trace.resolve()))
    prepared_seconds = 0.
    def initialize(self,*args,**kwargs):
        kwargs['measure_timing'] = True
        original_init(self,*args,**kwargs)
    def prepare(self,*args,**kwargs):
        nonlocal prepared_seconds
        started = time.perf_counter()
        try:
            return original_prepare(self,*args,**kwargs)
        finally:
            prepared_seconds += time.perf_counter()-started
    import ctypes
    processor = ctypes.windll.kernel32.GetCurrentProcessorNumber if sys.platform == 'win32' else lambda: -1
    if sys.platform == 'win32':
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentThread.restype = ctypes.c_void_p
        kernel.GetThreadPriority.argtypes = [ctypes.c_void_p]
        current_thread = kernel.GetCurrentThread()
        priority = lambda: kernel.GetThreadPriority(current_thread)
    else:
        priority = lambda: -999

    def begin(self):
        nonlocal origin
        origin = time.perf_counter_ns()
        return original_begin(self)

    def process(self,envelope):
        nonlocal count
        start = time.perf_counter_ns()
        cpu = time.process_time_ns()
        stages = self.stage_timing
        sync_before = stages['synchronization']['seconds']
        bch_before = stages['pbch_decode']['seconds']
        bank_before = prepared_seconds
        try:
            return original_process(self,envelope)
        finally:
            end = time.perf_counter_ns()
            if count < len(trace):
                trace[count] = (envelope.block.sequence_number,envelope.block.start_sample,
                    (start-envelope.scheduled_monotonic_ns)/1e6,(end-start)/1e6,
                    (time.process_time_ns()-cpu)/1e6,
                    (envelope.received_monotonic_ns-envelope.scheduled_monotonic_ns)/1e6,
                    (self.stage_timing['synchronization']['seconds']-sync_before)*1000,
                    (self.stage_timing['pbch_decode']['seconds']-bch_before)*1000,
                    (prepared_seconds-bank_before)*1000,int(self._subspace_cutoffs is None),
                    self._dense_search_fallbacks,processor(),int(self._tracking_missed),priority())
                count += 1

    def collection(phase,info):
        generation = info['generation']
        if phase == 'start':
            starts[generation] = time.perf_counter_ns()
        elif origin is not None:
            end = time.perf_counter_ns()
            start = starts.pop(generation,end)
            collections.append(((start-origin)/1e9,(end-start)/1e6,generation,info['collected']))

    class Output:
        def __getattr__(self,name):
            return getattr(original_stdout,name)

        def write(self,value):
            start = time.perf_counter_ns()
            result = original_stdout.write(value)
            end = time.perf_counter_ns()
            if origin is not None and end-start > 1_000_000:
                writes.append(((start-origin)/1e9,(end-start)/1e6,len(value)))
            return result

    ProcessSourceRunner.begin = begin
    LTEBroadcastReceiver.__init__ = initialize
    LTEBroadcastReceiver._prepare_blind_powers = prepare
    LTEBroadcastReceiver.process_envelope = process
    gc.callbacks.append(collection)
    sys.stdout = Output()
    try:
        return cli(remaining)
    finally:
        sys.stdout = original_stdout
        gc.callbacks.remove(collection)
        ProcessSourceRunner.begin = original_begin
        ProcessSourceRunner.__init__ = original_runner_init
        ProcessSourceRunner.get = original_get
        LTEBroadcastReceiver.__init__ = original_init
        LTEBroadcastReceiver._prepare_blind_powers = original_prepare
        cli_module.ReplayFileFactory = original_factory
        scheduling_module.realtime_scope = original_realtime_scope
        LTEBroadcastReceiver.process_envelope = original_process
        args.trace.write_text(json.dumps(dict(
            columns=('sequence','start_sample','queue_ms','processing_wall_ms','processing_cpu_ms','source_lateness_ms',
                     'sync_ms','bch_ms','bank_ms','full_bank','dense_fallbacks','processor','tracking_missed','relative_priority'),
            blocks=trace[:count].tolist(),gc_collections=collections,slow_stdout_writes=writes),indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
