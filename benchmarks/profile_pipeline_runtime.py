"""Attribution only: deterministic real IQ, no scheduled-throughput claims."""
import argparse
import cProfile
import dataclasses
import io
from pathlib import Path
import pstats

import numpy as np

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
from benchmarks.sustained_evaluation import _pipeline


def generic():
    stream = StreamInfo('profile', 'synthetic', 2e6)
    pipe = _pipeline(stream, fixed_quiet_batch=True)
    rng = np.random.default_rng(730)
    for seq in range(32):
        x = ((rng.normal(size=16384)+1j*rng.normal(size=16384))*.0070710678).astype(np.complex64)
        for start in range(0, len(x), 4096):
            x[start:start+256] += .1
        pipe.process_block(IQBlock(stream, seq, seq*16384, None, x))
    pipe.flush()
    pipe.shutdown()


def lte(path):
    stream = StreamInfo('profile', 'recording', 2e6, 2.14e9)
    receiver = LTEBroadcastReceiver(stream, LTESyncConfig.low_snr())
    with path.open('rb') as source:
        for seq in range(32):
            x = np.fromfile(source, '<c8', 16384)
            receiver.process_block(IQBlock(stream, seq, seq*16384, None, x))
    receiver.flush()


def features(output):
    from signal_processing.runtime import FeatureAnalysisWorker
    from tests.regressions.test_process_analysis import capture
    example = capture()
    original = FeatureAnalysisWorker._worker_loop
    def profiled(worker):
        profiler = cProfile.Profile()
        try:
            profiler.runcall(original, worker)
        finally:
            profiler.dump_stats(str(output.with_suffix('.worker.prof')))
    FeatureAnalysisWorker._worker_loop = profiled
    worker = None
    try:
        worker = FeatureAnalysisWorker(synchronous=False, analysis_backend='process_pool',
            result_delivery='callbacks', max_queue_size=4)
        completed = []  # Only 200 status strings, no IQ/result history.
        worker.add_result_sink(lambda sc: completed.append(sc.feature_status))
        for k in range(200):
            assert worker.wait_for_capacity(timeout=5)
            event = dataclasses.replace(example.event, event_id=f'profile-{k}')
            result = worker.submit(dataclasses.replace(example, event=event))
            assert result.feature_status == 'pending'
        worker.shutdown(timeout=10)
        assert len(completed) == 200 and set(completed) == {'ok'}
    finally:
        if worker is not None:
            worker.shutdown(timeout=10)
        FeatureAnalysisWorker._worker_loop = original
    report = io.StringIO()
    pstats.Stats(str(output.with_suffix('.worker.prof')), stream=report).strip_dirs().sort_stats('cumulative').print_stats(40)
    print(report.getvalue())


def feature_math():
    from signal_processing.runtime import FeatureAnalysisWorker
    stream = StreamInfo('feature-profile', 'synthetic', 2e6)
    pipe = _pipeline(stream, fixed_quiet_batch=True)
    rng = np.random.default_rng(730)
    x = ((rng.normal(size=16384)+1j*rng.normal(size=16384))*.0070710678).astype(np.complex64)
    for a in range(0, len(x), 4096):
        x[a:a+256] += .1
    captures = pipe.process_block(IQBlock(stream, 0, 0, None, x))+pipe.flush()
    pipe.shutdown()
    worker = FeatureAnalysisWorker()
    try:
        for k in range(200):
            example = captures[k % len(captures)]
            event = dataclasses.replace(example.event, event_id=f'math-{k}')
            assert worker.submit(dataclasses.replace(example, event=event)).feature_status == 'ok'
            worker.get_result(block=False)
    finally:
        worker.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('generic', 'lte', 'features', 'feature_math'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    profiler = cProfile.Profile()
    if args.mode == 'features':
        # Python 3.14 cProfile reserves one interpreter monitoring tool ID;
        # enabling a second concurrent Profile in the parent would kill the
        # actual worker. Profile only its execution thread for this mode.
        features(args.output)
        return
    else:
        task = generic if args.mode == 'generic' else feature_math if args.mode == 'feature_math' else lambda: lte(args.input)
        profiler.runcall(task)
    profiler.dump_stats(str(args.output))
    report = io.StringIO()
    pstats.Stats(profiler, stream=report).strip_dirs().sort_stats('cumulative').print_stats(32)
    print(report.getvalue())


if __name__ == '__main__':
    main()
