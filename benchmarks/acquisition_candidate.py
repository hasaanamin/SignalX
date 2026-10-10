"""Controlled backend comparison using the unchanged sustained workload and gate."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def main():
    from benchmarks.sustained_evaluation import bounded_acquisition
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['thread', 'process', 'process_pool', 'isolated'], default='process')
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--producer', choices=['thread', 'process'], default='thread')
    parser.add_argument('--quiet-batch', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Diagnostic only; perturbs timing')
    parser.add_argument('--gil-interval', type=float, help='Explicit interpreter-wide switch interval experiment; restored after run')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Do not overwrite existing observations')
    if args.profile:
        import cProfile
        import pstats
        import io
        profiler = cProfile.Profile()
        profiler.enable()
    previous_interval = sys.getswitchinterval()
    try:
        if args.gil_interval is not None:
            sys.setswitchinterval(args.gil_interval)
        result = bounded_acquisition(args.seconds, features=True, analysis_backend=args.backend,
                                     producer_backend=args.producer, fixed_quiet_batch=args.quiet_batch)
        result['gil_switch_interval_seconds'] = sys.getswitchinterval()
    finally:
        sys.setswitchinterval(previous_interval)
    if args.profile:
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(65)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix('.profile.txt').write_text(stream.getvalue(), encoding='utf8')
        result['profiled_not_acceptance'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v, list)}, indent=2))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
