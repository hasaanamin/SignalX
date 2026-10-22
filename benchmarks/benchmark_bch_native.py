"""Kernel timing, separated from JIT readiness and scheduled acquisition."""
import argparse
import json
import time

import numpy as np

from signal_processing.lte.bch import decode_tail_biting
from signal_processing.lte._native import diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=500)
    args = parser.parse_args()
    rng = np.random.default_rng(3199)
    inputs = rng.normal(size=(32, 120))
    rows = []
    for backend, threads in (('reference', 1), ('numba', 1), ('numba', 2), ('numba', 4)):
        started = time.perf_counter()
        info = diagnostics(backend, threads)
        decode_tail_biting(inputs[0], backend=backend, threads=threads)
        readiness = time.perf_counter()-started
        elapsed = []
        for k in range(args.iterations):
            started = time.perf_counter()
            decode_tail_biting(inputs[k % len(inputs)], backend=backend, threads=threads)
            elapsed.append((time.perf_counter()-started)*1e6)
        rows.append(dict(**info, readiness_seconds=readiness,
                         microseconds_p50_p99_max=np.percentile(elapsed, [50, 99, 100]).tolist()))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
