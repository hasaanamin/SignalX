"""Timer-only diagnostic, NOT a replacement for production acquisition gates."""
import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np


def measure(wait, seconds=5.):
    interval=.001024
    start=time.perf_counter();cpu=time.thread_time()
    late=[]
    for tick in range(1,int(seconds/interval)+1):
        arrival=start+tick*interval
        while True:
            remaining=arrival-time.perf_counter()
            if remaining<=0:break
            wait(remaining)
        late.append(time.perf_counter()-arrival)
    return dict(ticks=len(late),expired=sum(x>interval for x in late),
        lateness_ms_p50_p95_p99_max=(1000*np.percentile(late,[50,95,99,100])).tolist(),
        cpu_seconds=time.thread_time()-cpu,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    results=dict(scope='timer-only, no pipeline or features; diagnostic not acceptance',
        event_wait=measure(threading.Event().wait),high_resolution_sleep=measure(time.sleep))
    Path(args.output).write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
