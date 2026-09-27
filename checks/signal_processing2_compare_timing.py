"""Comparable ingestion timing for old and new source trees."""
from pathlib import Path
import sys,json,time,cProfile,pstats,io
import numpy as np
src=Path(sys.argv[1]).resolve();sys.path.insert(0,str(src))
from signal_processing.core import StreamInfo,IQBlock
from signal_processing.detection import GaussianNoiseEstimator
from signal_processing.runtime import StreamingCapturePipeline
rng=np.random.default_rng(42);x=((rng.normal(size=200000)+1j*rng.normal(size=200000))*.01).astype(np.complex64)
stream=StreamInfo('perf','noise',1e6)
blocks=[IQBlock(stream,j,k,None,x[k:k+1024]) for j,k in enumerate(range(0,len(x),1024))]
def run():
    p=StreamingCapturePipeline(stream_info=stream,ring_capacity=16384,window_size=256,hop_size=64,
        noise_estimator=GaussianNoiseEstimator(window_size=256,calibrated_noise_power_linear=2e-4))
    t0=time.perf_counter();events=[]
    for b in blocks:events.extend(p.process_block(b))
    events+=p.flush();p.shutdown()
    elapsed=time.perf_counter()-t0
    return {'elapsed_s':elapsed,'throughput_MSps':len(x)/elapsed/1e6,'events':len(events)}
rows=[run() for _ in range(3)]
out={'source':str(src),'samples':len(x),'transport_blocks':len(blocks),'measurements':(len(x)-256)//64+1,
     'repeats':rows,'median_MSps':float(np.median([r['throughput_MSps'] for r in rows]))}
if len(sys.argv)>2:
    prof=cProfile.Profile();prof.enable();run();prof.disable();s=io.StringIO()
    pstats.Stats(prof,stream=s).sort_stats('cumulative').print_stats(12);out['profile']=s.getvalue()
print(json.dumps(out,indent=2))

