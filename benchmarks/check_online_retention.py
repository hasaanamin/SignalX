"""Traced ownership check, NOT an unperturbed timing benchmark."""
import gc
import json
import tracemalloc
import numpy as np
from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import OnlineNoiseConfig
from signal_processing.runtime.pipeline import StreamingCapturePipeline


def main():
    s = StreamInfo("retention", "synthetic", 1e6)
    p = StreamingCapturePipeline(s, online_noise=OnlineNoiseConfig(), ring_capacity=4096,
                                enable_feature_analysis=False)
    rng = np.random.default_rng(994)
    observations = []
    tracemalloc.start()
    try:
        for seq in range(2048):
            iq = ((rng.normal(size=1024)+1j*rng.normal(size=1024))*np.sqrt(1e-4/2)).astype(np.complex64)
            p.process_block(IQBlock(s, seq, seq*1024, None, iq))
            if seq in (511, 1023, 2047):
                gc.collect()
                current, peak = tracemalloc.get_traced_memory()
                observations.append(dict(samples=(seq+1)*1024, current_bytes=current, peak_bytes=peak,
                    noise=p.noise_estimator.diagnostics(), pipeline=p.retention_snapshot(),
                    detector_residual=len(p.detector._residual_iq), unfinished=p.worker.unfinished_work))
        p.flush()
    finally:
        p.shutdown()
        tracemalloc.stop()
    print(json.dumps(observations, indent=2))


if __name__ == "__main__":
    main()
