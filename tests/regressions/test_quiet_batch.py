"""Compare exact captures/state against chronological reference, never saved JSON."""
import dataclasses
import numpy as np
import pytest
from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import GaussianNoiseEstimator
from signal_processing.runtime import StreamingCapturePipeline


@pytest.mark.parametrize('origin', [0, 1, 1000, 10**12])
@pytest.mark.parametrize('n,h', [(16,8), (17,8), (64,32)])
@pytest.mark.parametrize('partition', [1, 1024, 7777])
def test_quiet_batch_exact_reference(origin, n, h, partition):
    stream = StreamInfo('batch', 'test', 1e6)
    rng = np.random.default_rng(32)
    x = ((rng.normal(size=9000)+1j*rng.normal(size=9000))*1e-4).astype(np.complex64)
    x[3000:3264] += .1
    x[7000:7140] += .3
    outputs = []
    for fast in (False, True):
        p = StreamingCapturePipeline(stream, window_size=n, hop_size=h, ring_capacity=4096,
            pre_trigger_samples=128, pre_signal_margin_samples=32, post_signal_samples=23,
            max_event_samples=1024, fixed_quiet_batch=fast, enable_feature_analysis=False,
            noise_estimator=GaussianNoiseEstimator(window_size=n,calibrated_noise_power_linear=1e-4))
        events=[]
        for seq, a in enumerate(range(0,len(x),partition)):
            events += p.process_block(IQBlock(stream,seq,origin+a,None,x[a:a+partition]))
        events += p.flush()
        summaries=[]
        for sc in events:
            event=sc.event
            fields={f.name:getattr(event,f.name) for f in dataclasses.fields(event) if f.name not in ('iq','event_id')}
            summaries.append((fields,event.iq.tobytes()))
            np.testing.assert_array_equal(event.iq,x[event.event_start_sample-origin:event.event_end_sample-origin])
        outputs.append((summaries,p.ring_buffer.available_range(),p.detector._residual_start_sample,
                        p.detector._next_window_sample,p._noise_admission_sample))
        p.shutdown()
    assert outputs[0] == outputs[1] and len(outputs[0][0]) == 2


def test_pure_quiet_preview_does_not_commit_future_extreme_samples():
    from signal_processing.detection import EnergyDetector
    from signal_processing.core import NoiseEstimate
    stream = StreamInfo('extreme','test',1e6)
    d = EnergyDetector(window_size=16,hop_size=8)
    x=np.full(64,1.2e-8,np.complex64);x[0]=1
    block=IQBlock(stream,0,10**12,None,x)
    assert not d._is_quiet_full_band(block,NoiseEstimate(stream.stream_id,0,-160))
    assert d._input_end is None and len(d._residual_iq)==0
    results=d.process_windows(block,NoiseEstimate(stream.stream_id,0,-160))
    assert all(not m.above_on_threshold for m in results[1:])
