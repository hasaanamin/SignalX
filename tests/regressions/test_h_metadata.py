"""Metadata admission is one atomic, exact-value public boundary."""
from dataclasses import replace, is_dataclass, fields
import enum
import numpy as np
import pytest
from signal_processing.core import IQBlock, StreamInfo, NoiseEstimate
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.detection import EnergyDetector, GaussianNoiseEstimator
from signal_processing.runtime import StreamingCapturePipeline


def frozen(value):
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value):
        return tuple((f.name, frozen(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, dict):
        return tuple((k, frozen(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(map(frozen, value))
    if hasattr(value, '__dict__'):
        return frozen(vars(value))
    return repr(value)


def snapshot(p):
    return frozen({k: v for k, v in vars(p).items() if k != '_worker'})


def make(s, mode='fixed'):
    est = GaussianNoiseEstimator(window_size=16, **(
        {'warmup_samples': 48} if mode == 'warmup' else {'calibrated_noise_power_linear': 1e-4}))
    return StreamingCapturePipeline(s, ring_capacity=512, window_size=16, hop_size=8,
        noise_estimator=est, start_persistence=1, end_persistence=1,
        pre_trigger_samples=16, pre_signal_margin_samples=0, post_signal_samples=0,
        min_post_trigger_samples=0, enable_feature_analysis=False,
        detector_mode='configured_band' if mode == 'filtered' else 'full_band',
        monitored_band_hz=(-100000.,100000.) if mode == 'filtered' else None)


@pytest.mark.parametrize('change', [dict(sample_rate_hz=1000000.0005),
    dict(sample_rate_hz=1000000.0000005), dict(center_frequency_hz=100000000.05),
    dict(center_frequency_hz=None), dict(stream_id='other'), dict(source_id='other'), dict(channel_id='other')])
@pytest.mark.parametrize('mode,origin', [('fixed',0), ('active',1), ('filtered',1000), ('warmup',10**12)])
def test_rejection_preserves_all_streaming_state_and_exact_retry(change,mode,origin):
    s=StreamInfo('h','actual-iq',1e6,100e6)
    p=make(s,mode); control=make(s,mode)
    x=np.full(160,.003,np.complex64); x[16:96]=.5
    if mode=='warmup': x[:48]=.003
    first=IQBlock(s,0,origin,1000000000,x[:29])
    p.process_block(first); control.process_block(first)
    before=snapshot(p)
    bad=IQBlock(replace(s,**change),1,origin+29,1000029000,x[29:])
    with pytest.raises(StreamConsistencyError): p.process_block(bad)
    assert snapshot(p)==before
    valid=replace(bad,stream=replace(s))
    actual=p.process_block(valid)+p.flush()
    expected=control.process_block(valid)+control.flush()
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected):
        assert (a.event.event_start_sample,a.event.event_end_sample)==(b.event.event_start_sample,b.event.event_end_sample)
        assert a.event.iq.tobytes()==b.event.iq.tobytes()
    p.shutdown(); control.shutdown()


def test_first_center_binding_is_not_committed_by_rejected_input():
    s=StreamInfo('h','source',1e6)
    p=make(s,'warmup'); p.noise_estimator.freeze(True)
    block=IQBlock(replace(s,center_frequency_hz=100e6),0,10**12,0,np.full(64,.003,np.complex64))
    before=snapshot(p)
    with pytest.raises(StreamConsistencyError): p.process_block(block)
    assert snapshot(p)==before and p._stream_info.center_frequency_hz is None
    p.noise_estimator.freeze(False); p.process_block(block)
    assert p._stream_info==block.stream
    p.flush();p.shutdown()


def test_direct_tiny_rate_change_rejected_before_filter_state():
    s=StreamInfo('h','source',1e6)
    d=EnergyDetector(window_size=16,hop_size=8,mode='configured_band',monitored_band_hz=(-100000.,100000.))
    b=IQBlock(replace(s,sample_rate_hz=1000000.0000005),0,1000,None,np.ones(8,np.complex64))
    before=frozen(vars(d))
    with pytest.raises(Exception) as exc: d.process_windows(b,NoiseEstimate('h',0,-40))
    from signal_processing.detection.exceptions import PolicyCompatibilityError
    assert isinstance(exc.value,PolicyCompatibilityError)
    assert frozen(vars(d))==before
    assert d.process_windows(replace(b,stream=s),NoiseEstimate('h',0,-40))==[]
