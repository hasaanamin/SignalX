"""Independent linear-convolution oracle for received-only overlap-save IQ."""
import dataclasses
import math

import numpy as np
import pytest

from signal_processing.core import StreamInfo, IQBlock
from signal_processing.lte import LTESyncConfig, LTESynchronizer


@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_native_frequency_reductions_agree_with_independent_stable_sums(scale):
    from signal_processing.lte._native import initialize_sync,_frequency_dots
    initialize_sync('numba')
    rng=np.random.default_rng(20261020)
    samples=((rng.normal(size=(4,128))+1j*rng.normal(size=(4,128)))*scale).astype(np.complex64).astype(np.complex128)
    templates=np.exp(2j*np.pi*np.arange(31)[:,None]*np.arange(128)[None,:]/257)
    templates.flags.writeable=False
    actual=_frequency_dots(samples,templates)
    expected=np.empty_like(actual)
    for repeat in range(4):
        for frequency in range(31):
            values=[samples[repeat,j]*templates[frequency,j] for j in range(128)]
            expected[repeat,frequency]=complex(math.fsum(v.real for v in values),math.fsum(v.imag for v in values))
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=scale*1e-13)


@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
@pytest.mark.parametrize('repeats',[1,4])
def test_streaming_blind_power_matches_direct_window_correlations(scale,repeats):
    stream=StreamInfo('oracle','noise',1_920_000.)
    config=dataclasses.replace(LTESyncConfig.low_snr(1_920_000.),integration_half_frames=repeats)
    receiver=LTESynchronizer(stream,config,sync_backend='numba')
    rng=np.random.default_rng(2233)
    size=137+(repeats-1)*9600+2500+127
    samples=((rng.normal(size=size)+1j*rng.normal(size=size))*scale).astype(np.complex64).astype(np.complex128)
    receiver._history.append(samples)
    receiver._buffer=receiver._history.view()
    for horizon in range(4096,size,4096): receiver._prepare_blind_powers(horizon)
    receiver._prepare_blind_powers(size)
    start=137+(repeats-1)*9600
    stop=start+2500
    actual=receiver._cached_blind_scores(start,stop)
    assert actual is not None
    for position in (start,start+384,start+385,start+2000,stop-1):
        energy=0.
        powers=np.zeros(len(receiver._templates))
        for repeat in range(repeats):
            window=samples[position-repeat*9600:position-repeat*9600+128]
            energy+=float(np.vdot(window,window).real)
            # Independent direct inner products, not another FFT algorithm.
            powers+=np.abs(np.array([np.vdot(t,window) for t in receiver._templates]))**2
        np.testing.assert_allclose(actual[:,position-start],powers/(62*energy),rtol=1e-12,atol=1e-15)


def test_search_cache_is_bounded_and_reset_does_not_reuse_old_iq():
    stream=StreamInfo('bounded','quiet',1_920_000.)
    receiver=LTESynchronizer(stream,LTESyncConfig.low_snr(1_920_000.),sync_backend='numba')
    iq=np.zeros(16384,np.complex64)
    counts=[]
    for seq in range(45):
        assert not receiver.process_block(IQBlock(stream,seq,10**12+seq*len(iq),None,iq))
        retained=sum((b-a) for a,b,_,_ in receiver._score_chunks)
        counts.append(retained)
        assert retained<=300+3*9600+2*receiver.config.search_chunk_samples
    assert max(counts[-10:])<=max(counts[10:20])
    receiver.reset()
    assert not receiver._score_chunks
    assert receiver._score_end==137
    receiver.process_block(IQBlock(stream,0,1,None,iq))
    assert receiver.snapshot()['received_input_end']==16385
