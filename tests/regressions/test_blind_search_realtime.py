"""Direct received-window oracles for continuous native blind acquisition."""
import dataclasses

import numpy as np
import pytest

from signal_processing.core import StreamInfo
from signal_processing.lte import LTESynchronizer, LTESyncConfig


def _finalize_native_cycle_while_planning(send):
    try:
        import gc
        import weakref
        from signal_processing.lte._fftw_native import optional_native_bank
        from signal_processing.lte._fft_bank import _planner_lock
        bank = optional_native_bank(np.ones((1,128),np.complex128))
        if bank is None:
            send.send((True,'portable backend'))
            return
        reference,finalizers = weakref.ref(bank),bank._finalizers
        cycle = [bank]
        cycle.append(cycle)
        del bank,cycle
        with _planner_lock:
            gc.collect()
        send.send((reference() is None and not any(f.alive for f in finalizers),'native finalized'))
    except BaseException as exc:
        send.send((False,repr(exc)))
    finally:
        send.close()


def test_native_plan_gc_during_planner_argument_creation_does_not_deadlock():
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    receive,send = context.Pipe(duplex=False)
    process = context.Process(target=_finalize_native_cycle_while_planning,args=(send,))
    process.start()
    send.close()
    try:
        assert receive.poll(30),'Native plan finalization deadlocked under the planner lock'
        passed,detail = receive.recv()
        assert passed,detail
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        process.close()
        receive.close()


def test_native_preparation_copies_all_received_iq_directly_into_owned_double_history(monkeypatch):
    from signal_processing.core import IQBlock
    rng = np.random.default_rng(56783)
    iq = (rng.normal(size=16384)+1j*rng.normal(size=16384)).astype(np.complex64)
    expected = iq.astype(np.complex128)
    stream = StreamInfo('native-copy','received',1_920_000.)
    receiver = LTESynchronizer(stream,LTESyncConfig.native(low_snr=True))
    def redundant_conversion(*args):
        raise AssertionError('Native history must not allocate an intermediate double transport array')
    monkeypatch.setattr(receiver._channel,'process',redundant_conversion)
    block = IQBlock(stream,0,10**12,None,iq)
    receiver.process_block(block)
    iq.fill(99)
    np.testing.assert_array_equal(receiver._buffer,expected)
    assert receiver._buffer.dtype == np.complex128
    assert not np.shares_memory(receiver._buffer,block.iq)
    assert receiver._channel.input_count == receiver._channel.output_count == len(iq)


def test_native_fft_plans_are_owned_and_destroyed_independently(monkeypatch):
    import gc
    import weakref
    from signal_processing.lte import _fftw_native as native
    library = native._library()
    if library is None:
        assert native.optional_native_bank(np.ones((1,128),np.complex128)) is None
        return
    destroyed = []
    original = native._destroy
    def destroy(owner,lock,plan):
        original(owner,lock,plan)
        destroyed.append(plan)
    monkeypatch.setattr(native,'_destroy',destroy)
    kernels = np.ones((3,128),np.complex128)
    first,second = (native.NativeFFTBank(kernels,library) for _ in range(2))
    first_handles = {int(first._forward),int(first._inverse)}
    second_handles = {int(second._forward),int(second._inverse)}
    assert not first_handles & second_handles
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None and set(destroyed) == first_handles
    iq = np.frombuffer(np.arange(1151,dtype=np.complex128).tobytes(),np.complex128)
    powers = np.empty((3,2048))
    second.write_powers(iq,0,powers,1024,0)
    expected = np.abs(np.convolve(iq,kernels[0],'valid'))**2
    np.testing.assert_allclose(powers[0,:1024],expected,rtol=2e-12)
    del second
    gc.collect()
    assert len(destroyed) == 4 and set(destroyed) == first_handles | second_handles


def test_sparse_selection_clears_prior_scores_across_query_sizes():
    from signal_processing.lte._native import _subspace_select,subspace_workspace
    rng = np.random.default_rng(8283)
    workspace = subspace_workspace(128,27)
    energy,projections,maxima,result,branches,positions,previous = workspace[:7]
    templates = np.ones((27,128),np.complex128)
    cutoffs = rng.uniform(.1,1.,(27,3))
    for count in (128,17,93,1,128):
        energy[:] = rng.choice([0.,1.],128)
        projections[:] = rng.uniform(0,1,(6,128))
        maxima[:] = projections[:,:count].max(axis=1)
        expected = [(b,j) for b in range(27) for j in range(count)
                    if energy[j] > 0 and cutoffs[b,0]*projections[2*(b%3),j]
                    +cutoffs[b,1]*projections[2*(b%3)+1,j] >= cutoffs[b,2]]
        selected,b,p,n = _subspace_select(energy,projections,maxima,cutoffs,templates,count,
                                         result,branches,positions,previous,*workspace[10:12])
        assert list(zip(b[:n],p[:n])) == expected
        np.testing.assert_array_equal(selected,np.zeros_like(result))
        selected[b[:n],p[:n]] = rng.uniform(1,2,n)


@pytest.mark.parametrize('backend',['scipy','fftw','fftw-python'])
@pytest.mark.parametrize('repeats',[1,2,4,8])
def test_power_history_wraps_preserve_all_branches_and_amplitudes(monkeypatch,backend,repeats):
    from signal_processing.lte import _fft_bank
    from signal_processing.lte._native import _ring_scores
    if backend.startswith('fftw'):
        pytest.importorskip('pyfftw')
        if backend == 'fftw-python':
            monkeypatch.setattr(_fft_bank,'optional_bank',_fft_bank.FFTBank)
    else:
        monkeypatch.setattr(_fft_bank,'optional_bank',lambda kernels: None)
    config = dataclasses.replace(LTESyncConfig.native(low_snr=True),integration_half_frames=repeats)
    receiver = LTESynchronizer(StreamInfo('wrap','oracle',1_920_000.),config,sync_backend='numba')
    assert receiver.execution['blind_fft_backend'] == ('fftw' if backend.startswith('fftw') else backend)
    rng = np.random.default_rng(271819)
    count = 3*receiver._power_ring_capacity+4096
    samples = (rng.normal(size=count)+1j*rng.normal(size=count)).astype(np.complex64)
    for start in range(0,count,4096):
        samples[start:start+4096] *= (1e-20,1.,1e20,0.)[(start//4096)%4]
    samples = samples.astype(np.complex128)
    for start in range(0,count,4096):
        end = min(start+4096,count)
        receiver._history.append(samples[start:end])
        receiver._buffer = receiver._history.view()
        receiver._prepare_blind_powers(end)
        stop = end-127
        begin = max(stop-2500,137+(repeats-1)*9600)
        if stop > begin:
            actual = _ring_scores(receiver._power_ring,receiver._energy_ring,begin,stop,
                                  repeats,receiver._power_ring_capacity)
            assert actual is not None
            for p in (begin,(begin+stop)//2,stop-1):
                power = np.zeros(len(receiver._power_templates_conjugate))
                energy = 0.
                for repeat in range(repeats):
                    window = samples[p-repeat*9600:p-repeat*9600+128]
                    energy += float(np.vdot(window,window).real)
                    power += np.abs([np.dot(t,window) for t in receiver._power_templates_conjugate])**2
                expected = power/(62*energy) if energy else np.zeros_like(power)
                np.testing.assert_allclose(actual[:,p-begin],expected,rtol=2e-12,atol=1e-15)
            from signal_processing.lte._native import _subspace_project,subspace_workspace
            direct,grouped = subspace_workspace(stop-begin,27),subspace_workspace(stop-begin,27)
            left = _subspace_project(receiver._power_ring,receiver._energy_ring,begin,stop,repeats,
                                     receiver._power_ring_capacity,*direct[:3])
            right = _subspace_project(receiver._group_power_ring,receiver._energy_ring,begin,stop,repeats,
                                      receiver._power_ring_capacity,*grouped[:3])
            for expected,observed in zip(left,right):
                np.testing.assert_array_equal(observed,expected)
        keep = max(0,end-300-(repeats-1)*9600)
        receiver._history.discard_before(keep)
        receiver._base = keep
    allocation = receiver.snapshot()['search_power_allocation_bytes']
    receiver.reset()
    assert receiver.snapshot()['search_power_allocation_bytes'] == allocation
    assert receiver._cached_blind_scores(137,200) is None


def test_native_blind_kernels_are_ready_before_first_received_iq():
    from signal_processing.core import IQBlock
    from signal_processing.lte._native import (
        _window_energy,_correlation_powers,_ring_scores,_subspace_project,_subspace_select,_subspace_dots,_repair_small_powers,_subthreshold_tone,_group_power_slice,
    )
    stream = StreamInfo('ready','noise',1_920_000.)
    receiver = LTESynchronizer(stream,LTESyncConfig.native(low_snr=True),sync_backend='numba')
    kernels = (_window_energy,_correlation_powers,_ring_scores,_subspace_project,_subspace_select,_subspace_dots,_repair_small_powers,_subthreshold_tone,_group_power_slice)
    before = [tuple(kernel.signatures) for kernel in kernels]
    native_before = None
    if receiver.execution['blind_fft_dispatch'] == 'numba-row-loop':
        from signal_processing.lte._fftw_native import _execute
        native_before = tuple(_execute.signatures)
    rng = np.random.default_rng(2233)
    for seq in range(4):
        iq = (rng.normal(size=16384)+1j*rng.normal(size=16384)).astype(np.complex64)
        assert not receiver.process_block(IQBlock(stream,seq,seq*len(iq),None,iq))
    assert not receiver.flush()
    assert [tuple(kernel.signatures) for kernel in kernels] == before
    if native_before is not None:
        assert tuple(_execute.signatures) == native_before


@pytest.mark.parametrize('threshold',[.10,.22])
def test_subspace_screen_keeps_every_qualifying_pss_branch(threshold):
    config = dataclasses.replace(LTESyncConfig.native(low_snr=True),pss_threshold=threshold)
    receiver = LTESynchronizer(StreamInfo('branches','oracle',1_920_000.),config,sync_backend='numba')
    rng = np.random.default_rng(7893)
    size = 32*4096
    samples = (rng.normal(size=size)+1j*rng.normal(size=size)).astype(np.complex128)
    positions = []
    for branch,template in enumerate(receiver._templates):
        p = 8*4096+branch*3072+500
        samples[p:p+128] += 8*template
        positions.append((p,branch))
    verified = 0
    for start in range(0,size,4096):
        stop = start+4096
        receiver._history.append(samples[start:stop])
        receiver._buffer = receiver._history.view()
        receiver._prepare_blind_powers(stop)
        for p,branch in positions:
            if start <= p < stop-127:
                full = receiver._cached_blind_scores(p,p+1)
                screened = receiver._cached_blind_scores(p,p+1,prune=True)
                assert full[branch,0] > threshold
                qualifying = full[:,0] >= threshold
                np.testing.assert_allclose(screened[qualifying,0],full[qualifying,0],rtol=1e-12)
                assert receiver._subspace_workspace[-1][0] == screened.max()
                verified += 1
        keep = max(0,stop-300-3*9600)
        receiver._history.discard_before(keep)
        receiver._base = keep
    assert verified == 27


def test_prefilter_can_be_disabled_without_changing_profile_identity():
    stream = StreamInfo('control','oracle',1_920_000.)
    config = LTESyncConfig.native(low_snr=True)
    fast = LTESynchronizer(stream,config,sync_backend='numba')
    reference = LTESynchronizer(stream,config,sync_backend='numba',blind_prefilter=False)
    assert fast.config.profile_id == reference.config.profile_id
    assert len(fast._search_kernels) == 21
    assert len(reference._search_kernels) == 27


def tone_templates():
    from signal_processing.lte.sequences import pss_sequence,sync_symbol
    return np.array([(sync_symbol(pss_sequence(n))*np.exp(2j*np.pi*f*np.arange(128)/1_920_000.)).conj()
        for f in np.arange(-30000,30001,7500) for n in range(3)])


def direct_maximum(samples,templates):
    energy = np.convolve(samples.real**2+samples.imag**2,np.ones(128),'valid')
    return max(float(np.max(np.abs(np.convolve(samples,t[::-1],'valid'))**2/(62*energy))) for t in templates)


@pytest.mark.parametrize('frequency',[-800000.,-100000.,0.,100000.,800000.])
@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_tone_bound_matches_all_direct_windows_and_never_rejects_a_matching_filter(frequency,scale):
    from signal_processing.lte._native import _subthreshold_tone,initialize_sync
    initialize_sync('numba')
    samples = (scale*np.exp(1j*(.731+2*np.pi*frequency*np.arange(2049)/1_920_000.))).astype(np.complex64).astype(np.complex128)
    samples.flags.writeable = False
    templates = tone_templates()
    assert direct_maximum(samples,templates) < .1
    assert _subthreshold_tone(samples,templates,.1)
    # The proof uses the ACTUAL filter response, not a rule declaring tones
    # uninteresting. A filter matched to the same tone must remain eligible.
    matched = (np.sqrt(62/128)*np.exp(-2j*np.pi*frequency*np.arange(128)/1_920_000.))[None,:]
    assert direct_maximum(samples,matched) > .99
    assert not _subthreshold_tone(samples,matched,.1)


@pytest.mark.parametrize('branch',range(27))
def test_tonal_background_does_not_suppress_any_real_qualifying_pss(branch):
    from signal_processing.lte._native import _subthreshold_tone,initialize_sync
    initialize_sync('numba')
    templates = tone_templates()
    samples = np.exp(2j*np.pi*100000*np.arange(769)/1_920_000.)
    samples[256:384] += 3*templates[branch].conj()
    samples = samples.astype(np.complex64).astype(np.complex128)
    samples.flags.writeable = False
    assert direct_maximum(samples,templates) > .1
    assert not _subthreshold_tone(samples,templates,.1)


@pytest.mark.parametrize('partition',[1024,16384])
def test_tone_to_real_lte_rebuilds_received_history_and_decodes(partition):
    from signal_processing.core import IQBlock
    from signal_processing.lte import LTEBroadcastReceiver
    from tests.lte_pbch_fixture import waveform
    stream = StreamInfo('tone-transition','encoded',1_920_000.)
    receiver = LTEBroadcastReceiver(stream,LTESyncConfig.native(low_snr=True),decode_cadence='bch_block')
    tone = np.exp(2j*np.pi*100000*np.arange(96000)/1_920_000.).astype(np.complex64)
    signal = waveform(42,ports=2,rate=1_920_000.,frames=8,snr_db=-5,cfo=12345)
    samples = np.concatenate((tone,signal,tone))
    mibs = []
    for seq,start in enumerate(range(0,len(samples),partition)):
        mibs.extend(receiver.process_block(IQBlock(stream,seq,start,None,samples[start:start+partition])).mibs)
    mibs.extend(receiver.flush().mibs)
    assert len(mibs) >= 2 and all(m.pci == 42 for m in mibs)
    assert receiver.snapshot()['tone_rejected_search_chunks'] > 0
    assert receiver.snapshot()['discontinuities'] == 0
    assert receiver.snapshot()['received_input_end'] == len(samples)
    assert len(receiver._search_kernels) == 21


@pytest.mark.parametrize('backend',['scipy','fftw'])
@pytest.mark.parametrize('frequency',[-100000.,100000.])
def test_dense_narrowband_search_switches_once_to_exact_prepared_bank(monkeypatch,backend,frequency):
    from signal_processing.lte import _fft_bank
    if backend == 'scipy':
        monkeypatch.setattr(_fft_bank,'optional_bank',lambda kernels: None)
    else:
        pytest.importorskip('pyfftw')
    receiver = LTESynchronizer(StreamInfo('tone','oracle',1_920_000.),
                              LTESyncConfig.native(low_snr=True),sync_backend='numba')
    allocation = receiver.snapshot()['search_power_allocation_bytes']
    plans = receiver._full_fft_bank
    monkeypatch.setattr(_fft_bank,'optional_bank',lambda kernels: pytest.fail('live FFT planning'))
    samples = np.exp(2j*np.pi*frequency*np.arange(40960)/1_920_000.)
    receiver._history.append(samples)
    receiver._buffer = receiver._history.view()
    receiver._prepare_blind_powers(len(samples))
    for begin in (32641,36737):
        scores = receiver._cached_blind_scores(begin,begin+4096,prune=True)
        assert scores is not None
        for position in (begin,begin+2000,begin+4095):
            energy = 0.
            power = np.zeros(27)
            for repeat in range(4):
                window = samples[position-repeat*9600:position-repeat*9600+128]
                energy += float(np.vdot(window,window).real)
                power += np.abs([np.dot(t,window) for t in receiver._templates_conjugate])**2
            np.testing.assert_allclose(scores[:,position-begin],power/(62*energy),rtol=2e-11,atol=1e-14)
    assert receiver.snapshot()['dense_search_fallbacks'] == 1
    assert receiver._fft_bank is plans
    assert receiver.snapshot()['search_power_allocation_bytes'] == allocation
    receiver.reset()
    assert len(receiver._search_kernels) == 21
    assert receiver._cached_blind_scores(30000,34096,prune=True) is None
    assert receiver.snapshot()['search_power_allocation_bytes'] == allocation


def test_lock_loss_and_reacquisition_use_bounded_native_history_without_full_fft_fallback():
    from signal_processing.core import IQBlock
    from signal_processing.lte import LTEBroadcastReceiver
    from tests.lte_pbch_fixture import waveform
    stream = StreamInfo('transition','encoded',1_920_000.)
    receiver = LTEBroadcastReceiver(stream,LTESyncConfig.native(low_snr=True),decode_cadence='bch_block')
    signal = waveform(42,ports=2,rate=1_920_000.,frames=8,snr_db=-5,cfo=12345)
    reference = LTEBroadcastReceiver(stream,LTESyncConfig.native(low_snr=True),
                                     decode_cadence='bch_block',blind_prefilter=False)
    rng = np.random.default_rng(8375)
    noise = (rng.normal(size=len(signal))+1j*rng.normal(size=len(signal))).astype(np.complex64)
    iq = np.concatenate((signal,noise,signal,noise,signal))
    def reject_rebuild(*args,**kwargs):
        raise AssertionError('A tracking miss must use already received correlation history')
    receiver._bounded_scores = reject_rebuild
    mibs,reference_mibs = [],[]
    screened_quiet = False
    for seq,start in enumerate(range(0,len(iq),16384)):
        block = IQBlock(stream,seq,start,None,iq[start:start+16384])
        mibs.extend(receiver.process_block(block).mibs)
        reference_mibs.extend(reference.process_block(block).mibs)
        if len(signal)+115200 <= receiver._prepared_end < 2*len(signal):
            screened_quiet |= receiver._subspace_cutoffs is not None
        if receiver._tracking_missed or receiver._locked is None:
            assert receiver._score_end == receiver._processed-127
        assert sum(b-a for a,b,_,_ in receiver._score_chunks) <= 300+3*9600+2*4096
    mibs.extend(receiver.flush().mibs)
    reference_mibs.extend(reference.flush().mibs)
    assert receiver.snapshot()['received_input_end'] == len(iq)
    assert receiver.snapshot()['discontinuities'] == 0
    assert any(m.input_frame_start < len(signal) for m in mibs)
    assert any(m.input_frame_start >= 2*len(signal) for m in mibs)
    assert screened_quiet
    def decoded(values):
        return [(m.pci,m.antenna_ports,m.frame_quarter,m.input_frame_start,m.mib.payload_bits) for m in values]
    assert decoded(mibs) == decoded(reference_mibs)
