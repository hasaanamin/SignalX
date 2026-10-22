"""Real spawn/slot leases: no fabricated processed IQ or terminal records."""
import dataclasses
import multiprocessing
import time
import threading

import numpy as np
import pytest

from signal_processing.core import StreamInfo,IQBlock
from signal_processing.runtime.process_source_runner import ProcessSourceRunner, ReplayFileFactory
from signal_processing.runtime.source_runner import SourceRunnerFailed
from signal_processing.sources.streaming import SourceReadTimeout,SourceEnvelope,SourceMetadata,SourceCancelled


def _factory_failure():
    raise RuntimeError('genuine source startup failure')


@pytest.mark.parametrize('explicit_cpus',[False,True])
def test_scoped_acquisition_policy_preserves_bytes_and_reports_real_cleanup(tmp_path,explicit_cpus):
    import sys
    samples = np.full(16384,.01+.02j,np.complex64)
    path = tmp_path/'policy.cf32'
    samples.tofile(path)
    stream = StreamInfo('policy','recording',1_920_000.)
    import psutil
    selected = [psutil.Process().cpu_affinity()[0]] if explicit_cpus else None
    runner = ProcessSourceRunner(ReplayFileFactory(str(path),stream),stream,performance_qos=True,
                                 acquisition_cpus=selected)
    try:
        runner.start(start_paused=True)
        ready = runner.source_execution()
        if selected is not None:
            assert psutil.Process(runner.pid).cpu_affinity() == selected
        assert ready is not None
        if sys.platform == 'win32':
            assert ready['performance']['applied'] and ready['performance']['thread_qos_applied']
            assert ready['realtime']['applied'] and ready['realtime']['priority'] == 2
        runner.begin()
        lease = runner.get(timeout=5)
        assert lease.envelope.block.iq.tobytes() == samples.tobytes()
        runner.ack(lease)
        assert runner.get(timeout=5) is None
    finally:
        runner.shutdown(drain=False)
    finished = runner.source_execution()
    assert finished['performance']['restored'] and finished['realtime']['restored']
    assert runner.source_snapshot().expired_samples == 0
    assert runner.snapshot().processed_samples == len(samples)
    assert runner.snapshot().outstanding_deliveries == 0 and not runner.snapshot().reader_alive


class _PositionedLossSource:
    stream_info=StreamInfo('loss','positioned',1e6,2.14e9)
    def __init__(self): self.sequence=0;self.cancelled=False
    def read_envelope(self,timeout=None):
        if self.cancelled: raise SourceCancelled('cancelled')
        seq=self.sequence
        if seq==2: return None
        self.sequence+=1
        start=10**12+seq*64
        block=IQBlock(self.stream_info,seq,start,10**9+seq*64000,np.full(32,.003,np.complex64),
                      dropped_samples_before=32 if seq else 0,overflow=bool(seq))
        return SourceEnvelope(block,SourceMetadata(timestamp_origin='recording'),time.perf_counter_ns())
    def cancel(self): self.cancelled=True
    def close(self): self.cancelled=True


@dataclasses.dataclass
class _GatedFactory:
    entered: object
    release: object
    def __call__(self):
        self.entered.set()
        if not self.release.wait(5): raise TimeoutError('Factory gate timed out')
        return _PositionedLossSource()


@dataclasses.dataclass
class _GatedPublication:
    semaphore: object
    entered: object
    release_gate: object
    def release(self):
        self.entered.set()
        if not self.release_gate.wait(5): raise TimeoutError('Publication gate timed out')
        self.semaphore.release()
    def acquire(self,*args,**kwargs): return self.semaphore.acquire(*args,**kwargs)


def make_runner(tmp_path, *, blocks=12, capacity=3, scheduled=False):
    x = (np.arange(blocks*32,dtype=np.float32)+.5j).astype(np.complex64)
    path = tmp_path/'source.cf32'
    x.tofile(path)
    stream = StreamInfo('spawn','file',1000.,2.14e9)
    factory = ReplayFileFactory(str(path),stream,32,scheduled,1.)
    return ProcessSourceRunner(factory,stream,capacity=capacity,max_block_samples=32), x


def test_process_loss_accounting_exact_iq_and_foreign_ack(tmp_path):
    runner,x = make_runner(tmp_path,blocks=12)
    runner.start()
    try:
        # Finite unpaced source deterministically fills slots before any ack.
        deadline = time.monotonic()+5
        while runner.snapshot().reader_alive and time.monotonic()<deadline:
            time.sleep(.005)
        assert not runner.snapshot().reader_alive
        leases = []
        while (lease := runner.get(timeout=2)) is not None:
            leases.append(lease)
            np.testing.assert_array_equal(lease.envelope.block.iq,x[lease.envelope.block.start_sample:lease.envelope.block.end_sample])
            assert not lease.envelope.block.iq.flags.writeable
        assert len(leases)==3
        with pytest.raises(ValueError): runner.ack(dataclasses.replace(leases[0]))
        for lease in reversed(leases): runner.ack(lease)
        with pytest.raises(ValueError): runner.ack(leases[0])
        runner.shutdown()
        s = runner.snapshot()
        assert s.accounting_balanced and s.state=='CLOSED' and s.peak_outstanding==3
        assert s.processed_samples==96 and s.queue_dropped_samples==288
        assert s.trailing_lost_samples==288 and s.outstanding_deliveries==0
        assert runner.source_snapshot().returned_samples==384
    finally:
        runner.shutdown(drain=False)


def test_out_of_order_ack_cannot_recycle_another_lease(tmp_path):
    runner,x = make_runner(tmp_path,blocks=8,capacity=3,scheduled=True)
    runner.start()
    a,b,c = (runner.get(timeout=3) for _ in range(3))
    saved_a,saved_c = a.envelope.block.iq.copy(), c.envelope.block.iq.copy()
    runner.ack(b)
    d = runner.get(timeout=3)
    np.testing.assert_array_equal(d.envelope.block.iq,x[d.envelope.block.start_sample:d.envelope.block.end_sample])
    np.testing.assert_array_equal(a.envelope.block.iq,saved_a)
    np.testing.assert_array_equal(c.envelope.block.iq,saved_c)
    runner.ack(a); runner.ack(c); runner.ack(d)
    while (lease := runner.get(timeout=3)) is not None:
        np.testing.assert_array_equal(lease.envelope.block.iq,x[lease.envelope.block.start_sample:lease.envelope.block.end_sample])
        runner.ack(lease)
    runner.shutdown()
    assert runner.snapshot().accounting_balanced


def test_timeout_retry_and_cancelled_queued_work(tmp_path):
    runner,_ = make_runner(tmp_path,scheduled=True)
    runner.start()
    lease = runner.get(timeout=3)
    with pytest.raises(TimeoutError): runner.shutdown(drain=False,timeout=.02)
    assert runner.snapshot().state=='CLOSING'
    runner.ack(lease)
    runner.shutdown(drain=False,timeout=3)
    s = runner.snapshot()
    assert s.state=='CLOSED' and s.accounting_balanced and not s.outstanding_deliveries


@pytest.mark.parametrize('spin',[False,True])
def test_real_process_death_is_failure_not_eof(tmp_path,spin):
    runner,_ = make_runner(tmp_path,scheduled=True)
    runner.start()
    lease = runner.get(timeout=3,spin=spin)
    runner._process.terminate()  # Real OS worker death, not a fake worker/result.
    runner._process.join(3)
    runner.ack(lease)
    with pytest.raises(SourceRunnerFailed):
        while (queued := runner.get(timeout=3,spin=spin)) is not None: runner.ack(queued)
    runner.shutdown(drain=False)
    assert runner.snapshot().error and runner.snapshot().state=='CLOSED'


def test_startup_failure_is_reported_and_reaped():
    runner=ProcessSourceRunner(_factory_failure,StreamInfo('bad','source',1000.))
    with pytest.raises(SourceRunnerFailed,match='genuine source startup failure'): runner.start()
    runner.shutdown(drain=False,timeout=3)
    assert runner.snapshot().state=='CLOSED' and runner.snapshot().error


@pytest.mark.parametrize('spin',[False,True])
def test_waiting_get_does_not_lock_out_shutdown(tmp_path,spin,monkeypatch):
    runner,_ = make_runner(tmp_path,scheduled=True)
    runner.start(start_paused=True)
    entered = threading.Event()
    original_poll = runner._poll
    def poll():
        result = original_poll()
        entered.set()
        return result
    monkeypatch.setattr(runner,'_poll',poll)
    outcomes=[]
    def consume():
        try:
            while (lease := runner.get(timeout=2,spin=spin)) is not None: runner.ack(lease)
            outcomes.append('done')
        except SourceReadTimeout: outcomes.append('timeout')
    t=threading.Thread(target=consume)
    t.start()
    assert entered.wait(2)
    runner.shutdown(drain=True,timeout=3)
    t.join(3)
    assert not t.is_alive() and outcomes==['done']
    assert runner.snapshot().accounting_balanced


@pytest.mark.parametrize('spin',[False,True])
def test_explicit_readiness_has_no_phantom_clock_or_samples(tmp_path,spin):
    runner,_=make_runner(tmp_path,blocks=1)
    runner.start(start_paused=True)
    assert runner.pid and runner.snapshot().state=='READY'
    assert runner.snapshot().read_samples==0
    with pytest.raises(SourceReadTimeout): runner.get(timeout=0,spin=spin)
    with pytest.raises(SourceReadTimeout): runner.get(timeout=.01,spin=spin)
    runner.begin()
    with pytest.raises(RuntimeError): runner.begin()
    lease=runner.get(timeout=3,spin=spin)
    runner.ack(lease)
    assert runner.get(timeout=3,spin=spin) is None
    runner.shutdown()
    assert runner._slots is None and runner._ready is None
    assert runner.get(timeout=0,spin=spin) is None


def test_spin_selection_rejects_integer_before_source_start(tmp_path):
    runner,_ = make_runner(tmp_path)
    with pytest.raises(ValueError,match='Spin wait'):
        runner.get(timeout=0,spin=1)


def test_malformed_actual_slot_is_failed_accounted_and_recoverable(tmp_path):
    runner,_=make_runner(tmp_path)
    runner.start()
    deadline=time.monotonic()+5
    while runner.snapshot().reader_alive and time.monotonic()<deadline: time.sleep(.005)
    # Corrupt a genuine bounded transport packet, not the analysis result.
    runner._slots[0] ^= 1
    with pytest.raises(SourceRunnerFailed,match='Invalid acquired envelope'): runner.get(timeout=3)
    runner.shutdown(drain=False,timeout=3)
    s=runner.snapshot()
    assert s.failed_samples==32 and s.cancelled_samples==64
    assert s.accounting_balanced and not s.outstanding_deliveries and s.error
    assert runner._slots is None


def test_actual_known_unknown_loss_large_coordinates_and_timestamp_survive_spawn():
    runner=ProcessSourceRunner(_PositionedLossSource,_PositionedLossSource.stream_info,capacity=2,max_block_samples=32)
    runner.start()
    a,b=runner.get(timeout=3),runner.get(timeout=3)
    assert a.envelope.block.start_sample==10**12
    block=b.envelope.block
    assert block.start_sample==10**12+64 and block.dropped_samples_before==32 and block.overflow
    assert block.timestamp_ns==1000064000 and b.envelope.metadata.timestamp_origin=='recording'
    assert b.envelope.received_monotonic_ns>0 and b.envelope.scheduled_monotonic_ns is None
    runner.ack(b);runner.ack(a)
    assert runner.get(timeout=3) is None
    runner.shutdown()
    assert runner.snapshot().accounting_balanced


def test_shutdown_deadline_during_real_blocked_source_startup():
    ctx=multiprocessing.get_context('spawn')
    entered,release=ctx.Event(),ctx.Event()
    runner=ProcessSourceRunner(_GatedFactory(entered,release),_PositionedLossSource.stream_info)
    failures=[]
    def start():
        try: runner.start()
        except SourceCancelled as exc: failures.append(str(exc))
    thread=threading.Thread(target=start)
    thread.start()
    try:
        assert entered.wait(5)
        before=time.monotonic()
        with pytest.raises(TimeoutError): runner.shutdown(timeout=.02)
        assert time.monotonic()-before<.5
        assert runner.snapshot().state=='CLOSING' and runner.snapshot().read_samples==0
        release.set();thread.join(3)
        assert not thread.is_alive() and len(failures)==1
        runner.shutdown(timeout=3)
        assert runner.snapshot().state=='CLOSED' and runner.snapshot().accounting_balanced
    finally:
        release.set();thread.join(5);runner.shutdown(drain=False,timeout=5)


def test_death_between_real_admission_and_ready_publication_is_terminal_failure(tmp_path):
    runner,_=make_runner(tmp_path,blocks=1)
    entered,release=runner._ctx.Event(),runner._ctx.Event()
    # Pause the REAL semaphore publication after genuine IQ has been copied.
    runner._ready=_GatedPublication(runner._ready,entered,release)
    runner.start()
    try:
        assert entered.wait(5)
        runner._process.terminate();runner._process.join(3)
        with pytest.raises(SourceRunnerFailed): runner.get(timeout=1)
        runner.shutdown(drain=False,timeout=2)
        s=runner.snapshot()
        assert s.admitted_samples==s.failed_samples==32
        assert s.queued_samples==s.outstanding_deliveries==0
        assert s.state=='CLOSED' and s.accounting_balanced and s.error
    finally:
        # A terminated waiter cannot acknowledge multiprocessing.Condition's
        # notify_all handshake. Signalling its Event after death can block the
        # test forever (and hides the already completed runner assertions).
        if runner.snapshot().reader_alive:
            release.set()
        runner.shutdown(drain=False,timeout=3)
