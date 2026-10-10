"""Audit-derived regressions with real IQ, required sinks and spawned sources."""
import threading

import numpy as np
import pytest

from benchmarks import process_source, sustained_evaluation
from signal_processing.core import IQBlock, StreamInfo
from signal_processing.runtime import FeatureAnalysisWorker


def _exit_before_readiness(connection, output, stop, total, block_size, rate):
    connection.close()


def test_source_startup_failure_closes_initialized_feature_worker(monkeypatch):
    pipelines = []
    monkeypatch.setattr(process_source, '_run', _exit_before_readiness)
    with pytest.raises(EOFError):
        sustained_evaluation.bounded_acquisition(.1, sample_rate_hz=10000,
            features=True, producer_backend='process', analysis_backend='process',
            configure_pipeline=pipelines.append)
    assert pipelines[0].worker.state.value == 'CLOSED'
    assert pipelines[0].worker.unfinished_work == 0


def test_quiet_windows_are_reduced_once(monkeypatch):
    from signal_processing.detection import energy_detector
    calls = []
    original = energy_detector.window_mean_powers
    def counted(iq, n, h):
        result = original(iq, n, h)
        calls.extend([n]*len(result))
        return result
    monkeypatch.setattr(energy_detector, 'window_mean_powers', counted)
    stream = StreamInfo('one-pass', 'synthetic', 1e6)
    pipe = sustained_evaluation._pipeline(stream, fixed_quiet_batch=True)
    try:
        assert pipe.process_block(IQBlock(stream, 0, 1000, None,
            np.full(1024, .003, np.complex64))) == []
        assert len(calls) == (1024-64)//32+1
        assert pipe.consumed_end_sample == 2024
    finally:
        pipe.shutdown()


def test_prepared_quiet_state_matches_public_detector_and_cannot_replay():
    from signal_processing.detection import EnergyDetector
    from signal_processing.core import NoiseEstimate
    from signal_processing.core.exceptions import StreamConsistencyError
    stream = StreamInfo('prepared', 'synthetic', 1e6)
    reference = EnergyDetector(window_size=17, hop_size=8)
    fast = EnergyDetector(window_size=17, hop_size=8)
    estimate = NoiseEstimate(stream.stream_id, 0, -40)
    for seq, size in enumerate((8, 24, 1024, 64)):
        start = 10**12 if seq == 0 else block.end_sample
        block = IQBlock(stream, seq, start, None, np.full(size, .003, np.complex64))
        expected = reference.process_windows(block, estimate)
        prepared = fast._prepare_quiet_full_band(block, estimate)
        if prepared is None:
            assert fast.process_windows(block, estimate) == expected
        else:
            assert fast._input_end == prepared.prior_input_end
            foreign = EnergyDetector(window_size=17, hop_size=8)
            with pytest.raises(StreamConsistencyError):
                foreign._commit_quiet_full_band(prepared)
            assert foreign._input_end is None
            assert fast._commit_quiet_full_band(prepared) == expected[-1]
            snapshot = fast._input_end, fast._next_window_sample, fast._residual_iq.tobytes()
            with pytest.raises(StreamConsistencyError):
                fast._commit_quiet_full_band(prepared)
            assert snapshot == (fast._input_end, fast._next_window_sample, fast._residual_iq.tobytes())
        assert fast._input_end == reference._input_end
        assert fast._next_window_sample == reference._next_window_sample
        np.testing.assert_array_equal(fast._residual_iq, reference._residual_iq)
    prepared = fast._prepare_quiet_full_band(IQBlock(stream, 4, block.end_sample,
        None, np.full(64, .003, np.complex64)), estimate)
    fast.reset()
    with pytest.raises(StreamConsistencyError):
        fast._commit_quiet_full_band(prepared)


@pytest.mark.parametrize('unknown_loss', [False, True])
def test_single_pass_quiet_then_active_metadata_retry_and_gap(unknown_loss):
    import dataclasses
    from signal_processing.detection import GaussianNoiseEstimator
    from signal_processing.core.exceptions import StreamConsistencyError
    from signal_processing.runtime import StreamingCapturePipeline
    stream = StreamInfo('gap-one-pass', 'synthetic', 1e6)
    origin = 10**12
    outputs = []
    for fast in (False, True):
        pipe = StreamingCapturePipeline(stream, window_size=17, hop_size=8,
            ring_capacity=4096, fixed_quiet_batch=fast, enable_feature_analysis=False,
            start_persistence=1, end_persistence=1, pre_trigger_samples=8,
            pre_signal_margin_samples=0, post_signal_samples=0, min_post_trigger_samples=0,
            noise_estimator=GaussianNoiseEstimator(window_size=17, calibrated_noise_power_linear=1e-4))
        try:
            events = pipe.process_block(IQBlock(stream, 0, origin, None, np.full(64, .003, np.complex64)))
            events += pipe.process_block(IQBlock(stream, 1, origin+64, None, np.full(17, .1, np.complex64)))
            good = IQBlock(stream, 2, origin+113, None, np.full(64, .003, np.complex64),
                           dropped_samples_before=32, overflow=unknown_loss)
            bad = dataclasses.replace(good, stream=dataclasses.replace(stream, sample_rate_hz=1e6+.0005))
            def snapshot():
                return (pipe.consumed_end_sample, pipe.received_end_sample,
                        pipe._input_residual_iq.tobytes(), pipe.detector._input_end,
                        pipe.detector._residual_iq.tobytes(), pipe.ring_buffer.stream_generation)
            before = snapshot()
            with pytest.raises(StreamConsistencyError):
                pipe.process_block(bad)
            assert snapshot() == before and pipe.state.value == 'OPEN'
            events += pipe.process_block(good)
            events += pipe.flush()
            outputs.append([(sc.event.event_start_sample, sc.event.event_end_sample,
                sc.event.iq.tobytes(), sc.stream_epoch, sc.known_lost_samples,
                sc.unknown_loss, sc.termination_reason) for sc in events])
        finally:
            pipe.shutdown()
    assert outputs[0] == outputs[1] and outputs[0]


def real_capture():
    stream = StreamInfo('stage1', 'synthetic', 1e6)
    pipe = sustained_evaluation._pipeline(stream)
    try:
        x = np.full(1024, .003, np.complex64)
        x[:256] = .1
        captures = pipe.process_block(IQBlock(stream, 0, 0, None, x)) + pipe.flush()
        return captures[0]
    finally:
        pipe.shutdown()


@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_required_sink_remains_outstanding_until_completion(backend):
    entered, release = threading.Event(), threading.Event()
    def sink(result):
        assert result.feature_status == 'ok'
        entered.set()
        assert release.wait(8)
    worker = FeatureAnalysisWorker(synchronous=False, analysis_backend=backend,
                                  result_delivery='callbacks', result_sink=sink)
    try:
        assert worker.submit(real_capture()).feature_status == 'pending'
        assert entered.wait(8)
        assert worker.unfinished_work == 1
        with pytest.raises(TimeoutError):
            worker.shutdown(timeout=.02)
        assert worker.state.value == 'CLOSING' and worker.unfinished_work == 1
    finally:
        release.set()
        worker.shutdown(timeout=5)
    assert worker.unfinished_work == 0 and worker.state.value == 'CLOSED'


def test_real_source_death_cannot_escape_required_cleanup(monkeypatch):
    sources, pipelines = [], []
    original = process_source.ProcessScheduledSource
    class KilledSource(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sources.append(self)

        def start(self, started):
            self._process.terminate()
            self._process.join(2)
            assert not self._process.is_alive()
    monkeypatch.setattr(process_source, 'ProcessScheduledSource', KilledSource)
    try:
        result = sustained_evaluation.bounded_acquisition(.1, sample_rate_hz=10000,
            block_size=128, features=True, producer_backend='process',
            analysis_backend='process', configure_pipeline=pipelines.append)
        assert not result['passed'] and not result['source_accounting_complete']
        assert not result['sample_accounting_balanced']
        assert result['unaccounted_samples'] == 1000
        assert result['dropped_samples'] == 0  # Missing evidence is not measured loss.
        assert any('Scheduled source died' in e for e in result['errors'])
        assert result['worker_state'] == 'CLOSED'
        assert result['worker_unfinished'] == 0
        assert not result['producer_alive']
        assert sources[0]._closed
    finally:
        # Also clean up the broken baseline when demonstrating fail-before.
        for pipe in pipelines:
            pipe.shutdown(timeout=5)
        for source in sources:
            if not getattr(source, '_closed', False):
                source._connection.close()
                source.close()
