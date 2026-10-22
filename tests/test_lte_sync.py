"""Independent sequence, rational channel and public LTE synchronization tests."""
from __future__ import annotations

import dataclasses
from fractions import Fraction
import json

import numpy as np
import pytest
from scipy.signal import upfirdn

from signal_processing.core import IQBlock, StreamInfo, StreamConsistencyError
from signal_processing.lte import LTESynchronizer, LTESyncConfig, LTESyncObservation
from signal_processing.lte.channel import LTEChannel
from signal_processing.lte.sequences import pss_sequence, sss_sequence, sync_symbol, normal_cp_lengths
from signal_processing.sources import ArrayIQSource, SourceEnvelope, SourceMetadata
from tests.lte_fixture import REFERENCE_PATH, waveform


def run(iq, *, rate=2_000_000., block=4093, origin=0, config=None):
    stream = StreamInfo("lte", "reference", rate, 2.14e9)
    receiver = LTESynchronizer(stream, config)
    source = ArrayIQSource(iq, stream, block_size=block, start_sample=origin)
    observations = []
    while (chunk := source.read_block()) is not None:
        observations.extend(receiver.process_block(chunk))
    observations.extend(receiver.flush())
    return receiver, observations


def test_all_reference_sequences_match_original_compiled_srsran_and_scalar_pss():
    with np.load(REFERENCE_PATH, allow_pickle=False) as reference:
        for sector in range(3):
            np.testing.assert_allclose(pss_sequence(sector), reference["pss"][sector], rtol=0, atol=2e-12)
        for pci in range(504):
            for half in (0, 1):
                np.testing.assert_array_equal(sss_sequence(pci, half), reference["sss"][pci, half])
    assert sum(normal_cp_lengths())+7*128 == 960


def test_ofdm_symbol_matches_independent_direct_dft_sum():
    sequence = pss_sequence(1)
    bins = list(range(-31, 0))+list(range(1, 32))
    expected = np.array([sum(v*np.exp(2j*np.pi*k*n/128) for k, v in zip(bins, sequence))/np.sqrt(128)
                         for n in range(128)])
    np.testing.assert_allclose(sync_symbol(sequence), expected, rtol=0, atol=3e-14)


@pytest.mark.parametrize("partition", [1, 7, 25, 103, 4096])
def test_rational_channel_is_partition_invariant_and_has_exact_sample_mapping(partition):
    rng = np.random.default_rng(2233)
    iq = (rng.normal(size=4099)+1j*rng.normal(size=4099)).astype(np.complex64)
    iq[0] = 1e8
    channel = LTEChannel(2_000_000.)
    expected = upfirdn(channel.taps, iq.astype(np.complex128), up=24, down=25)[:(len(iq)*24+24)//25]
    results = []
    for start in range(0, len(iq), partition):
        index, samples = channel.process(iq[start:start+partition])
        assert index == sum(map(len, results))
        results.append(samples)
        assert channel.retained_samples <= 100
    np.testing.assert_allclose(np.concatenate(results), expected, rtol=1e-13, atol=1e-12)
    assert channel.input_position(24) == 0
    assert channel.input_position(25) == Fraction(25, 24)
    # Independent explicit upsample/convolve control, not the streaming algorithm.
    small = iq[:80].astype(np.complex128)
    up = np.zeros(len(small)*24, np.complex128)
    up[::24] = small
    direct = np.convolve(up, channel.taps)[::25][:(len(small)*24+24)//25]
    other = LTEChannel(2_000_000.)
    np.testing.assert_allclose(other.process(small)[1], direct, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("rate", [1_920_000., 2_000_000.])
@pytest.mark.parametrize("pci", [0, 1, 2, 42, 167, 251, 502, 503])
def test_real_sync_iq_recovers_identity_timing_and_repeated_confirmation(rate, pci):
    iq = waveform(pci, frames=2, rate=rate, cfo=13_200., snr_db=10)
    receiver, observations = run(iq, rate=rate, origin=10**12)
    confirmed = [o for o in observations if o.state == "SYNCHRONIZED"]
    assert confirmed and all(o.pci == pci for o in confirmed)
    assert receiver.snapshot()["pci"] == pci
    for o in confirmed:
        assert abs(o.frequency_offset_hz-13_200) < 700
        frame_offset = o.input_frame_start-Fraction(10**12+431)
        assert abs(float(frame_offset % int(rate/100))) < 2 or abs(float(frame_offset % int(rate/100))-rate/100) < 2
        assert o.confirmations >= 2
        restored = LTESyncObservation.from_dict(json.loads(json.dumps(o.to_dict(), allow_nan=False)))
        assert restored == o


@pytest.mark.parametrize("cfo", [-30_000., -12_345., 0., 7_500., 30_000.])
def test_zero_db_noise_and_signed_frequency_offsets(cfo):
    _, observations = run(waveform(301, frames=4, cfo=cfo, snr_db=0, seed=2233))
    confirmed = [o for o in observations if o.state == "SYNCHRONIZED"]
    assert confirmed and all(o.pci == 301 for o in confirmed)
    assert abs(confirmed[-1].frequency_offset_hz-cfo) < 1200


def test_transport_partition_and_origin_do_not_change_sync_measurements():
    iq = waveform(174, frames=2, snr_db=5, cfo=-19_100)
    _, expected = run(iq, block=16384)
    assert expected
    for size in (1, 137, 997, 29000):
        _, actual = run(iq, block=size, origin=10**12)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert (a.pci, a.state, a.processing_pss_sample, a.processing_frame_start) == (b.pci, b.state, b.processing_pss_sample, b.processing_frame_start)
            assert a.input_pss_sample-10**12 == b.input_pss_sample
            assert a.input_frame_start-10**12 == b.input_frame_start
            np.testing.assert_allclose([a.pss_metric, a.sss_metric, a.frequency_offset_hz],
                                       [b.pss_metric, b.sss_metric, b.frequency_offset_hz], rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("kind", ["zero", "noise", "tone"])
def test_negative_controls_do_not_lock(kind):
    rng = np.random.default_rng(842)
    if kind == "zero":
        iq = np.zeros(100_000, np.complex64)
    elif kind == "tone":
        iq = np.exp(2j*np.pi*.111*np.arange(100_000)).astype(np.complex64)
    else:
        iq = (rng.normal(size=100_000)+1j*rng.normal(size=100_000)).astype(np.complex64)
    receiver, observations = run(iq)
    assert not any(o.state == "SYNCHRONIZED" for o in observations)
    assert receiver.snapshot()["pci"] is None


def test_atomic_rejection_with_pending_filter_and_search_state_corrected_retry():
    iq = waveform(42, frames=1)
    stream = StreamInfo("s", "r", 2e6, 2.14e9)
    receiver = LTESynchronizer(stream)
    first = IQBlock(stream, 0, 10**12, 1_000_000_000, iq[:29])
    receiver.process_block(first)
    good = IQBlock(stream, 1, 10**12+29, 1_000_014_500, iq[29:1000])
    before = receiver.snapshot()
    for bad in (dataclasses.replace(good, stream=dataclasses.replace(stream, sample_rate_hz=2e6+.0005)),
                dataclasses.replace(good, stream=dataclasses.replace(stream, center_frequency_hz=2.14e9+.05)),
                dataclasses.replace(good, timestamp_ns=2_000_014_500),
                dataclasses.replace(good, start_sample=10**12+30)):
        with pytest.raises(StreamConsistencyError):
            receiver.process_block(bad)
        assert receiver.snapshot() == before
    receiver.process_block(good)
    assert receiver.snapshot()["received_input_end"] == 10**12+1000


def test_gap_restarts_filter_and_confirmation_without_splicing_or_losing_old_results():
    iq = waveform(42, frames=2)
    stream = StreamInfo("s", "r", 2e6)
    receiver = LTESynchronizer(stream)
    first = receiver.process_block(IQBlock(stream, 0, 0, None, iq))
    first += receiver.flush()
    assert any(o.state == "SYNCHRONIZED" for o in first)
    receiver.reset()
    observations = receiver.process_block(IQBlock(stream, 0, 0, None, iq[:20000]))
    assert observations
    after = receiver.process_block(IQBlock(stream, 1, 20123, None, iq[:20000], 123, True))
    after += receiver.flush()
    assert receiver.snapshot()["discontinuities"] == 1
    assert all(o.epoch > observations[0].epoch for o in after)
    assert any(o.state == "SYNCHRONIZED" for o in first)


def test_eof_does_not_pad_a_missing_pss_or_duplicate_outputs_and_replay_reset_works():
    iq = waveform(76, frames=1)
    receiver, observations = run(iq)
    assert observations
    assert receiver.flush() == ()
    stream = StreamInfo("lte", "reference", 2e6, 2.14e9)
    with pytest.raises(RuntimeError):
        receiver.process_block(IQBlock(stream, 99, len(iq), None, iq[:1]))
    receiver.reset()
    replayed = []
    for b in (ArrayIQSource(iq, stream, len(iq)).read_block(),):
        replayed.extend(receiver.process_block(b))
    replayed.extend(receiver.flush())
    assert [o.pci for o in replayed] == [o.pci for o in observations]


def test_envelope_provenance_and_configuration_changes_reject_before_mutation():
    iq = waveform(2, frames=1)
    stream = StreamInfo("s", "r", 2e6)
    receiver = LTESynchronizer(stream)
    e = SourceEnvelope(IQBlock(stream, 0, 0, None, iq), SourceMetadata(9, 3, "unknown", "manual", 24))
    observations = receiver.process_envelope(e)
    assert observations
    assert all(o.source_metadata == e.metadata and o.source_configuration_id == e.configuration_id for o in observations)
    assert LTESyncObservation.from_dict(json.loads(json.dumps(observations[0].to_dict()))) == observations[0]
    next_e = dataclasses.replace(e, block=IQBlock(stream, 1, len(iq), None, iq),
                                metadata=dataclasses.replace(e.metadata, gain_db=25))
    before = receiver.snapshot()
    with pytest.raises(StreamConsistencyError):
        receiver.process_envelope(next_e)
    assert receiver.snapshot() == before


def test_complete_residual_sync_pair_is_evaluated_before_gap_just_like_eof():
    iq = waveform(87, frames=1, rate=1_920_000., onset=0)
    stream = StreamInfo("s", "r", 1_920_000.)
    eof = LTESynchronizer(stream)
    gap = LTESynchronizer(stream)
    # The first SSS/PSS pair completes at sample 960, before a 4096 search chunk.
    prefix = IQBlock(stream, 0, 0, None, iq[:960])
    assert eof.process_block(prefix) == gap.process_block(prefix) == ()
    expected = eof.flush()
    assert len(expected) == 1 and expected[0].pci == 87
    actual = gap.process_block(IQBlock(stream, 1, 992, None, np.zeros(64, np.complex64), 32))
    assert len(actual) == 1
    assert actual[0].to_dict() == expected[0].to_dict()
    assert gap.snapshot()["epoch"] == expected[0].epoch+1


def test_partial_fault_preserves_observation_from_inside_search_and_typed_cause(monkeypatch):
    iq = waveform(87, frames=1, rate=1_920_000., onset=0)
    stream = StreamInfo("s", "r", 1_920_000.)
    # A 16k search includes more than one genuine synchronization observation.
    receiver = LTESynchronizer(stream, LTESyncConfig(input_sample_rate_hz=1_920_000., search_chunk_samples=16384))
    original = receiver._decode
    calls = []

    def fail_after_first(*args):
        result = original(*args)
        if result is not None:
            calls.append(result)
            if len(calls) == 2:
                raise ArithmeticError("injected later numeric failure")
        return result

    monkeypatch.setattr(receiver, "_decode", fail_after_first)
    with pytest.raises(ArithmeticError) as exc:
        receiver.process_block(IQBlock(stream, 0, 0, None, iq))
    assert len(exc.value.completed_observations) == 1
    assert exc.value.completed_observations[0].pci == 87
    snap = receiver.snapshot()
    assert snap["state"] == "FAULTED"
    assert exc.value.received_input_end == len(iq)
    assert exc.value.prepared_input_end == 16384
    with pytest.raises(RuntimeError):
        receiver.flush()


def test_lock_expires_without_refresh_and_state_memory_is_bounded():
    iq = waveform(1, frames=2)
    stream = StreamInfo("s", "r", 2e6)
    receiver = LTESynchronizer(stream)
    receiver.process_block(IQBlock(stream, 0, 0, None, iq))
    assert receiver.snapshot()["state"] == "SYNCHRONIZED"
    rng = np.random.default_rng(720)
    end = len(iq)
    for seq in range(1, 31):
        noise = (rng.normal(size=16384)+1j*rng.normal(size=16384)).astype(np.complex64)
        receiver.process_block(IQBlock(stream, seq, end, None, noise))
        end += len(noise)
        snap = receiver.snapshot()
        assert snap["retained_processing_samples"] < 4096+300
        assert snap["retained_resampler_samples"] <= 100
        assert snap["candidate_cells"] <= receiver.config.max_cells
    assert receiver.snapshot()["pci"] is None
    assert receiver.snapshot()["lock_losses"] >= 1


@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_sync_metrics_preserve_physical_scale(scale):
    iq = waveform(76, frames=2, snr_db=10)
    _, expected = run(iq)
    _, scaled = run((iq*scale).astype(np.complex64))
    assert [(o.pci, o.state, o.processing_pss_sample) for o in scaled] == [(o.pci, o.state, o.processing_pss_sample) for o in expected]
    np.testing.assert_allclose([o.sss_metric for o in scaled], [o.sss_metric for o in expected], rtol=2e-6)


def test_configuration_identity_is_immutable_and_all_controls_participate():
    config = LTESyncConfig()
    assert dataclasses.replace(config, pss_threshold=.23).profile_id != config.profile_id
    receiver = LTESynchronizer(StreamInfo("s", "r", 2e6), config)
    with pytest.raises(AttributeError):
        receiver.config = dataclasses.replace(config, pss_threshold=.23)


def test_public_cli_decodes_real_iq_file_and_reports_summary(tmp_path, capsys):
    from signal_processing.lte.cli import main
    path = tmp_path/"sync.cf32"
    iq = waveform(71, frames=2, snr_db=10)
    iq.tofile(path)
    assert main(["--input", str(path), "--require-lock", "--enforce-acquisition"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    confirmed = [r for r in records if r["type"] == "observation" and r["state"] == "SYNCHRONIZED"]
    assert confirmed and all(r["pci"] == 71 for r in confirmed)
    assert records[-1]["type"] == "summary" and records[-1]["acquisition_passed"]


def test_public_cli_no_lock_exit_is_distinct_from_source_error(tmp_path, capsys):
    from signal_processing.lte.cli import main
    path = tmp_path/"silence.cf32"
    np.zeros(20000, np.complex64).tofile(path)
    assert main(["--input", str(path), "--require-lock"]) == 2
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[-1]["acquisition_passed"] and records[-1]["error"] is None


@pytest.mark.parametrize("cfo", [-22300., 0., 13200., 30000.])
def test_confirmed_tracking_matches_blind_reference_statistic_and_timing(cfo):
    iq = waveform(214, frames=5, cfo=cfo, snr_db=0, seed=2233)
    tracked, actual = run(iq)
    _, expected = run(iq, config=LTESyncConfig(tracking=False))
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert (a.pci, a.state, a.processing_pss_sample) == (b.pci, b.state, b.processing_pss_sample)
        np.testing.assert_allclose([a.pss_metric, a.sss_metric, a.frequency_offset_hz],
                                   [b.pss_metric, b.sss_metric, b.frequency_offset_hz], rtol=1e-12, atol=1e-10)
    snap = tracked.snapshot()
    assert snap["tracking_searches"] > snap["blind_searches"]


def test_unrelated_ofdm_is_not_confirmed_as_lte():
    receiver, observations = run(waveform(frames=4, with_sync=False, snr_db=5))
    assert not any(o.state == "SYNCHRONIZED" for o in observations)
    assert receiver.snapshot()["pci"] is None


def test_earlier_large_impulse_does_not_erase_small_real_sync_windows():
    iq = (waveform(91, frames=3, onset=1000, snr_db=10)*1e-8).astype(np.complex64)
    iq[0] = 1
    receiver, observations = run(iq)
    assert any(o.state == "SYNCHRONIZED" and o.pci == 91 for o in observations)
    assert receiver.snapshot()["pci"] == 91


def test_zero_cfo_with_tiny_step_remains_bounded_and_huge_search_rejects_safely():
    config = LTESyncConfig(max_frequency_offset_hz=0., frequency_step_hz=1e-308)
    receiver = LTESynchronizer(StreamInfo("s", "r", 2e6), config)
    assert receiver.snapshot()["frequency_hypotheses"] == 1
    with pytest.raises(ValueError):
        LTESyncConfig(max_frequency_offset_hz=100000., frequency_step_hz=1e-308)
    default = LTESynchronizer(StreamInfo("s", "r", 2e6))
    actual = np.array([f for nid, f in default._branches if nid == 0])
    np.testing.assert_array_equal(actual, np.arange(-30000., 30000.+1e-8, 7500.))


@pytest.mark.parametrize("kwargs", [{"input_sample_rate_hz": 1e6}, {"frequency_step_hz": 0},
    {"pss_threshold": float("nan")}, {"confirmations": True}, {"max_candidates": 0},
    {"max_frequency_offset_hz": 100000, "frequency_step_hz": 1}])
def test_invalid_search_configuration_rejects(kwargs):
    with pytest.raises(ValueError):
        LTESyncConfig(**kwargs)
