"""Repeated-cell-search behavior on actual independently encoded LTE IQ."""
import dataclasses
import json

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo, StreamConsistencyError
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig, LTEPBCHObservation
from tests.lte_pbch_fixture import fixture, waveform


def receive(iq, *, rate=2e6, size=4093, origin=0, low=True):
    stream = StreamInfo("weak", "independent", rate, 2.14e9)
    config = LTESyncConfig.low_snr(rate) if low else LTESyncConfig(input_sample_rate_hz=rate)
    receiver = LTEBroadcastReceiver(stream, config)
    mibs, observations = [], []
    for seq, start in enumerate(range(0, len(iq), size)):
        batch = receiver.process_block(IQBlock(stream, seq, start+origin, None, iq[start:start+size]))
        mibs.extend(batch.mibs)
        observations.extend(batch.synchronization)
    batch = receiver.flush()
    return receiver, mibs+list(batch.mibs), observations+list(batch.synchronization)


@pytest.mark.parametrize("ports", [1, 2, 4])
@pytest.mark.parametrize("rate", [1_920_000., 2_000_000.])
def test_weak_real_iq_recovers_exact_mib_without_truth_configuration(ports, rate):
    iq = waveform(42, ports=ports, rate=rate, snr_db=-5, cfo=12345, seed=771)
    receiver, mibs, observations = receive(iq, rate=rate, origin=10**12)
    expected, _ = fixture(4, ports)
    assert mibs and any(o.pci == 42 and o.state == "SYNCHRONIZED" for o in observations)
    frames = set()
    for result in mibs:
        frame = round(float(result.input_frame_start-(10**12+431))/(rate/100))
        assert frame not in frames
        frames.add(frame)
        assert result.pci == 42 and result.antenna_ports == ports
        assert result.mib.payload_bits == tuple(map(int, expected))
        assert result.frame_quarter == frame
        assert abs(float(result.input_frame_start-(10**12+431+frame*rate/100))) <= 2
        assert LTEPBCHObservation.from_dict(json.loads(json.dumps(result.to_dict()))) == result
    state = receiver.snapshot()
    assert state["retained_processing_samples"] <= 29100
    assert state["retained_broadcast_samples"] <= 62400
    assert state["retained_bch_frames"] <= 4


def test_zadoff_chu_frequency_delay_alias_cannot_suppress_real_sss_hypothesis():
    iq = waveform(92, index=27, onset=142, cfo=8397.858297079434, seed=89564698, snr_db=0)
    _, mibs, observations = receive(iq)
    assert len(mibs) == 4
    assert all(o.pci == 92 for o in observations)
    assert [m.frame_quarter for m in mibs] == [0, 1, 2, 3]
    assert all(m.mib.payload_bits == tuple(map(int, fixture(27, 1)[0])) for m in mibs)


def test_dwell_is_explicit_and_no_future_or_padded_repetition_is_used():
    iq = waveform(rate=1_920_000., frames=1, snr_db=10)
    receiver, mibs, observations = receive(iq, rate=1_920_000.)
    assert not mibs and not observations
    assert receiver.snapshot()["processed_samples"] == len(iq)
    assert receiver.flush().mibs == ()


def test_low_snr_profile_partition_origin_and_preparation_horizons():
    iq = waveform(251, ports=2, snr_db=-5, cfo=-7500, seed=2233)
    _, expected, expected_sync = receive(iq, size=16384)
    assert expected
    for size in (1, 137, 997, 29000):
        _, actual, actual_sync = receive(iq, size=size, origin=10**12)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert dataclasses.replace(a, input_frame_start=a.input_frame_start-10**12,
                                       prepared_input_end=b.prepared_input_end) == b
        assert [(o.pci, o.processing_pss_sample, o.state) for o in actual_sync] == [
            (o.pci, o.processing_pss_sample, o.state) for o in expected_sync]


@pytest.mark.parametrize("unknown", [False, True])
def test_gap_clears_repetition_history_and_corrected_retry_is_atomic(unknown):
    iq = waveform(rate=1_920_000., snr_db=-5)
    stream = StreamInfo("weak", "independent", 1_920_000., 2.14e9)
    receiver = LTEBroadcastReceiver(stream, LTESyncConfig.low_snr(1_920_000.))
    assert not receiver.process_block(IQBlock(stream, 0, 10**12, None, iq[:25000])).mibs
    before = receiver.snapshot()
    bad_stream = dataclasses.replace(stream, sample_rate_hz=1_920_000.+.0005)
    with pytest.raises(StreamConsistencyError):
        receiver.process_block(IQBlock(bad_stream, 1, 10**12+25000, None, iq[25000:30000]))
    assert receiver.snapshot() == before
    gap = 0 if unknown else 32
    receiver.process_block(IQBlock(stream, 1, 10**12+25000+gap, None, iq[:25000],
                                  dropped_samples_before=gap, overflow=unknown))
    batch = receiver.flush()
    assert not batch.mibs and not batch.synchronization
    receiver.reset()
    recovered = receiver.process_block(IQBlock(stream, 0, 10**12, None, iq)).mibs+receiver.flush().mibs
    assert recovered and all(m.epoch == 2 for m in recovered)


@pytest.mark.parametrize("kind", ["zero", "noise", "tone", "unrelated_ofdm"])
def test_integrated_negative_controls_do_not_acquire_cell_or_mib(kind):
    from tests.lte_fixture import waveform as sync_waveform
    rng = np.random.default_rng(1042)
    if kind == "zero":
        iq = np.zeros(90000, np.complex64)
    elif kind == "noise":
        iq = (rng.normal(size=90000)+1j*rng.normal(size=90000)).astype(np.complex64)
    elif kind == "tone":
        iq = np.exp(2j*np.pi*.111*np.arange(90000)).astype(np.complex64)
    else:
        iq = sync_waveform(frames=4, with_sync=False)
    _, mibs, observations = receive(iq)
    assert not mibs and not any(o.state == "SYNCHRONIZED" for o in observations)


def test_integration_controls_are_immutable_bounded_and_identified():
    baseline = LTESyncConfig()
    assert baseline.integration_half_frames == 1
    assert baseline.profile_id == "lte_sync_6b9742e3cd19c17a028c"
    assert LTESyncConfig.low_snr().profile_id != baseline.profile_id
    for invalid in (True, 0, 3, 4., 9):
        with pytest.raises(ValueError):
            LTESyncConfig(integration_half_frames=invalid)
    with pytest.raises(dataclasses.FrozenInstanceError):
        baseline.integration_half_frames = 4


def test_new_integration_control_preserves_legacy_positional_constructor():
    default = LTESyncConfig()
    old_arguments = [getattr(default, f.name) for f in dataclasses.fields(default)
                     if f.name != "integration_half_frames"]
    assert LTESyncConfig(*old_arguments) == default
    assert LTESyncConfig(*old_arguments, integration_half_frames=4).integration_half_frames == 4


def test_integrated_pss_metric_matches_independent_direct_window_sum():
    from pathlib import Path
    iq = waveform(174, rate=1_920_000., snr_db=0, cfo=0)
    _, _, observations = receive(iq, rate=1_920_000.)
    assert observations
    with np.load(Path(__file__).with_name("fixtures")/"lte_sync_sequences.npz", allow_pickle=False) as ref:
        bins = np.r_[np.arange(-31, 0), np.arange(1, 32)] % 128
        grid = np.zeros(128, complex)
        grid[bins] = ref["pss"][174 % 3]
        template = np.fft.ifft(grid)*np.sqrt(128)
    for observation in observations:
        windows = [iq[observation.processing_pss_sample-j*9600:observation.processing_pss_sample-j*9600+128].astype(complex)
                   for j in range(4)]
        corrected = template*np.exp(2j*np.pi*observation.frequency_offset_hz*np.arange(128)/1_920_000.)
        numerator = sum(abs(sum(a*b.conjugate() for a, b in zip(y, corrected)))**2 for y in windows)
        energy = sum(sum(v.real*v.real+v.imag*v.imag for v in y) for y in windows)
        assert observation.pss_metric == pytest.approx(numerator/(62*energy), rel=2e-13, abs=1e-15)


def test_weak_profile_cli_is_explicit_and_preserves_default(tmp_path, capsys):
    from signal_processing.lte.cli import main
    path = tmp_path/"weak.cf32"
    waveform(snr_db=-5, seed=771).tofile(path)
    assert main(["--input", str(path), "--require-mib", "--profile", "low_snr"]) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["decoded_mibs"] > 0


def test_long_fixture_encoder_is_independent_and_matches_all_external_c_vectors():
    from tests.lte_pbch_fixture import PATH, encode_payload
    with np.load(PATH, allow_pickle=False) as reference:
        for bits, ports, matched in zip(reference["bits"], reference["ports"], reference["matched"]):
            np.testing.assert_array_equal(encode_payload(bits[:24], int(ports)), matched)


@pytest.mark.parametrize("quarter", [1, 2, 3])
def test_arbitrary_bch_phase_and_advancing_sfn_across_longer_recording(quarter):
    from tests.lte_pbch_fixture import frame_fixture
    iq = waveform(174, ports=2, first_quarter=quarter, frames=8, snr_db=-5, cfo=-12345, seed=2233)
    _, mibs, _ = receive(iq)
    assert mibs
    frames = set()
    for result in mibs:
        frame = round(float(result.input_frame_start-431)/20000)
        assert frame not in frames
        frames.add(frame)
        assert result.frame_quarter == (frame+quarter) % 4
        assert result.mib.payload_bits == tuple(map(int, frame_fixture(4, 2, frame, quarter)[0]))
