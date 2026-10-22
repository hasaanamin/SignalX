"""Public PBCH/MIB and independent external-codeword regression checks."""
import binascii
import dataclasses
from fractions import Fraction
import json

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo, StreamConsistencyError
from signal_processing.lte import LTEBroadcastReceiver, LTEMIB, LTEPBCHObservation, PBCHDecoder
from signal_processing.lte.bch import crc16, decode_tail_biting, rate_match_indices
from tests.lte_pbch_fixture import PATH, fixture, slot_grids, waveform


def run(iq, *, rate=2e6, size=4093, origin=0):
    stream = StreamInfo("lte", "reference", rate, 2.14e9)
    receiver = LTEBroadcastReceiver(stream)
    mibs, sync = [], []
    for sequence, start in enumerate(range(0, len(iq), size)):
        batch = receiver.process_block(IQBlock(stream, sequence, origin+start, None, iq[start:start+size]))
        mibs.extend(batch.mibs)
        sync.extend(batch.synchronization)
    batch = receiver.flush()
    mibs.extend(batch.mibs)
    sync.extend(batch.synchronization)
    return receiver, mibs, sync


def test_original_external_encoder_vectors_and_crc_oracle():
    with np.load(PATH, allow_pickle=False) as reference:
        assert len(reference["bits"]) == 90
        for bits, encoded, matched in zip(reference["bits"], reference["encoded"], reference["matched"]):
            np.testing.assert_array_equal(decode_tail_biting(1-2*encoded.astype(float)), bits)
            np.testing.assert_array_equal(encoded[rate_match_indices()], matched)
            checksum = binascii.crc_hqx(np.packbits(bits[:24]).tobytes(), 0)
            assert crc16(bits[:24]) == checksum
            crc = [(checksum >> k) & 1 for k in range(15, -1, -1)]
            assert crc16(list(bits[:24])+crc) == 0


@pytest.mark.parametrize("ports", [1, 2, 4])
@pytest.mark.parametrize("quarter", [0, 1, 2, 3])
def test_public_grid_decoder_all_masks_quarters_and_mib_fields(ports, quarter):
    payload, grids = slot_grids(503, ports, index=5, quarter=quarter)
    gains = [1, .6+.3j, -.2+.7j, .5-.4j][:ports]
    grid = sum(g*grids[p, :5] for p, g in enumerate(gains))
    decoded = PBCHDecoder(503).decode(grid, 0)
    assert decoded is not None
    mib, actual_ports, actual_quarter, count = decoded
    assert mib.payload_bits == tuple(map(int, payload))
    assert (actual_ports, actual_quarter, count) == (ports, quarter, 1)
    assert mib.downlink_resource_blocks == 100
    assert mib.downlink_bandwidth_hz == 20_000_000


@pytest.mark.parametrize("rate", [1_920_000., 2_000_000.])
@pytest.mark.parametrize("ports", [1, 2, 4])
def test_real_positioned_iq_sync_to_crc_valid_mib(rate, ports):
    iq = waveform(174, rate=rate, ports=ports, cfo=-12_345, snr_db=15)
    receiver, mibs, sync = run(iq, rate=rate, origin=10**12)
    expected, _ = fixture(4, ports)
    assert any(s.state == "SYNCHRONIZED" for s in sync)
    assert len(mibs) == 4
    for frame, result in enumerate(mibs):
        assert result.mib.payload_bits == tuple(map(int, expected))
        assert result.antenna_ports == ports and result.pci == 174
        assert result.frame_quarter == frame
        assert result.system_frame_number == 4*result.mib.system_frame_number_msb+frame
        assert abs(float(result.input_frame_start-(10**12+431+int(rate/100)*frame))) < 2
        assert LTEPBCHObservation.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False))) == result
    assert receiver.snapshot()["pbch_decoded_frames"] == 4
    assert receiver.flush().mibs == ()


def test_partition_invariance_and_exact_rational_coordinates():
    iq = waveform(251, frames=2, cfo=7500, ports=2)
    _, expected, _ = run(iq, size=16384)
    assert expected
    for size in (1, 137, 997, 29000):
        _, actual, _ = run(iq, size=size, origin=10**12)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            # Prepared/received horizon describes the transport call, not the
            # logical FFT window: it legitimately varies with partition size.
            assert a.prepared_input_end > a.input_frame_start+1900
            assert dataclasses.replace(a, input_frame_start=a.input_frame_start-10**12,
                                       prepared_input_end=b.prepared_input_end) == b


def test_zero_grid_never_creates_crc_valid_zero_mib_and_rejection_is_atomic():
    decoder = PBCHDecoder(42)
    assert decoder.decode(np.zeros((5, 72), complex), 0) is None
    for bad in (np.zeros((4, 72), complex), np.ones((5, 72)), np.full((5, 72), complex(np.nan))):
        with pytest.raises(ValueError):
            decoder.decode(bad, 19200)
        assert decoder.retained_frames == decoder.attempted_frames == 1


def test_stream_metadata_and_timestamp_rejection_preserves_pending_broadcast_iq():
    iq = waveform(frames=2)
    stream = StreamInfo("lte", "reference", 2e6, 2.14e9)
    receiver = LTEBroadcastReceiver(stream)
    receiver.process_block(IQBlock(stream, 0, 0, 1_000_000_000, iq[:4093]))
    before = receiver.snapshot()
    good = IQBlock(stream, 1, 4093, 1_002_046_500, iq[4093:20000])
    for bad in (dataclasses.replace(good, timestamp_ns=2_002_046_500),
                dataclasses.replace(good, stream=dataclasses.replace(stream, sample_rate_hz=2e6+.0005))):
        with pytest.raises(StreamConsistencyError):
            receiver.process_block(bad)
        assert receiver.snapshot() == before
    assert receiver.process_block(good).mibs


def test_mib_immutability_and_serialization_rejects_changed_derived_fields():
    payload, _ = fixture()
    mib = LTEMIB(list(map(int, payload)))
    assert isinstance(mib.payload_bits, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        mib.payload_bits = (0,)*24
    result = LTEPBCHObservation(mib, 42, 1, 0, 1, 0, "s", "p", 0, Fraction(0), 0., 2000)
    record = result.to_dict()
    record["mib"]["downlink_resource_blocks"] = 6
    with pytest.raises(ValueError):
        LTEPBCHObservation.from_dict(record)


def test_four_frame_combining_recovers_genuine_noisy_external_codeword():
    rng = np.random.default_rng(1223)
    combined = PBCHDecoder(42)
    for quarter in range(4):
        payload, grids = slot_grids(42, 1, quarter=quarter)
        grid = grids[0, :5]
        noisy = grid+2*(rng.normal(size=grid.shape)+1j*rng.normal(size=grid.shape))
        assert PBCHDecoder(42).decode(noisy, quarter*19200) is None
        result = combined.decode(noisy, quarter*19200)
        if quarter < 3:
            assert result is None
        else:
            assert result == (LTEMIB(tuple(map(int, payload))), 1, 3, 4)
        assert combined.retained_frames <= 4


@pytest.mark.parametrize("pci", [0, 1, 2, 5, 42, 503])
def test_all_six_bandwidth_codes_crs_shifts_and_raw_extension_bits(pci):
    for index, rb in enumerate((6, 15, 25, 50, 75, 100)):
        payload, grids = slot_grids(pci, 1, index=index, quarter=index % 4)
        result = PBCHDecoder(pci).decode(grids[0, :5], 0)
        assert result is not None
        assert result[0].payload_bits == tuple(map(int, payload))
        assert result[0].downlink_resource_blocks == rb


@pytest.mark.parametrize("scale", [1e-20, 1., 1e20, 1e-200, 1e200])
def test_grid_decoding_scale_invariance_without_channel_floor(scale):
    payload, grids = slot_grids(42, 2)
    grid = (grids[0, :5]+(.6+.3j)*grids[1, :5])*scale
    result = PBCHDecoder(42).decode(grid, 0)
    assert result is not None and result[0].payload_bits == tuple(map(int, payload))


@pytest.mark.parametrize("unknown", [False, True])
def test_gap_finalizes_real_old_prefix_and_never_combines_epochs(unknown):
    iq = waveform(frames=2, rate=1_920_000.)
    stream = StreamInfo("lte", "reference", 1_920_000., 2.14e9)
    receiver = LTEBroadcastReceiver(stream)
    # Confirmation and final search segment remain pending until discontinuity.
    prefix = iq[:11500]
    first = receiver.process_block(IQBlock(stream, 0, 10**12, None, prefix))
    assert not first.mibs
    gap = 0 if unknown else 32
    second = receiver.process_block(IQBlock(stream, 1, 10**12+11500+gap, None,
        np.zeros(300, np.complex64), dropped_samples_before=gap, overflow=unknown))
    assert len(second.mibs) == 1
    assert second.mibs[0].epoch == 0
    assert abs(float(second.mibs[0].input_frame_start-(10**12+431))) < 2
    assert receiver.snapshot()["epoch"] == 1
    assert receiver.snapshot()["retained_bch_frames"] == 0
    assert receiver.flush().mibs == ()


def test_later_fault_preserves_genuine_completed_mib_and_requires_reset(monkeypatch):
    iq = waveform(frames=3, rate=1_920_000.)
    stream = StreamInfo("lte", "reference", 1_920_000., 2.14e9)
    receiver = LTEBroadcastReceiver(stream)
    original = receiver._search

    def fail_later(horizon, emitted, final=False):
        if horizon > 20000:
            raise ArithmeticError("injected later numerical failure")
        return original(horizon, emitted, final)

    monkeypatch.setattr(receiver, "_search", fail_later)
    with pytest.raises(ArithmeticError) as caught:
        receiver.process_block(IQBlock(stream, 0, 0, None, iq))
    assert len(caught.value.completed_mibs) == 1
    assert caught.value.completed_mibs[0].mib.payload_bits == tuple(map(int, fixture()[0]))
    assert receiver.snapshot()["state"] == "FAULTED"
    with pytest.raises(RuntimeError):
        receiver.flush()
    monkeypatch.setattr(receiver, "_search", original)
    receiver.reset()
    assert receiver.snapshot()["retained_broadcast_samples"] == 0
    result = receiver.process_block(IQBlock(stream, 0, 0, None, iq))
    assert result.mibs and all(m.epoch == 1 for m in result.mibs)


def test_bad_pbch_with_real_pss_sss_does_not_return_mib():
    _, mibs, sync = run(waveform(corrupt_pbch=True))
    assert any(s.state == "SYNCHRONIZED" for s in sync)
    assert not mibs


def test_repeated_frames_remain_bounded_and_reset_replays_exactly():
    iq = waveform(frames=12, rate=1_920_000.)
    receiver, mibs, _ = run(iq, rate=1_920_000.)
    assert len(mibs) == 12
    state = receiver.snapshot()
    assert state["retained_broadcast_samples"] <= 24000
    assert state["retained_bch_frames"] <= 4
    assert state["pending_broadcast_frames"] <= 1
    receiver.reset()
    replay = receiver.process_block(IQBlock(StreamInfo("lte", "reference", 1_920_000., 2.14e9), 0, 0, None, iq))
    replay_mibs = replay.mibs+receiver.flush().mibs
    assert len(replay_mibs) == len(mibs)
    assert [dataclasses.replace(a, epoch=0, prepared_input_end=b.prepared_input_end)
            for a, b in zip(replay_mibs, mibs)] == mibs


def test_source_provenance_survives_mib_delivery():
    from signal_processing.sources import SourceEnvelope, SourceMetadata
    iq = waveform(frames=2)
    stream = StreamInfo("lte", "reference", 2e6, 2.14e9)
    receiver = LTEBroadcastReceiver(stream)
    metadata = SourceMetadata(source_epoch=42, configuration_epoch=3, gain_mode="manual", gain_db=20)
    envelope = SourceEnvelope(IQBlock(stream, 0, 1000, None, iq), metadata)
    mibs = receiver.process_envelope(envelope).mibs
    assert mibs and all(m.source_metadata == metadata and m.source_configuration_id == envelope.configuration_id for m in mibs)


def test_cli_returns_real_mib_or_explicit_no_mib_exit(tmp_path, capsys):
    from signal_processing.lte.cli import main
    path = tmp_path/"input.cf32"
    waveform(frames=2).tofile(path)
    assert main(["--input", str(path), "--require-mib"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(line["type"] == "mib" and line["crc_valid"] for line in lines)
    assert lines[-1]["decoded_mibs"] == 2
    waveform(frames=2, corrupt_pbch=True).tofile(path)
    assert main(["--input", str(path), "--require-mib"]) == 2
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["decoded_mibs"] == 0


@pytest.mark.parametrize("pci,index,onset,cfo,seed", [
    (474, 23, 413, 10880.90349060153, 998557235),
    (503, 16, 888, -1323.8206532363038, 884920303)])
def test_noisy_half_frame_timing_refinements_do_not_duplicate_pbch(pci, index, onset, cfo, seed):
    # Exact two independently generated cases that exposed repeated decoding
    # of one physical frame at adjacent integer timing estimates.
    iq = waveform(pci, index=index, onset=onset, cfo=cfo, seed=seed, ports=4, snr_db=0)
    _, mibs, _ = run(iq)
    payload, _ = fixture(index, 4)
    assert len(mibs) == 4
    assert [m.frame_quarter for m in mibs] == [0, 1, 2, 3]
    assert all(m.mib.payload_bits == tuple(map(int, payload)) for m in mibs)


def test_output_collection_is_inside_single_receiver_lifecycle_gate(monkeypatch):
    import threading
    from signal_processing.sources import SourceBusy
    stream = StreamInfo("lte", "reference", 2e6, 2.14e9)
    receiver = LTEBroadcastReceiver(stream)
    iq = waveform(frames=2)
    admitted, release = threading.Event(), threading.Event()
    original = receiver._finish_output
    results, errors = [], []

    def held_finish(sync):
        admitted.set()
        assert release.wait(5)
        return original(sync)

    monkeypatch.setattr(receiver, "_finish_output", held_finish)

    def submit():
        try:
            results.append(receiver.process_block(IQBlock(stream, 0, 0, None, iq)))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=submit)
    thread.start()
    try:
        assert admitted.wait(5)
        with pytest.raises(SourceBusy):
            receiver.reset()
        with pytest.raises(SourceBusy):
            receiver.flush()
        with pytest.raises(SourceBusy):
            receiver.process_block(IQBlock(stream, 1, len(iq), None, iq[:64]))
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert len(results) == 1 and len(results[0].mibs) == 2
    assert receiver.flush().mibs == ()


def test_discontinuous_public_grid_input_cannot_reuse_old_frame_evidence():
    decoder = PBCHDecoder(42)
    _, grids = slot_grids(42, 1)
    assert decoder.decode(grids[0, :5], 0) is not None
    assert decoder.decode(grids[0, :5], 100000) is not None
    assert decoder.retained_frames == 1
    before = decoder.attempted_frames
    with pytest.raises(ValueError):
        decoder.decode(grids[0, :5], 99999)
    assert decoder.attempted_frames == before and decoder.retained_frames == 1
