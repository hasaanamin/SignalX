"""Real replay and export checks using a small positioned dataset fixture."""

import csv
import hashlib
import json
import threading

import numpy as np
import pytest

from signal_processing.ui.hdf5_runner import RunCancelled
from signal_processing.ui.smoke_runner import (
    SmokeConfig,
    compact_smoke_result,
    comparison_rows,
    export_smoke_csv,
    inspect_smoke_dataset,
    match_activity,
    run_smoke_record,
)


@pytest.fixture
def record(tmp_path):
    n = 4096
    rng = np.random.default_rng(47)
    iq = ((rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(1e-4 / 2)).astype(np.complex64)
    iq[1000:2500] += (0.05 * np.exp(2j * np.pi * 0.1 * np.arange(1500))).astype(np.complex64)
    np.save(tmp_path / "iq.npy", iq)
    sha = hashlib.sha256(iq.tobytes()).hexdigest()
    emitter = dict(
        emitter_id="r:e0",
        modulation="tone",
        start_sample=1000,
        end_sample=2500,
        requested_snr_fullband_db=14.0,
        center_frequency_offset_hz=100000.0,
        nominal_bandwidth_hz=1.0,
    )
    a = dict(
        record_id="r",
        scenario="single",
        num_samples=n,
        sample_rate_hz=1e6,
        receiver_center_frequency_hz=915e6,
        iq_sha256=sha,
        emitters=[emitter],
        background={"model": "white", "expected_fullband_noise_power_linear": 999.0},
        calibration_prefix=dict(
            start_sample=0,
            end_sample=512,
            settling_samples=64,
            training_samples=448,
            noise_only=True,
        ),
        fullband_activity_intervals=[
            dict(start_sample=1000, end_sample=2500, contributing_emitter_ids=["r:e0"])
        ],
    )
    (tmp_path / "annotations.json").write_text(json.dumps(a))
    truth = np.zeros(n, dtype=bool)
    truth[1000:2500] = True
    np.savez(
        tmp_path / "truth_masks.npz", any_signal=truth, active_emitter_count=truth.astype(np.int16)
    )
    manifest = dict(
        record_id="r",
        scenario="single",
        iq_file="iq.npy",
        annotations_file="annotations.json",
        masks_file="truth_masks.npz",
        iq_sha256=sha,
    )
    (tmp_path / "manifest.jsonl").write_text(json.dumps(manifest))
    (tmp_path / "dataset.json").write_text(json.dumps({"status": "COMPLETE"}))
    return inspect_smoke_dataset(tmp_path)[0]


def test_replay_preserves_iq_and_measures_authorized_prefix(record):
    r = run_smoke_record(record)
    iq = np.load(record.iq_path)
    expected = np.mean(np.abs(iq[64:512].astype(np.complex128)) ** 2)
    assert r.metrics["calibrated_noise_power"] == pytest.approx(expected, rel=1e-12)
    assert r.metrics["matched"] == 1
    assert r.metrics["exact_capture_iq"]
    assert len(r.captures[0].features.features) == 56
    assert r.captures[0].event.pre_trigger_samples >= 256
    assert not r.display_samples.flags.writeable


def test_online_replay_does_not_read_prefix_snr_or_ideal_power(record):
    # A short record deliberately remains in STARTING. All truths are scored,
    # including those during startup; missing a reference is not a fake pass.
    config = SmokeConfig(detector_profile="online_ar1", enable_features=False)
    first = run_smoke_record(record, config)
    ann = json.loads(record.annotations_path.read_text())
    ann.pop("calibration_prefix")
    ann["sweep_snr_db"] = 999
    ann["background"]["expected_fullband_noise_power_linear"] = 1e100
    record.annotations_path.write_text(json.dumps(ann))
    second = run_smoke_record(record, config)
    assert first.metrics["online_noise"] == second.metrics["online_noise"]
    assert first.captures == second.captures == ()
    assert second.metrics["evaluation_start"] == 0
    assert second.metrics["calibration_training_samples"] == 0
    assert second.metrics["misses"] == 1


def test_assignment_maximizes_cardinality_before_iou():
    matches = match_activity([(0, 10), (10, 20)], [(0, 20), (0, 4)], 0.2)
    assert {(i, j) for i, j, _ in matches} == {(0, 1), (1, 0)}
    assert match_activity([(0, 10)], [(9, 20)], 0.2) == ()
    assert match_activity([], [(0, 10)], 0.2) == ()


def test_missed_truth_and_empty_outcome_export(record, tmp_path):
    r = run_smoke_record(record, SmokeConfig(window_size=4096, hop_size=64, enable_features=False))
    assert len(r.captures) == 0  # No complete 4096-sample decision after calibration.
    path = tmp_path / "miss.csv"
    export_smoke_csv(path, [r])
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["row_type"] for r in rows] == ["record_summary", "emitter_truth", "activity_truth"]
    assert rows[-1]["match_status"] == "missed"
    assert rows[1]["truth_emitter_id"] == "r:e0"
    assert rows[-1]["event_id"] == ""


def test_results_export_contains_validity_quality_and_identity(record, tmp_path):
    r = run_smoke_record(record)
    rows = comparison_rows(r, include="results")
    assert not any(row["row_type"] == "emitter_truth" for row in rows)
    event = next(row for row in rows if row["row_type"] == "detected_event")
    assert event["event_id"] == r.captures[0].event_id
    assert "quality_json" in event and "extraction_policy_json" in event
    assert sum(key.endswith(".valid") for key in event) == 56
    path = tmp_path / "combined.csv"
    export_smoke_csv(path, [r])
    with path.open(encoding="utf-8-sig", newline="") as f:
        read = list(csv.DictReader(f))
    assert sum(row["row_type"] == "detected_event" for row in read) == 1
    assert (
        next(row for row in read if row["row_type"] == "activity_truth")["event_id"]
        == event["event_id"]
    )


def test_cancel_is_observed_before_admission(record):
    stop = threading.Event()
    stop.set()
    with pytest.raises(RunCancelled):
        run_smoke_record(record, cancel_event=stop)


def test_reject_corrupt_iq(record):
    x = np.load(record.iq_path)
    x[0] += 1
    np.save(record.iq_path, x)
    with pytest.raises(ValueError, match="checksum"):
        run_smoke_record(record)


def test_reject_contradictory_masks(record):
    np.savez(
        record.masks_path,
        any_signal=np.zeros(4096, dtype=bool),
        active_emitter_count=np.zeros(4096),
    )
    with pytest.raises(ValueError, match="mask"):
        run_smoke_record(record)


def test_partition_invariant_replay(record):
    a = run_smoke_record(record, SmokeConfig(block_size=1024, enable_features=False))
    b = run_smoke_record(record, SmokeConfig(block_size=137, enable_features=False))
    assert [(c.event.detected_start_sample, c.event.detected_end_sample) for c in a.captures] == [
        (c.event.detected_start_sample, c.event.detected_end_sample) for c in b.captures
    ]
    for ca, cb in zip(a.captures, b.captures):
        np.testing.assert_array_equal(ca.event.iq, cb.event.iq)


def test_compact_results_drop_iq_without_changing_exports(record, tmp_path):
    import gc
    import weakref

    full = run_smoke_record(record)
    iq_ref = weakref.ref(full.captures[0].event.iq)
    compact = compact_smoke_result(full)
    assert not hasattr(compact.captures[0].event, "iq")
    for mode in ("combined", "truth", "results"):
        assert comparison_rows(full, mode) == comparison_rows(compact, mode)
    del full
    gc.collect()
    assert iq_ref() is None
    export_smoke_csv(tmp_path / "compact.csv", iter([compact]))
    assert compact.metrics["exact_capture_iq"]


def test_ar1_is_explicitly_uncertified_stress_comparison(record):
    # This checks evaluator labelling; the complete dataset supplies real AR1 IQ.
    a = json.loads(record.annotations_path.read_text())
    a["background"]["model"] = "ar1"
    a["background"]["ar1_rho"] = 0.8
    record.annotations_path.write_text(json.dumps(a))
    result = run_smoke_record(record)
    assert result.metrics["noise_model"] == "ar1"
    assert result.metrics["threshold_assumptions_match"] is False
    assert "MODEL MISMATCH" in result.metrics["threshold_support"]
    assert all("MODEL MISMATCH" in row["threshold_support"] for row in comparison_rows(result))
    assert result.metrics["exact_capture_iq"]


def test_480_record_manifest_and_count_guard(record):
    root = record.iq_path.parent
    base = json.loads((root / "manifest.jsonl").read_text())
    rows = [{**base, "record_id": f"r{i}"} for i in range(480)]
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (root / "dataset.json").write_text(
        json.dumps({"status": "COMPLETE", "records_expected": 480, "records_completed": 480})
    )
    assert len(inspect_smoke_dataset(root)) == 480
    (root / "dataset.json").write_text(json.dumps({"status": "COMPLETE", "records_expected": 481}))
    with pytest.raises(ValueError, match="manifest count"):
        inspect_smoke_dataset(root)
