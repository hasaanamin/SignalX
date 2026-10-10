"""Public regression tests for the labelled HDF5 pipeline test bench."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from signal_processing.ui.hdf5_runner import (
    HarnessConfig,
    RunCancelled,
    inspect_hdf5_dataset,
    run_hdf5_pipeline,
)


def _dataset(path, frames: int = 8, frame_length: int = 1024) -> None:
    time = np.arange(frame_length, dtype=np.float64)
    x = np.empty((frames, frame_length, 2), dtype=np.float32)
    for index in range(frames):
        frequency = 0.04 + 0.015 * (index % 4)
        tone = np.exp(2j * np.pi * frequency * time).astype(np.complex64)
        x[index, :, 0] = tone.real
        x[index, :, 1] = tone.imag
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=x)
        handle.create_dataset("y_mod", data=np.arange(frames, dtype=np.int16) % 2)
        handle.create_dataset("y_snr", data=np.where(np.arange(frames) % 2, 30, 20).astype(np.int16))
        handle.create_dataset("y_chan", data=(np.arange(frames) % 2).astype(np.int8))
        handle.attrs["mod2id_json"] = json.dumps({"BPSK": 0, "QPSK": 1})
        handle.attrs["channel_meaning"] = "0=clean, 1=multipath(ref)"
        handle.attrs["frame_len"] = frame_length
        handle.attrs["source"] = "unit-test"
        handle.attrs["subset_name"] = "test"


def test_inspection_reports_schema_and_immutable_counts(tmp_path) -> None:
    path = tmp_path / "test.h5"
    _dataset(path)
    summary = inspect_hdf5_dataset(path)
    assert summary.frame_count == 8
    assert summary.frame_length == 1024
    assert summary.modulation_counts == {"BPSK": 4, "QPSK": 4}
    assert summary.snr_counts == {20: 4, 30: 4}
    with pytest.raises(TypeError):
        summary.snr_counts[20] = 0


def test_real_pipeline_detects_frames_and_extracts_default_schema(tmp_path) -> None:
    path = tmp_path / "test.h5"
    _dataset(path)
    result = run_hdf5_pipeline(
        path,
        HarnessConfig(frame_count=4, normalize_frames=True, enable_features=True),
    )
    assert result.selected_indices == (0, 1, 2, 3)
    assert len(result.captures) == 4
    assert result.matched_truths == 4
    assert result.missed_truths == 0
    assert result.extra_events == 0
    assert result.detection_rate == 1.0
    assert all(sidecar.feature_status == "ok" for sidecar in result.captures)
    assert all(sidecar.features is not None for sidecar in result.captures)
    assert all(len(sidecar.features.features) == 56 for sidecar in result.captures)
    assert all(match.truth_index is not None and match.overlap_samples > 0 for match in result.matches)
    assert result.display_sample_indices.flags.writeable is False
    assert result.display_magnitudes.flags.writeable is False


def test_filters_and_random_selection_are_reproducible(tmp_path) -> None:
    path = tmp_path / "test.h5"
    _dataset(path, frames=12)
    config = HarnessConfig(
        frame_count=3,
        modulation="QPSK",
        snr_db=30,
        channel=1,
        random_selection=True,
        selection_seed=77,
        enable_features=False,
    )
    first = run_hdf5_pipeline(path, config)
    second = run_hdf5_pipeline(path, config)
    assert first.selected_indices == second.selected_indices
    assert all(truth.modulation == "QPSK" for truth in first.truths)
    assert all(truth.snr_db == 30 and truth.channel == 1 for truth in first.truths)
    assert all(sidecar.feature_status == "skipped" for sidecar in first.captures)


def test_cancellation_before_admission_is_explicit(tmp_path) -> None:
    path = tmp_path / "test.h5"
    _dataset(path)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RunCancelled, match="cancelled"):
        run_hdf5_pipeline(path, HarnessConfig(frame_count=2), cancel_event=cancelled)


def test_invalid_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.zeros((2, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        inspect_hdf5_dataset(path)
