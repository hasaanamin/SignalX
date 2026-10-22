"""CLI configuration routing; numerical execution is covered by runner tests."""

from types import SimpleNamespace

import pytest

from signal_processing.ui import hdf5_app, smoke_runner


@pytest.mark.parametrize("profile,hop", [(None, 8), ("online_ar1", 64), ("online_gaussian", 64)])
def test_detector_cli_defaults_to_spectral_but_preserves_online_options(monkeypatch, profile, hop):
    configs = []
    monkeypatch.setattr(smoke_runner, "inspect_smoke_dataset", lambda path: [SimpleNamespace(record_id="routing")])

    def observe_config(record, config):
        configs.append(config)
        return SimpleNamespace(record_id="routing", metrics={"misses": 0, "extras": 0, "feature_failures": 0})

    monkeypatch.setattr(smoke_runner, "run_smoke_record", observe_config)
    monkeypatch.setattr(smoke_runner, "compact_smoke_result", lambda result: result)
    args = ["--headless", "--smoke-dataset", "routing-only"]
    if profile:
        args += ["--detector-profile", profile]
    assert hdf5_app.main(args) == 0
    assert len(configs) == 1
    assert configs[0].detector_profile == (profile or "ar1_spectral")
    assert configs[0].window_size == 256
    assert configs[0].hop_size == hop


@pytest.mark.parametrize("profile,mode", [(None, "legacy"), ("online_ar1", "online_ar1")])
def test_hdf5_cli_defaults_to_legacy_and_allows_explicit_online(monkeypatch, profile, mode):
    calls = []
    monkeypatch.setattr(hdf5_app, "_headless", lambda path, frames, noise_mode: calls.append(noise_mode) or 0)
    args = ["--headless", "--dataset", "routing-only.h5"]
    if profile:
        args += ["--detector-profile", profile]
    assert hdf5_app.main(args) == 0
    assert calls == [mode]
