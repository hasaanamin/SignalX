"""Acceptance enforcement for source-only replay and optional real DSP execution."""
import numpy as np

from benchmarks.evaluate_replay import evaluate, main


def test_cli_pass_path_uses_real_recording_and_balances_all_samples(tmp_path, capsys):
    path = tmp_path / "iq.cf32"
    np.ones(128, np.complex64).tofile(path)
    assert main(["--input", str(path), "--unpaced", "--block-size", "128",
                 "--capacity", "1", "--enforce-acceptance"]) == 0
    assert '"sample_accounting_balanced": true' in capsys.readouterr().out


def test_cli_fails_required_acquisition_gate_with_absent_capacity(tmp_path, capsys):
    path = tmp_path / "iq.cf32"
    np.ones(1024, np.complex64).tofile(path)
    # One-sample source blocks race against a deliberately paused real consumer.
    # Regardless of its first admission timing, capacity cannot hold the recording.
    assert main(["--input", str(path), "--unpaced", "--block-size", "1",
                 "--capacity", "1", "--consumer-delay", ".02",
                 "--enforce-acceptance"]) == 1
    assert '"passed": false' in capsys.readouterr().out


def test_replay_evaluation_runs_actual_pipeline_and_drains_results():
    result = evaluate(seconds=.005, scheduled=False, block_size=1024,
                      capacity=32, noise_power=1.8e-5)
    assert result["passed"] and result["sample_accounting_balanced"]
    assert result["scope"] == "capture/features"
    assert result["runner"]["outstanding_deliveries"] == 0
    assert result["errors"] == []
