"""Isolated source/LTE public tests against the installed wheel, including spawn."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',type=Path)
    parser.add_argument('source',type=Path)
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    target,source = args.target.resolve(),args.source.resolve()
    report = (args.report or target.parent/'installed_tests.xml').resolve()
    if report.exists():
        raise ValueError('Choose a fresh installed-test report path')
    support = target.parent/"lte_test_support"
    tests = support/"tests"
    tests.mkdir(parents=True, exist_ok=True)
    for filename in ("__init__.py", "test_replay_source.py", "test_source_runner.py", "test_process_source_runner.py", "test_runtime_import_boundaries.py", "test_lte_realtime_cli.py", "test_replay_evaluation.py",
                     "test_lte_sync.py", "lte_fixture.py", "test_lte_pbch.py", "lte_pbch_fixture.py", "test_lte_low_snr.py",
                     "test_runtime_audit_harness.py", "test_pluto_source.py"):
        shutil.copy2(source/"tests"/filename, tests/filename)
    (tests/"regressions").mkdir(exist_ok=True)
    for filename in ('__init__.py', 'test_process_analysis.py', 'test_large_block_and_native.py',
                     'test_native_history_and_progress.py', 'test_isolated_capture.py', 'test_scheduling_repairs.py',
                     'test_h_metadata.py', 'test_g_revision.py', 'test_f_revision.py', 'test_on_demand_features.py',
                     'test_streaming_correlation_cache.py','test_shared_payload_completion.py',
                     'test_realtime_1920.py','test_blind_search_realtime.py','test_performance_qos.py','test_ordered_publication.py',
                     'test_core_reservation.py'):
        original = source/'tests/regressions'/filename
        if original.exists():
            shutil.copy2(original, tests/'regressions'/filename)
    (tests/"fixtures").mkdir(exist_ok=True)
    for filename in ("lte_sync_sequences.npz", "lte_sync_sequences.json",
                     "lte_bch_reference.npz", "lte_bch_reference.json"):
        shutil.copy2(source/"tests/fixtures"/filename, tests/"fixtures"/filename)
    sys.path.insert(0, str(target))
    sys.path.insert(1, str(support))
    os.chdir(support)
    import signal_processing
    assert Path(signal_processing.__file__).resolve().is_relative_to(target)
    count = 0
    for file in (source/"signal_processing").rglob("*.py"):
        installed = target/file.relative_to(source)
        assert hashlib.sha256(file.read_bytes()).digest() == hashlib.sha256(installed.read_bytes()).digest(), str(file)
        count += 1
    print(f"Installed origin: {signal_processing.__file__}; {count} production modules byte-identical", flush=True)
    namespace = types.ModuleType("benchmarks")
    namespace.__path__ = [str(source/"benchmarks")]
    sys.modules["benchmarks"] = namespace
    import pytest
    # The support tree may live below an existing checkout. An explicit root
    # keeps importlib test names spawn-importable as tests.*, rather than output.*.
    result = pytest.main(["-q", "--import-mode=importlib", f"--rootdir={support}", str(tests),
                          f"--junitxml={report}"])
    for name, module in list(sys.modules.items()):
        if (name == "signal_processing" or name.startswith("signal_processing.")) and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(target), (name, module.__file__)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
