"""Isolated wheel parity/import verification plus current replay public tests.

Run: python -I benchmarks/verify_replay_install.py INSTALLED_TARGET SOURCE_ROOT
Tests are copied to a separate support directory so spawn can import their public
test entry points without putting the production checkout on sys.path.
"""
import hashlib
import os
from pathlib import Path
import shutil
import sys
import types


def main():
    target, source = (Path(arg).resolve() for arg in sys.argv[1:])
    support = target.parent / "replay_test_support"
    tests = support / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    for filename in ("__init__.py", "test_replay_source.py", "test_source_runner.py", "test_replay_evaluation.py"):
        shutil.copy2(source/"tests"/filename, tests/filename)
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
    # Benchmark code is repository-side; this only exposes that namespace.
    namespace = types.ModuleType("benchmarks")
    namespace.__path__ = [str(source/"benchmarks")]
    sys.modules["benchmarks"] = namespace
    import pytest
    status = pytest.main(["-q", "--import-mode=importlib", str(tests),
                          f"--junitxml={target.parent/'installed_tests.xml'}"])
    for name, module in list(sys.modules.items()):
        if (name == "signal_processing" or name.startswith("signal_processing.")) and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(target), (name, module.__file__)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
