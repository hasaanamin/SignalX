"""Run with -I: verify wheel bytes/origins, then public tests against that wheel."""
import hashlib
import os
from pathlib import Path
import sys


def main():
    target, source = (Path(arg).resolve() for arg in sys.argv[1:])
    sys.path.insert(0, str(target))
    # Independent checks import their sibling probe modules. This directory has
    # no production package; keep the installed target first and assert origins.
    sys.path.insert(1, str(source/"checks"))
    import signal_processing
    assert Path(signal_processing.__file__).resolve().is_relative_to(target)
    count = 0
    for file in (source/"signal_processing").rglob("*.py"):
        installed = target/file.relative_to(source)
        assert hashlib.sha256(file.read_bytes()).digest() == hashlib.sha256(installed.read_bytes()).digest(), str(file)
        count += 1
    print(f"Installed origin: {signal_processing.__file__}; {count} production modules byte-identical", flush=True)
    os.environ["SIGNAL_PROCESSING_SRC"] = str(target)
    # Repository-only benchmarks are intentionally not in the wheel. Register
    # just that namespace for the failure-injection check, not the checkout root.
    import types
    benchmark_namespace = types.ModuleType("benchmarks")
    benchmark_namespace.__path__ = [str(source/"benchmarks")]
    sys.modules["benchmarks"] = benchmark_namespace
    import pytest
    files = ["tests/test_online_noise.py", "tests/test_ar1_detection.py", "tests/test_smoke_dataset_ui.py", "tests/test_hdf5_pipeline_ui.py",
             "checks/test_signal_processing2_acceptance.py", "checks/test_new_mechanism_acceptance.py",
             "checks/test_prior_repairs_against_signal_processing2.py", "checks/test_examples2_acceptance.py",
             "checks/test_f_public_acceptance.py", "checks/test_g_public_acceptance.py"]
    status = pytest.main(["-q", "--import-mode=importlib", *[str(source/f) for f in files]])
    for name, module in list(sys.modules.items()):
        if (name == "signal_processing" or name.startswith("signal_processing.")) and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(target), (name, module.__file__)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
