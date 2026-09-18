from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

from typing import Any
from dataclasses import fields
import numpy as np

from .models import TimingInput, TimeAnalysisResult
from .processor import TimeAnalysisProcessor

_TIMING_INPUT_FIELDS = {f.name for f in fields(TimingInput)}


def analyze_time_and_timing(
    timing_input: TimingInput | dict[str, Any] | None = None,
    enabled_modules: list[str] | None = None,
    **kwargs: Any,
) -> TimeAnalysisResult:
    """
    Stateless batch / one-shot entry point for Time and Timing Analysis.

    Instantiates a new TimeAnalysisProcessor per invocation. Does NOT preserve
    or maintain streaming state across calls.

    For real-time / multi-source continuous streaming, instantiate and reuse
    a single long-lived TimeAnalysisProcessor:

    >>> processor = TimeAnalysisProcessor()
    >>> while running:
    ...     timing_input = build_timing_input(...)
    ...     result = processor.process(timing_input)

    Parameters
    ----------
    timing_input : TimingInput | dict[str, Any] | None
        Input timing specification or feature dictionary.
    enabled_modules : list[str] | None
        Optional subset of analysis modules to run.
    **kwargs : Any
        Convenience keyword arguments passed to TimingInput (e.g. sample_rate_hz=1e6, pulse_start_indices=[...]).

    Returns
    -------
    TimeAnalysisResult
        Consolidated analysis result.
    """
    if timing_input is None:
        inp = TimingInput(**{k: v for k, v in kwargs.items() if k in _TIMING_INPUT_FIELDS})
    elif isinstance(timing_input, dict):
        merged = {**timing_input, **kwargs}
        inp = TimingInput(**{k: v for k, v in merged.items() if k in _TIMING_INPUT_FIELDS})
    else:
        inp = timing_input

    processor = TimeAnalysisProcessor()
    return processor.process(timing_input=inp, enabled_modules=enabled_modules)


__all__ = ["analyze_time_and_timing", "TimeAnalysisProcessor"]


def main() -> None:
    print("=" * 60)
    print("  Time and Timing Analysis: Main Pipeline Execution")
    print("=" * 60)
    fs = 2e6
    pulse_indices = np.arange(100, 20100, 1000)
    durations = np.full(len(pulse_indices), 20e-6)
    burst_labels = ["Burst_0"] * 10 + ["Burst_1"] * 10

    inp = TimingInput(
        sample_rate_hz=fs,
        pulse_start_indices=pulse_indices,
        pulse_durations=durations,
        burst_labels=burst_labels,
        hardware_timestamp_ns=1718000000123456789,
        observation_duration_sec=0.02,
    )

    result = analyze_time_and_timing(inp)

    print(f"Executed modules: {result.executed_modules}")
    if result.toa:
        print(f"TOA Reference: {result.toa.timing_reference}, Sample 0 (ns): {result.toa.sample0_hardware_timestamp_ns}")
    if result.pri and result.pri.nominal_pri_sec is not None:
        print(f"Nominal PRI: {result.pri.nominal_pri_sec * 1e6:.1f} us")
    if result.prf:
        print(f"Nominal PRF: {result.prf.nominal_prf_hz:.1f} Hz")
    if result.burst_timing:
        print(f"Num bursts: {result.burst_timing.num_bursts}, Mean BRI: {result.burst_timing.mean_bri_sec}")
    if result.pulse_density:
        print(f"Pulse density: {result.pulse_density.pulse_density_pps:.1f} PPS")
    print("=" * 60)


if __name__ == "__main__":
    main()
