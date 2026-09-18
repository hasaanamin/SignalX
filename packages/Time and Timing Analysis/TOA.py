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

from typing import Any, Sequence
import numpy as np

from .models import TOAResult, UINT64_MAX, UINT64_MIN
from ._common import (
    validate_sample_rate,
    validate_uint64_timestamp,
    validate_processing_delay_sec,
    calculate_hardware_toas_ns,
    calculate_relative_toas_sec,
    ensure_1d_float_array,
    ensure_1d_int_array,
)


class TimeOfArrival:
    """
    Time of Arrival (TOA) analysis calculator.

    Computes accurate event arrival timestamps with support for:
    1. Genuine hardware capture timestamps (sample-0 absolute nanoseconds, integer precision).
    2. Capture-relative timestamps (seconds from sample 0 of the acquisition window).
    3. Direct externally supplied absolute or relative TOAs.
    """

    def __init__(self, processing_delay_sec: float = 0.0) -> None:
        self.processing_delay_sec = validate_processing_delay_sec(processing_delay_sec)

    def analyze(
        self,
        pulse_start_indices: np.ndarray | Sequence[int] | None = None,
        sample_rate_hz: float | None = None,
        hardware_timestamp_ns: int | None = None,
        pulse_toas: np.ndarray | Sequence[float] | None = None,
        pulse_toas_ns: np.ndarray | Sequence[int] | None = None,
        pulse_toas_absolute_ns: np.ndarray | Sequence[int] | None = None,
        pulse_toas_relative_sec: np.ndarray | Sequence[float] | None = None,
        toa_reference: str | None = None,
        processing_delay_sec: float | None = None,
        already_delay_corrected: bool = False,
        toa_already_delay_corrected: bool = False,
    ) -> TOAResult:
        """
        Executes TOA calculation with explicit reference semantics.
        """
        p_delay = self.processing_delay_sec if processing_delay_sec is None else validate_processing_delay_sec(processing_delay_sec)
        hw_ts = validate_uint64_timestamp(hardware_timestamp_ns)
        already_corr = bool(already_delay_corrected or toa_already_delay_corrected)

        # Validate conflicting direct inputs
        abs_ns = pulse_toas_absolute_ns if pulse_toas_absolute_ns is not None else pulse_toas_ns
        rel_sec = pulse_toas_relative_sec if pulse_toas_relative_sec is not None else pulse_toas

        if abs_ns is not None and rel_sec is not None:
            raise ValueError(
                "Ambiguous direct TOA input: both absolute nanoseconds and relative seconds supplied simultaneously."
            )

        # Validate contradictory reference declarations
        if toa_reference is not None:
            if toa_reference == "hardware_absolute" and hw_ts is None and rel_sec is not None and abs_ns is None:
                raise ValueError(
                    "Contradictory timing reference: 'hardware_absolute' declared but only relative TOAs supplied and no hardware timestamp exists."
                )
            if toa_reference == "capture_relative" and abs_ns is not None and rel_sec is None:
                raise ValueError(
                    "Contradictory timing reference: 'capture_relative' declared but absolute nanosecond TOAs supplied."
                )

        meta: dict[str, Any] = {}

        # Mode A: Direct absolute nanosecond TOAs provided
        if abs_ns is not None:
            toas_ns = ensure_1d_int_array(abs_ns, name="pulse_toas_ns")
            delay_applied = (not already_corr and p_delay > 0.0)
            if delay_applied:
                delay_ns = int(round(p_delay * 1e9))
                toas_ns = [t - delay_ns for t in toas_ns]
            meta["processing_delay_applied"] = delay_applied
            meta["known_processing_delay_sec"] = p_delay
            meta["already_delay_corrected"] = already_corr

            toas_sec = np.array([float(t) / 1e9 for t in toas_ns], dtype=np.float64)
            ref = toa_reference or ("hardware_absolute" if hw_ts is not None else "externally_absolute")

            return TOAResult(
                toas_sec=toas_sec,
                toas_ns=toas_ns,
                timing_reference=ref,
                sample_count=len(toas_ns),
                time_unit="s",
                sample0_hardware_timestamp_ns=hw_ts,
                group_delay_corrected_sec=p_delay,
                processing_delay_applied=delay_applied,
                metadata=meta,
            )

        # Mode B: Direct relative seconds TOAs provided
        if rel_sec is not None:
            raw_sec = ensure_1d_float_array(rel_sec, name="pulse_toas")
            delay_applied = (not already_corr and p_delay > 0.0)
            if delay_applied:
                toas_sec = raw_sec - p_delay
            else:
                toas_sec = raw_sec.copy()
            meta["processing_delay_applied"] = delay_applied
            meta["known_processing_delay_sec"] = p_delay
            meta["already_delay_corrected"] = already_corr

            ref = toa_reference or "capture_relative"

            toas_ns_converted: list[int] | None = None
            if hw_ts is not None and ref == "hardware_absolute":
                toas_ns_converted = [hw_ts + int(round(float(t) * 1e9)) for t in toas_sec]
                toas_sec = np.array([float(t) / 1e9 for t in toas_ns_converted], dtype=np.float64)

            return TOAResult(
                toas_sec=toas_sec,
                toas_ns=toas_ns_converted,
                timing_reference=ref,
                sample_count=len(toas_sec),
                time_unit="s",
                sample0_hardware_timestamp_ns=hw_ts,
                group_delay_corrected_sec=p_delay,
                processing_delay_applied=delay_applied,
                metadata=meta,
            )

        # Mode C: Computed from pulse_start_indices and sample_rate_hz
        if pulse_start_indices is None:
            raise ValueError("Either pulse_start_indices or direct pulse_toas must be provided")

        fs = validate_sample_rate(sample_rate_hz)
        indices_list = ensure_1d_int_array(pulse_start_indices, name="pulse_start_indices")
        indices = np.array(indices_list, dtype=np.float64)

        delay_applied = (not already_corr and p_delay > 0.0)
        eff_delay = p_delay if delay_applied else 0.0
        meta["processing_delay_applied"] = delay_applied
        meta["known_processing_delay_sec"] = p_delay
        meta["already_delay_corrected"] = already_corr

        if hw_ts is not None:
            toas_sec, toas_ns = calculate_hardware_toas_ns(
                hw_timestamp_ns=hw_ts,
                indices=indices,
                sample_rate_hz=fs,
                processing_delay_sec=eff_delay,
            )
            timing_ref = toa_reference or "hardware_absolute"
            sample0_hw = hw_ts
        else:
            if toa_reference == "hardware_absolute":
                raise ValueError("Contradictory timing reference: 'hardware_absolute' declared without hardware_timestamp_ns.")
            toas_sec = calculate_relative_toas_sec(
                indices=indices,
                sample_rate_hz=fs,
                processing_delay_sec=eff_delay,
            )
            toas_ns = None
            timing_ref = toa_reference or "capture_relative"
            sample0_hw = None

        return TOAResult(
            toas_sec=toas_sec,
            toas_ns=toas_ns,
            timing_reference=timing_ref,
            sample_count=len(toas_sec),
            time_unit="s",
            sample0_hardware_timestamp_ns=sample0_hw,
            group_delay_corrected_sec=p_delay,
            processing_delay_applied=delay_applied,
            metadata=meta,
        )


def compute_toa(
    pulse_start_indices: np.ndarray | Sequence[int] | None = None,
    sample_rate_hz: float | None = None,
    hardware_timestamp_ns: int | None = None,
    pulse_toas: np.ndarray | Sequence[float] | None = None,
    pulse_toas_ns: np.ndarray | Sequence[int] | None = None,
    pulse_toas_absolute_ns: np.ndarray | Sequence[int] | None = None,
    pulse_toas_relative_sec: np.ndarray | Sequence[float] | None = None,
    toa_reference: str | None = None,
    processing_delay_sec: float = 0.0,
    already_delay_corrected: bool = False,
    toa_already_delay_corrected: bool = False,
) -> TOAResult:
    """
    Functional API to compute Time of Arrival (TOA).
    """
    calc = TimeOfArrival(processing_delay_sec=processing_delay_sec)
    return calc.analyze(
        pulse_start_indices=pulse_start_indices,
        sample_rate_hz=sample_rate_hz,
        hardware_timestamp_ns=hardware_timestamp_ns,
        pulse_toas=pulse_toas,
        pulse_toas_ns=pulse_toas_ns,
        pulse_toas_absolute_ns=pulse_toas_absolute_ns,
        pulse_toas_relative_sec=pulse_toas_relative_sec,
        toa_reference=toa_reference,
        processing_delay_sec=processing_delay_sec,
        already_delay_corrected=already_delay_corrected,
        toa_already_delay_corrected=toa_already_delay_corrected,
    )


__all__ = ["TimeOfArrival", "compute_toa"]


def main() -> None:
    print("=" * 60)
    print("  TOA Module Demonstration")
    print("=" * 60)
    idx = [100, 200, 300]
    fs = 1e6
    hw_ts = 1718000000000000000
    res_abs = compute_toa(pulse_start_indices=idx, sample_rate_hz=fs, hardware_timestamp_ns=hw_ts)
    print(f"Hardware Absolute TOA: {res_abs.timing_reference}, TOAs (ns): {res_abs.toas_ns}")
    res_rel = compute_toa(pulse_start_indices=idx, sample_rate_hz=fs, hardware_timestamp_ns=None)
    print(f"Capture Relative TOA: {res_rel.timing_reference}, TOAs (s): {res_rel.toas_sec}")
    print("=" * 60)


if __name__ == "__main__":
    main()
