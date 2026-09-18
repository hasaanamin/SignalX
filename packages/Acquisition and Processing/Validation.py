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

import numpy as np
from ._common import _as_iq_array
from .Sample_Rate import handle_sample_rate


def validate_iq(
    iq_data,
    *,
    sample_rate=None,
    sampling_period=None,
    sample_timestamps=None,
    timestamp_unit=None,
    adc_full_scale=None,
    sample_rate_tolerance_ppm=100.0,
):
    """Validate IQ data/record without modifying it."""
    errors = []
    warnings = []
    metrics = {}

    try:
        x = _as_iq_array(iq_data)
    except (TypeError, ValueError) as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": [], "metrics": {}}

    n = int(x.size)
    metrics["observed_sample_count"] = n

    if isinstance(iq_data, dict):
        declared_count = iq_data.get("sample_count")
        if declared_count is not None and int(declared_count) != n:
            errors.append("sample_count does not match decoded IQ length")

        if iq_data.get("i") is not None and iq_data.get("q") is not None:
            i = np.asarray(iq_data["i"]).reshape(-1)
            q = np.asarray(iq_data["q"]).reshape(-1)
            if i.size != n or q.size != n:
                errors.append("I/Q arrays do not match complex_samples length")
            elif not np.allclose(i.astype(np.float64) + 1j * q.astype(np.float64), x):
                errors.append("complex_samples is inconsistent with I + jQ")

        if sample_rate is None:
            sample_rate = iq_data.get("sample_rate")
        if sampling_period is None:
            sampling_period = iq_data.get("sampling_period")
        if sample_timestamps is None:
            sample_timestamps = iq_data.get("sample_timestamps")
        if timestamp_unit is None:
            timestamp_unit = iq_data.get("timestamp_unit", "ns")
        if adc_full_scale is None:
            adc_full_scale = iq_data.get("adc_full_scale")
        sample_units = iq_data.get("sample_units")
    else:
        sample_units = None

    if timestamp_unit is None:
        timestamp_unit = "ns"

    fs = None
    if sample_rate is not None:
        try:
            fs = float(sample_rate)
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("invalid sample_rate")
            fs = None

    if fs is not None:
        metrics["nyquist_frequency_hz"] = fs / 2.0
        if sampling_period is not None:
            try:
                period = float(sampling_period)
                if not np.isfinite(period) or period <= 0:
                    raise ValueError
                if not np.isclose(period, 1.0 / fs, rtol=1e-9, atol=0.0):
                    errors.append("sampling_period does not match 1/sample_rate")
            except (TypeError, ValueError):
                errors.append("invalid sampling_period")

    if sample_timestamps is not None:
        ts = np.asarray(sample_timestamps).reshape(-1)
        if ts.size != n:
            errors.append("sample_timestamps length does not match sample_count")
        elif ts.size > 1 and np.any(np.diff(ts) < 0):
            errors.append("sample_timestamps must be nondecreasing")
        elif fs is not None and ts.size >= 2:
            try:
                rate_info = handle_sample_rate(
                    sample_rate=fs,
                    sample_count=n,
                    sample_timestamps=ts,
                    timestamp_unit=timestamp_unit,
                    tolerance_ppm=sample_rate_tolerance_ppm,
                )
                metrics["timestamp_derived_sample_rate_hz"] = rate_info["estimated_sample_rate_hz"]
                metrics["sample_rate_error_ppm"] = rate_info["sample_rate_error_ppm"]
                metrics["timestamp_jitter_rms_s"] = rate_info["timestamp_jitter_rms_s"]
                warnings.extend(rate_info["warnings"])
            except ValueError as exc:
                errors.append(str(exc))

    if adc_full_scale is not None:
        try:
            full_scale = float(adc_full_scale)
            if not np.isfinite(full_scale) or full_scale <= 0:
                raise ValueError
            # Only compare to ADC full-scale while samples are still in ADC-count units.
            if sample_units in (None, "counts"):
                hits = int(np.count_nonzero((np.abs(x.real) >= full_scale) | (np.abs(x.imag) >= full_scale)))
                metrics["full_scale_hits"] = hits
                if hits:
                    warnings.append("possible ADC clipping/full-scale hits detected")
        except (TypeError, ValueError):
            errors.append("invalid adc_full_scale")

    metrics.update({
        "dc_i": float(np.mean(x.real)),
        "dc_q": float(np.mean(x.imag)),
        "peak_magnitude": float(np.max(np.abs(x))),
        "rms_magnitude": float(np.sqrt(np.mean(np.abs(x) ** 2))),
    })

    if isinstance(iq_data, dict) and iq_data.get("quality_flags"):
        metrics["quality_flags"] = sorted(iq_data["quality_flags"])

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


# Alias for backward compatibility
validate_iq_record = validate_iq

__all__ = ["validate_iq", "validate_iq_record"]


def main() -> None:
    print("=" * 60)
    print("  validate_iq: Demonstration")
    print("=" * 60)
    res = validate_iq(np.exp(1j * 2 * np.pi * 50e3 * (np.arange(1024)/1e6)))
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
