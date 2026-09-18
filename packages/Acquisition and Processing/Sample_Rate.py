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
from ._common import _time_scale, _as_iq_array


def handle_sample_rate(
    iq_data=None,
    *,
    sample_rate=None,
    sample_count=None,
    sample_timestamps=None,
    timestamp_unit=None,
    tolerance_ppm=100.0,
    prefer="declared",
):
    """Validate/derive sample-rate information and optionally estimate Fs from timestamps."""
    if isinstance(iq_data, dict):
        if sample_rate is None:
            sample_rate = iq_data.get("sample_rate")
        if sample_count is None:
            sample_count = iq_data.get("sample_count")
        if sample_timestamps is None:
            sample_timestamps = iq_data.get("sample_timestamps")
        if timestamp_unit is None:
            timestamp_unit = iq_data.get("timestamp_unit", "ns")
    elif iq_data is not None and sample_count is None:
        sample_count = _as_iq_array(iq_data).size

    if timestamp_unit is None:
        timestamp_unit = "ns"
    scale = _time_scale(timestamp_unit)

    declared = None
    if sample_rate is not None:
        declared = float(sample_rate)
        if not np.isfinite(declared) or declared <= 0:
            raise ValueError("sample_rate must be a positive finite number")

    if sample_count is not None:
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")

    estimated = None
    jitter_rms_s = None
    duplicate_timestamps = False
    warnings = []

    if sample_timestamps is not None:
        ts = np.asarray(sample_timestamps).reshape(-1)
        if ts.size < 2:
            raise ValueError("at least two sample timestamps are required to estimate sample rate")
        if sample_count is not None and ts.size != sample_count:
            raise ValueError("sample_timestamps length does not match sample_count")
        d = np.diff(ts)
        if np.any(d < 0):
            raise ValueError("sample_timestamps must be nondecreasing")
        duplicate_timestamps = bool(np.any(d == 0))
        if duplicate_timestamps:
            warnings.append("sample timestamps contain duplicates/quantization")

        # Endpoint estimate is robust to quantized individual timestamps.
        span_s = float(ts[-1] - ts[0]) * scale
        if span_s <= 0:
            raise ValueError("timestamp span must be positive")
        estimated = (ts.size - 1) / span_s

        deltas_s = d.astype(np.float64) * scale
        expected_period = 1.0 / estimated
        jitter_rms_s = float(np.sqrt(np.mean((deltas_s - expected_period) ** 2)))

    if declared is None and estimated is None:
        raise ValueError("provide sample_rate or sample_timestamps")

    prefer = str(prefer).lower()
    if prefer not in {"declared", "estimated"}:
        raise ValueError("prefer must be 'declared' or 'estimated'")
    effective = declared if (prefer == "declared" and declared is not None) else estimated
    if effective is None:
        effective = declared if declared is not None else estimated

    rate_error_ppm = None
    if declared is not None and estimated is not None:
        rate_error_ppm = (estimated - declared) / declared * 1e6
        if abs(rate_error_ppm) > float(tolerance_ppm):
            warnings.append("timestamp-derived sample rate differs from declared sample rate")

    return {
        "sample_rate_hz": float(effective),
        "sampling_period_s": 1.0 / float(effective),
        "nyquist_frequency_hz": float(effective) / 2.0,
        "sample_count": sample_count,
        "sample_span_s": None if sample_count is None else (sample_count - 1) / float(effective),
        "capture_interval_s": None if sample_count is None else sample_count / float(effective),
        "declared_sample_rate_hz": declared,
        "estimated_sample_rate_hz": estimated,
        "sample_rate_error_ppm": rate_error_ppm,
        "timestamp_jitter_rms_s": jitter_rms_s,
        "duplicate_timestamps": duplicate_timestamps,
        "warnings": warnings,
    }


__all__ = ["handle_sample_rate"]


def main() -> None:
    print("=" * 60)
    print("  handle_sample_rate: Demonstration")
    print("=" * 60)
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * (np.arange(1024) / 1e6))
    res = handle_sample_rate(sample_iq, sample_rate=1e6)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
