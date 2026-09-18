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


def align_timestamps(
    event_timestamps,
    iq_data=None,
    *,
    sample_count=None,
    sample_rate=None,
    start_timestamp=None,
    sample_timestamps=None,
    timestamp_unit=None,
):
    """Map event/PDW timestamps to nearest IQ sample indices.

    Priority:
        1) actual per-sample timestamps
        2) sample-0 timestamp + uniform sample rate

    All timestamps must use the same clock/time reference.
    """
    if isinstance(iq_data, dict):
        if sample_count is None:
            sample_count = iq_data.get("sample_count")
        if sample_rate is None:
            sample_rate = iq_data.get("sample_rate")
        if start_timestamp is None:
            start_timestamp = iq_data.get("timestamp")
        if sample_timestamps is None:
            sample_timestamps = iq_data.get("sample_timestamps")
        if timestamp_unit is None:
            timestamp_unit = iq_data.get("timestamp_unit", "ns")
    elif iq_data is not None and sample_count is None:
        sample_count = _as_iq_array(iq_data).size

    if sample_count is None:
        raise ValueError("sample_count is required")
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    if timestamp_unit is None:
        timestamp_unit = "ns"
    scale = _time_scale(timestamp_unit)
    events = np.atleast_1d(event_timestamps)

    if sample_timestamps is not None:
        ts = np.asarray(sample_timestamps).reshape(-1)
        if ts.size != sample_count:
            raise ValueError("sample_timestamps length does not match sample_count")
        if ts.size > 1 and np.any(np.diff(ts) < 0):
            raise ValueError("sample_timestamps must be nondecreasing")

        results = []
        for event in events:
            e = event.item() if hasattr(event, "item") else event
            pos = int(np.searchsorted(ts, e, side="left"))
            if pos <= 0:
                idx = 0
            elif pos >= ts.size:
                idx = ts.size - 1
            else:
                before, after = pos - 1, pos
                idx = before if abs(e - ts[before]) <= abs(ts[after] - e) else after

            error_units = e - ts[idx]
            results.append({
                "event_timestamp": e,
                "nearest_sample_index": int(idx),
                "sample_timestamp": ts[idx].item() if hasattr(ts[idx], "item") else ts[idx],
                "alignment_error_s": float(error_units) * scale,
                "in_range": bool(ts[0] <= e <= ts[-1]),
                "method": "per_sample_timestamps",
            })
        return results

    if start_timestamp is None or sample_rate is None:
        raise ValueError(
            "provide sample_timestamps, or both start_timestamp and sample_rate"
        )
    fs = float(sample_rate)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sample_rate must be positive")

    results = []
    for event in events:
        e = event.item() if hasattr(event, "item") else event
        # Integer subtraction first preserves precision for epoch nanoseconds.
        if isinstance(e, (int, np.integer)) and isinstance(start_timestamp, (int, np.integer)):
            delta_seconds = (int(e) - int(start_timestamp)) * scale
        else:
            delta_seconds = (float(e) - float(start_timestamp)) * scale

        fractional = delta_seconds * fs
        nearest = int(np.rint(fractional))
        error_s = (fractional - nearest) / fs
        results.append({
            "event_timestamp": e,
            "fractional_sample_index": float(fractional),
            "nearest_sample_index": nearest,
            "alignment_error_s": float(error_s),
            "in_range": bool(0.0 <= fractional <= sample_count - 1),
            "method": "uniform_sample_rate",
        })
    return results


__all__ = ["align_timestamps"]


def main() -> None:
    print("=" * 60)
    print("  align_timestamps: Demonstration")
    print("=" * 60)
    events = np.array([100e-6, 500e-6])
    res = align_timestamps(events, sample_count=1024, sample_rate=1e6, start_timestamp=0)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
