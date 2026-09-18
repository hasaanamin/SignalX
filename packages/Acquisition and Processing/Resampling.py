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

from fractions import Fraction
import numpy as np
from ._common import _as_iq_array, _return_like


def resample_iq(
    iq_data,
    *,
    input_sample_rate=None,
    output_sample_rate=None,
    decimation_factor=None,
    max_denominator=100000,
    return_info=False,
):
    """Polyphase-resample complex IQ with anti-alias/interpolation filtering.

    Specify exactly one of output_sample_rate or decimation_factor.
    Canonical-record input produces canonical-record output with updated sample
    rate/count. Per-sample timestamps are cleared after resampling because a new
    time grid has been created; the sample-0 timestamp is preserved.
    """
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError("resample_iq requires scipy") from exc

    x = _as_iq_array(iq_data)

    if input_sample_rate is None and isinstance(iq_data, dict):
        input_sample_rate = iq_data.get("sample_rate")
    if input_sample_rate is None:
        raise ValueError("input_sample_rate is required")
    fs_in = float(input_sample_rate)
    if not np.isfinite(fs_in) or fs_in <= 0:
        raise ValueError("input_sample_rate must be positive")

    if (output_sample_rate is None) == (decimation_factor is None):
        raise ValueError("specify exactly one of output_sample_rate or decimation_factor")

    if decimation_factor is not None:
        if not isinstance(decimation_factor, (int, np.integer)) or int(decimation_factor) < 1:
            raise ValueError("decimation_factor must be a positive integer")
        up = 1
        down = int(decimation_factor)
        requested_fs = fs_in / down
    else:
        requested_fs = float(output_sample_rate)
        if not np.isfinite(requested_fs) or requested_fs <= 0:
            raise ValueError("output_sample_rate must be positive")
        ratio = requested_fs / fs_in
        frac = Fraction(ratio).limit_denominator(int(max_denominator))
        up, down = frac.numerator, frac.denominator

    y = resample_poly(x, up, down)
    actual_fs = fs_in * up / down
    error_ppm = (actual_fs - requested_fs) / requested_fs * 1e6

    info = {
        "operation": "resample",
        "input_sample_rate_hz": fs_in,
        "requested_sample_rate_hz": requested_fs,
        "actual_sample_rate_hz": actual_fs,
        "up": int(up),
        "down": int(down),
        "sample_rate_error_ppm": float(error_ppm),
        "input_sample_count": int(x.size),
        "output_sample_count": int(y.size),
    }

    out = _return_like(
        iq_data,
        y,
        sample_rate=actual_fs,
        sample_timestamps=None,
        processing_entry=info,
    )
    return (out, info) if return_info else out


def decimate_iq(
    iq_data,
    factor,
    *,
    input_sample_rate=None,
    return_info=False,
):
    """Integer-factor decimation wrapper around resample_iq()."""
    return resample_iq(
        iq_data,
        input_sample_rate=input_sample_rate,
        decimation_factor=factor,
        return_info=return_info,
    )


__all__ = ["resample_iq", "decimate_iq"]


def main() -> None:
    print("=" * 60)
    print("  resample_iq: Demonstration")
    print("=" * 60)
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * (np.arange(1024) / 1e6))
    res = resample_iq(sample_iq, input_sample_rate=1e6, output_sample_rate=2e6)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
