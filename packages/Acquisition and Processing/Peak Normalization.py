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
from ._common import _as_iq_array, _return_like


def normalize_iq(
    iq_data,
    *,
    mode="peak",
    target=1.0,
    adc_full_scale=None,
    return_info=False,
):
    """Normalize IQ by peak, RMS, or ADC full-scale amplitude."""
    x = _as_iq_array(iq_data)
    target = float(target)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be a positive finite value")

    mode = str(mode).lower()
    if mode == "peak":
        scale = float(np.max(np.abs(x)))
    elif mode == "rms":
        scale = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    elif mode in {"full_scale", "adc_full_scale"}:
        if adc_full_scale is None and isinstance(iq_data, dict):
            adc_full_scale = iq_data.get("adc_full_scale")
        if adc_full_scale is None:
            raise ValueError("adc_full_scale is required for full-scale normalization")
        scale = float(adc_full_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("adc_full_scale must be positive")
    else:
        raise ValueError("mode must be 'peak', 'rms', or 'adc_full_scale'")

    if scale == 0.0:
        gain = 1.0
        y = x.copy()
    else:
        gain = target / scale
        y = x * gain

    info = {
        "operation": "normalization",
        "mode": mode,
        "target": target,
        "original_scale": scale,
        "applied_gain": gain,
    }
    out = _return_like(iq_data, y, sample_units="normalized", processing_entry=info)
    return (out, info) if return_info else out


__all__ = ["normalize_iq"]


def main() -> None:
    print("=" * 60)
    print("  normalize_iq: Demonstration")
    print("=" * 60)
    sample_iq = np.array([1.0 + 1j * 2.0, -3.0 + 1j * 1.0])
    res = normalize_iq(sample_iq)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
