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


def filter_iq(
    iq_data,
    *,
    sample_rate=None,
    filter_type="lowpass",
    cutoff=None,
    num_taps=129,
    taps=None,
    taps_file=None,
    window="hamming",
    zero_phase=False,
    zi=None,
    return_info=False,
):
    """FIR-filter complex IQ.

    If taps is None, SciPy firwin designs lowpass/highpass/bandpass/bandstop taps.
    Standard firwin band filters have real coefficients and therefore symmetric
    positive/negative-frequency responses for complex IQ. Supply custom complex
    taps for an arbitrary one-sided complex filter.

    For live chunked processing use zero_phase=False and pass the returned 'zf'
    from one call as zi to the next call. zero_phase=True is offline/noncausal.
    """
    try:
        from scipy.signal import firwin, lfilter, filtfilt
    except ImportError as exc:
        raise ImportError("filter_iq requires scipy") from exc

    if taps_file is not None and taps is None:
        taps = np.load(taps_file)

    x = _as_iq_array(iq_data)

    if sample_rate is None and isinstance(iq_data, dict):
        sample_rate = iq_data.get("sample_rate")
    fs = None if sample_rate is None else float(sample_rate)
    if fs is not None and (not np.isfinite(fs) or fs <= 0):
        raise ValueError("sample_rate must be positive")

    designed_here = taps is None
    if taps is None:
        if fs is None:
            raise ValueError("sample_rate is required when designing filter taps")
        if cutoff is None:
            raise ValueError("cutoff is required when designing filter taps")
        if not isinstance(num_taps, (int, np.integer)) or int(num_taps) < 2:
            raise ValueError("num_taps must be an integer >= 2")
        num_taps = int(num_taps)
        ftype = str(filter_type).lower()
        valid = {"lowpass", "highpass", "bandpass", "bandstop"}
        if ftype not in valid:
            raise ValueError(f"filter_type must be one of {sorted(valid)}")
        nyquist = fs / 2.0

        if ftype in {"lowpass", "highpass"}:
            cutoff_checked = float(cutoff)
            if not 0 < cutoff_checked < nyquist:
                raise ValueError("cutoff must satisfy 0 < cutoff < sample_rate/2")
        else:
            if not isinstance(cutoff, (tuple, list, np.ndarray)) or len(cutoff) != 2:
                raise ValueError("bandpass/bandstop cutoff must be (low_hz, high_hz)")
            low, high = float(cutoff[0]), float(cutoff[1])
            if not 0 < low < high < nyquist:
                raise ValueError("band edges must satisfy 0 < low < high < Nyquist")
            cutoff_checked = (low, high)

        if ftype in {"highpass", "bandstop"} and num_taps % 2 == 0:
            raise ValueError("use an odd num_taps for highpass/bandstop FIR filters")

        taps_arr = firwin(
            num_taps,
            cutoff_checked,
            pass_zero=ftype,
            fs=fs,
            window=window,
        )
    else:
        taps_arr = np.asarray(taps)
        if taps_arr.ndim != 1 or taps_arr.size == 0:
            raise ValueError("taps must be a non-empty 1-D array")
        if not np.all(np.isfinite(taps_arr.real)) or not np.all(np.isfinite(taps_arr.imag)):
            raise ValueError("taps contains NaN or infinity")
        if np.iscomplexobj(taps_arr):
            taps_arr = taps_arr.astype(np.complex128)
        else:
            taps_arr = taps_arr.astype(np.float64)
        ftype = "custom"

    zf = None
    if zero_phase:
        if zi is not None:
            raise ValueError("zi cannot be used with zero_phase=True")
        # Gustafsson avoids the default pad-length limitation on short records.
        y = filtfilt(taps_arr, [1.0], x, method="gust")
    else:
        if zi is None:
            zi_work = np.zeros(max(len(taps_arr) - 1, 0), dtype=np.complex128)
        else:
            zi_work = np.asarray(zi, dtype=np.complex128).reshape(-1)
            expected_state = max(len(taps_arr) - 1, 0)
            if zi_work.size != expected_state:
                raise ValueError(f"zi must contain {expected_state} state values")
        y, zf = lfilter(taps_arr, [1.0], x, zi=zi_work)

    group_delay_samples = None
    group_delay_s = None
    if (
        np.allclose(np.imag(taps_arr), 0.0)
        and np.allclose(np.real(taps_arr), np.real(taps_arr)[::-1])
    ):
        group_delay_samples = (len(taps_arr) - 1) / 2.0
        if fs is not None:
            group_delay_s = group_delay_samples / fs

    info = {
        "operation": "filter",
        "filter_type": ftype,
        "designed_here": designed_here,
        "num_taps": int(len(taps_arr)),
        "group_delay_samples": group_delay_samples,
        "group_delay_s": group_delay_s,
        "zero_phase": bool(zero_phase),
        "zf": zf,
    }
    out = _return_like(iq_data, y, processing_entry={k: v for k, v in info.items() if k != "zf"})
    return (out, info) if return_info else out


__all__ = ["filter_iq"]


def main() -> None:
    print("=" * 60)
    print("  filter_iq: Demonstration")
    print("=" * 60)
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * (np.arange(1024) / 1e6))
    res = filter_iq(sample_iq, sample_rate=1e6, cutoff=200e3)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
