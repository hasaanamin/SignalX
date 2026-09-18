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


def calibrate_iq(
    iq_data,
    *,
    dc_offset=0j,
    iq_matrix=None,
    gain_linear=1.0,
    gain_db=None,
    phase_deg=0.0,
    volts_per_count=None,
    return_info=False,
):
    """Apply generic static linear IQ calibration.

    Model:
        [Ic, Qc]^T = M * ([I, Q]^T - [Ioff, Qoff]^T)
        x_cal = G * exp(j*phi) * (Ic + j*Qc)

    volts_per_count optionally converts calibrated ADC counts to volts.
    """
    x = _as_iq_array(iq_data)

    if isinstance(dc_offset, (tuple, list)):
        if len(dc_offset) != 2:
            raise ValueError("dc_offset tuple/list must be (I_offset, Q_offset)")
        offset = complex(float(dc_offset[0]), float(dc_offset[1]))
    else:
        offset = complex(dc_offset)

    y = x - offset

    if iq_matrix is not None:
        m = np.asarray(iq_matrix, dtype=np.float64)
        if m.shape != (2, 2) or not np.all(np.isfinite(m)):
            raise ValueError("iq_matrix must be a finite 2x2 matrix")
        i = y.real
        q = y.imag
        y = (m[0, 0] * i + m[0, 1] * q) + 1j * (m[1, 0] * i + m[1, 1] * q)

    if gain_db is not None:
        if gain_linear != 1.0:
            raise ValueError("specify gain_linear or gain_db, not both")
        gain = 10.0 ** (float(gain_db) / 20.0)  # amplitude dB -> linear
    else:
        gain = float(gain_linear)
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("calibration gain must be finite and non-negative")

    phase_deg = float(phase_deg)
    if not np.isfinite(phase_deg):
        raise ValueError("phase_deg must be finite")
    complex_gain = gain * np.exp(1j * np.deg2rad(phase_deg))
    y = y * complex_gain

    output_units = None
    if volts_per_count is not None:
        volts_per_count = float(volts_per_count)
        if not np.isfinite(volts_per_count) or volts_per_count <= 0:
            raise ValueError("volts_per_count must be a positive finite value")
        y = y * volts_per_count
        output_units = "volts"

    info = {
        "operation": "calibration",
        "dc_offset": offset,
        "iq_matrix": None if iq_matrix is None else np.asarray(iq_matrix).tolist(),
        "gain_linear": gain,
        "phase_deg": phase_deg,
        "volts_per_count": volts_per_count,
    }
    out = _return_like(iq_data, y, sample_units=output_units, processing_entry=info)
    return (out, info) if return_info else out


__all__ = ["calibrate_iq"]


def main() -> None:
    print("=" * 60)
    print("  calibrate_iq: Demonstration")
    print("=" * 60)
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * (np.arange(1024) / 1e6))
    res = calibrate_iq(sample_iq)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
