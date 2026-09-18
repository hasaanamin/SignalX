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

import copy
import numpy as np
from ._common import _extract_valid_bits, _unpack_contiguous_bits


def parse_iq(
    raw_data,
    *,
    layout="IQ",
    dtype=None,
    byteorder="little",
    offset_bytes=0,
    valid_bits=None,
    bit_shift=0,
    valid_bits_signed=True,
    packed_bits=None,
    packed_signed=True,
    packed_bit_order="lsb",
    sample_rate=None,
    timestamp=None,
    timestamp_unit="ns",
    sample_timestamps=None,
    adc_resolution=None,
    adc_full_scale=None,
    quality_flags=None,
    metadata=None,
):
    """Parse common raw/host IQ representations into one canonical record.

    Supported layouts:
        IQ       : I0,Q0,I1,Q1,...
        QI       : Q0,I0,Q1,I1,...
        planar   : I0,I1,...,Q0,Q1,...
        separate : raw_data=(I_array, Q_array)
        complex  : already complex
        Nx2      : [[I,Q], ...]
        2xN      : [[I...], [Q...]]
        auto     : detects complex / Nx2 / 2xN; flat real data defaults to IQ

    For raw bytes, dtype is required unless packed_bits is supplied.
    valid_bits/bit_shift support e.g. signed 12-bit data stored in int16.
    packed_bits supports contiguous fixed-width packed scalar samples.
    """
    layout = str(layout).lower()
    valid_layouts = {"iq", "qi", "planar", "separate", "complex", "nx2", "2xn", "auto"}
    if layout not in valid_layouts:
        raise ValueError(f"layout must be one of {sorted(valid_layouts)}")
    if int(offset_bytes) < 0:
        raise ValueError("offset_bytes must be >= 0")
    offset_bytes = int(offset_bytes)

    is_bytes = isinstance(raw_data, (bytes, bytearray, memoryview))

    if layout == "auto":
        if is_bytes:
            layout = "iq"
        elif isinstance(raw_data, tuple) and len(raw_data) == 2:
            layout = "separate"
        else:
            probe = np.asarray(raw_data)
            if np.iscomplexobj(probe):
                layout = "complex"
            elif probe.ndim == 2 and probe.shape[1] == 2:
                layout = "nx2"
            elif probe.ndim == 2 and probe.shape[0] == 2:
                layout = "2xn"
            else:
                layout = "iq"

    # Already-complex input.
    if layout == "complex":
        if is_bytes:
            if dtype is None:
                raise ValueError("dtype is required for complex byte input")
            dt = np.dtype(dtype)
            if dt.kind != "c":
                raise ValueError("layout='complex' requires a complex dtype")
            if byteorder == "little":
                dt = dt.newbyteorder("<")
            elif byteorder == "big":
                dt = dt.newbyteorder(">")
            elif byteorder == "native":
                dt = dt.newbyteorder("=")
            else:
                raise ValueError("invalid byteorder")
            buf = memoryview(raw_data)
            if offset_bytes > len(buf):
                raise ValueError("offset_bytes exceeds input size")
            buf = buf[offset_bytes:]
            if len(buf) % dt.itemsize != 0:
                raise ValueError("complex payload byte length is misaligned")
            z = np.frombuffer(buf, dtype=dt).astype(np.complex128)
        else:
            z = np.asarray(raw_data)
            if not np.iscomplexobj(z):
                raise ValueError("layout='complex' requires complex-valued input")
            z = z.astype(np.complex128).reshape(-1)
        i = z.real.copy()
        q = z.imag.copy()
        inferred_units = "native"

    elif layout == "separate":
        if not isinstance(raw_data, (tuple, list)) or len(raw_data) != 2:
            raise ValueError("layout='separate' requires raw_data=(I, Q)")
        i0 = np.asarray(raw_data[0])
        q0 = np.asarray(raw_data[1])
        i = _extract_valid_bits(i0, valid_bits, bit_shift, valid_bits_signed).astype(np.float64).reshape(-1)
        q = _extract_valid_bits(q0, valid_bits, bit_shift, valid_bits_signed).astype(np.float64).reshape(-1)
        inferred_units = "counts" if i0.dtype.kind in "iu" and q0.dtype.kind in "iu" else "native"

    elif layout in {"nx2", "2xn"}:
        data = np.asarray(raw_data)
        if layout == "nx2":
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("layout='Nx2' requires an array of shape (N, 2)")
            i0, q0 = data[:, 0], data[:, 1]
        else:
            if data.ndim != 2 or data.shape[0] != 2:
                raise ValueError("layout='2xN' requires an array of shape (2, N)")
            i0, q0 = data[0, :], data[1, :]
        i = _extract_valid_bits(i0, valid_bits, bit_shift, valid_bits_signed).astype(np.float64)
        q = _extract_valid_bits(q0, valid_bits, bit_shift, valid_bits_signed).astype(np.float64)
        inferred_units = "counts" if data.dtype.kind in "iu" else "native"

    else:
        # Flat scalar stream.
        if packed_bits is not None:
            if not is_bytes:
                raise TypeError("packed_bits requires bytes-like raw_data")
            buf = memoryview(raw_data)
            if offset_bytes > len(buf):
                raise ValueError("offset_bytes exceeds input size")
            data = _unpack_contiguous_bits(
                buf[offset_bytes:],
                packed_bits,
                signed=bool(packed_signed),
                bit_order=packed_bit_order,
            )
            inferred_units = "counts"
            if adc_resolution is None:
                adc_resolution = int(packed_bits)
        elif is_bytes:
            if dtype is None:
                raise ValueError("dtype is required for byte input unless packed_bits is used")
            dt = np.dtype(dtype)
            if byteorder == "little":
                dt = dt.newbyteorder("<")
            elif byteorder == "big":
                dt = dt.newbyteorder(">")
            elif byteorder == "native":
                dt = dt.newbyteorder("=")
            else:
                raise ValueError("byteorder must be 'little', 'big', or 'native'")
            buf = memoryview(raw_data)
            if offset_bytes > len(buf):
                raise ValueError("offset_bytes exceeds input size")
            buf = buf[offset_bytes:]
            if len(buf) % dt.itemsize != 0:
                raise ValueError("payload byte length is not a multiple of dtype size")
            raw_scalars = np.frombuffer(buf, dtype=dt)
            data = _extract_valid_bits(raw_scalars, valid_bits, bit_shift, valid_bits_signed)
            inferred_units = "counts" if dt.kind in "iu" else "native"
        else:
            data = np.asarray(raw_data).reshape(-1)
            inferred_units = "counts" if data.dtype.kind in "iu" else "native"
            data = _extract_valid_bits(data, valid_bits, bit_shift, valid_bits_signed)

        data = np.asarray(data).reshape(-1)
        if data.size == 0:
            raise ValueError("IQ data is empty")
        if data.size % 2 != 0:
            raise ValueError("flat IQ data must contain an even number of scalar values")

        if layout == "iq":
            i, q = data[0::2], data[1::2]
        elif layout == "qi":
            q, i = data[0::2], data[1::2]
        elif layout == "planar":
            half = data.size // 2
            i, q = data[:half], data[half:]
        else:
            raise ValueError(f"unsupported flat layout: {layout}")

        i = np.asarray(i, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

    if i.size == 0 or q.size == 0:
        raise ValueError("no IQ samples were decoded")
    if i.size != q.size:
        raise ValueError(f"I/Q length mismatch: I={i.size}, Q={q.size}")
    if not np.all(np.isfinite(i)) or not np.all(np.isfinite(q)):
        raise ValueError("decoded IQ contains NaN or infinity")

    z = i + 1j * q
    n = int(z.size)

    fs = None
    ts = None
    if sample_rate is not None:
        fs = float(sample_rate)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("sample_rate must be a positive finite number")
        ts = 1.0 / fs

    if sample_timestamps is not None:
        sample_timestamps = np.asarray(sample_timestamps).reshape(-1)
        if sample_timestamps.size != n:
            raise ValueError("sample_timestamps length must match sample_count")
        if sample_timestamps.size > 1 and np.any(np.diff(sample_timestamps) < 0):
            raise ValueError("sample_timestamps must be nondecreasing")

    if adc_resolution is None and valid_bits is not None:
        adc_resolution = int(valid_bits)
    if adc_resolution is not None:
        adc_resolution = int(adc_resolution)
        if adc_resolution <= 0:
            raise ValueError("adc_resolution must be positive")

    if adc_full_scale is not None:
        adc_full_scale = float(adc_full_scale)
        if not np.isfinite(adc_full_scale) or adc_full_scale <= 0:
            raise ValueError("adc_full_scale must be a positive finite value")

    return {
        "i": i,
        "q": q,
        "complex_samples": z.astype(np.complex128, copy=False),
        "sample_rate": fs,
        "sampling_period": ts,
        "sample_count": n,
        "timestamp": timestamp,
        "timestamp_unit": timestamp_unit,
        "sample_timestamps": sample_timestamps,
        "adc_resolution": adc_resolution,
        "adc_full_scale": adc_full_scale,
        "sample_units": inferred_units,
        "quality_flags": set() if quality_flags is None else set(quality_flags),
        "metadata": {} if metadata is None else copy.deepcopy(metadata),
    }


__all__ = ["parse_iq"]


def main() -> None:
    print("=" * 60)
    print("  parse_iq: Demonstration")
    print("=" * 60)
    res = parse_iq(np.exp(1j * 2 * np.pi * 50e3 * (np.arange(1024)/1e6)), sample_rate=1e6, layout='complex')
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
