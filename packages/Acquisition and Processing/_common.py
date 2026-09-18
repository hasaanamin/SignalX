from __future__ import annotations
import copy
import numpy as np

_TIME_UNIT_TO_SECONDS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "µs": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
}


def _time_scale(unit: str) -> float:
    try:
        return _TIME_UNIT_TO_SECONDS[str(unit).lower()]
    except KeyError as exc:
        raise ValueError(
            "timestamp_unit must be one of: s, ms, us, ns, ps"
        ) from exc


def _as_iq_array(iq_data) -> np.ndarray:
    """Convert canonical record / complex array / (I,Q) into complex128 1-D IQ."""
    if isinstance(iq_data, dict):
        if iq_data.get("complex_samples") is not None:
            x = np.asarray(iq_data["complex_samples"])
        elif iq_data.get("i") is not None and iq_data.get("q") is not None:
            i = np.asarray(iq_data["i"])
            q = np.asarray(iq_data["q"])
            if i.shape != q.shape:
                raise ValueError("I/Q shape mismatch")
            x = i.astype(np.float64) + 1j * q.astype(np.float64)
        else:
            raise ValueError(
                "IQ record must contain 'complex_samples' or both 'i' and 'q'"
            )
    elif isinstance(iq_data, tuple) and len(iq_data) == 2:
        i = np.asarray(iq_data[0])
        q = np.asarray(iq_data[1])
        if i.shape != q.shape:
            raise ValueError("I/Q shape mismatch")
        x = i.astype(np.float64) + 1j * q.astype(np.float64)
    elif isinstance(iq_data, (list, tuple)) and len(iq_data) > 0 and isinstance(iq_data[0], (tuple, list)) and len(iq_data[0]) == 2:
        i = np.array([p[0] for p in iq_data], dtype=np.float64)
        q = np.array([p[1] for p in iq_data], dtype=np.float64)
        x = i + 1j * q
    else:
        x = np.asarray(iq_data)
        if not np.iscomplexobj(x):
            raise TypeError(
                "Expected complex IQ samples, a canonical IQ record, or (I, Q). "
                "Use parse_iq() for raw/interleaved data."
            )

    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if x.size == 0:
        raise ValueError("IQ data must contain at least one sample")
    if not np.all(np.isfinite(x.real)) or not np.all(np.isfinite(x.imag)):
        raise ValueError("IQ data contains NaN or infinity")
    return x


def _append_processing_metadata(record: dict, entry: dict) -> None:
    metadata = copy.deepcopy(record.get("metadata", {}))
    processing = list(metadata.get("processing", []))
    processing.append(entry)
    metadata["processing"] = processing
    record["metadata"] = metadata


def _return_like(
    original,
    samples: np.ndarray,
    *,
    sample_rate=None,
    timestamp=None,
    timestamp_is_set=False,
    sample_timestamps="__KEEP__",
    sample_units=None,
    processing_entry=None,
):
    """Return ndarray for ndarray input, updated canonical record for record input."""
    y = np.asarray(samples, dtype=np.complex128).reshape(-1)

    if not isinstance(original, dict):
        return y

    out = copy.deepcopy(original)
    out["complex_samples"] = y
    out["i"] = y.real.copy()
    out["q"] = y.imag.copy()
    out["sample_count"] = int(y.size)

    fs = sample_rate if sample_rate is not None else out.get("sample_rate")
    if fs is not None:
        fs = float(fs)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("sample_rate must be a positive finite number")
        out["sample_rate"] = fs
        out["sampling_period"] = 1.0 / fs

    if timestamp_is_set:
        out["timestamp"] = timestamp

    if sample_timestamps != "__KEEP__":
        out["sample_timestamps"] = sample_timestamps

    if sample_units is not None:
        out["sample_units"] = sample_units

    if processing_entry is not None:
        _append_processing_metadata(out, processing_entry)

    return out


def _extract_valid_bits(values, valid_bits=None, bit_shift=0, signed=True):
    values = np.asarray(values)
    if valid_bits is None:
        return values
    if values.dtype.kind not in "iu":
        raise ValueError("valid_bits can only be used with integer storage")

    storage_bits = values.dtype.itemsize * 8
    valid_bits = int(valid_bits)
    bit_shift = int(bit_shift)
    if not 1 <= valid_bits <= storage_bits:
        raise ValueError("valid_bits must fit inside the storage type")
    if bit_shift < 0 or bit_shift + valid_bits > storage_bits:
        raise ValueError("bit_shift + valid_bits exceeds storage width")

    storage_mask = (1 << storage_bits) - 1
    if values.dtype.kind == "i":
        u = values.astype(np.int64).astype(np.uint64) & storage_mask
    else:
        u = values.astype(np.uint64) & storage_mask

    mask = (1 << valid_bits) - 1
    u = (u >> bit_shift) & mask

    if not signed:
        return u

    sign_bit = 1 << (valid_bits - 1)
    full_range = 1 << valid_bits
    out = u.astype(np.int64)
    return np.where((u & sign_bit) != 0, out - full_range, out)


def _unpack_contiguous_bits(buffer, bits, *, signed=True, bit_order="lsb"):
    bits = int(bits)
    if not 1 <= bits <= 32:
        raise ValueError("packed_bits must be between 1 and 32")

    raw = memoryview(buffer).cast("B")
    if len(raw) == 0:
        raise ValueError("packed payload is empty")

    total_bits = len(raw) * 8
    if total_bits % bits != 0:
        raise ValueError(
            "packed payload does not contain a whole number of packed values"
        )

    count = total_bits // bits
    out = np.empty(count, dtype=np.int64)
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    full_range = 1 << bits

    if bit_order == "lsb":
        acc = 0
        acc_bits = 0
        index = 0
        for byte in raw:
            acc |= int(byte) << acc_bits
            acc_bits += 8
            while acc_bits >= bits:
                value = acc & mask
                acc >>= bits
                acc_bits -= bits
                if signed and (value & sign_bit):
                    value -= full_range
                out[index] = value
                index += 1

    elif bit_order == "msb":
        acc = 0
        acc_bits = 0
        index = 0
        for byte in raw:
            acc = (acc << 8) | int(byte)
            acc_bits += 8
            while acc_bits >= bits:
                shift = acc_bits - bits
                value = (acc >> shift) & mask
                acc &= (1 << shift) - 1 if shift else 0
                acc_bits -= bits
                if signed and (value & sign_bit):
                    value -= full_range
                out[index] = value
                index += 1
    else:
        raise ValueError("packed_bit_order must be 'lsb' or 'msb'")

    return out


def _normalize_struct_schema(schema):
    if not schema:
        raise ValueError("schema must contain at least one field")

    normalized = []
    for field in schema:
        if isinstance(field, dict):
            name = field["name"]
            fmt = field["format"]
            scale = field.get("scale")
            offset = field.get("offset")
        else:
            if len(field) not in (2, 3, 4):
                raise ValueError(
                    "schema entries must be (name, format[, scale[, offset]])"
                )
            name, fmt = field[0], field[1]
            scale = field[2] if len(field) >= 3 else None
            offset = field[3] if len(field) == 4 else None
        normalized.append((str(name), str(fmt), scale, offset))
    return normalized


def _byteorder_prefix(byteorder: str) -> str:
    mapping = {
        "little": "<",
        "big": ">",
        "network": "!",
        "native": "=",
    }
    try:
        return mapping[str(byteorder).lower()]
    except KeyError as exc:
        raise ValueError(
            "byteorder must be 'little', 'big', 'network', or 'native'"
        ) from exc
