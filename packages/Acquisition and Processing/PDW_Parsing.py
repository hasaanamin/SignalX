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

import struct
from ._common import _normalize_struct_schema, _byteorder_prefix


def parse_pdw(
    raw_data,
    schema,
    *,
    byteorder="little",
    offset_bytes=0,
    count=None,
    field_map=None,
):
    """Parse fixed-size PDW records using a sender-defined binary schema.

    A schema entry is one of:
        (name, struct_format)
        (name, struct_format, scale)
        (name, struct_format, scale, offset)

    Example:
        [
            ("toa_ns", "Q"),
            ("pulse_width_ns", "I"),
            ("rf_frequency_hz", "Q"),
            ("pulse_amplitude", "f"),
            ("peak_amplitude", "f"),
            ("phase_deg", "f"),
            ("rise_time_ns", "I"),
            ("fall_time_ns", "I"),
            ("receiver_id", "H"),
            ("channel_id", "H"),
            ("quality_flags", "I"),
        ]

    field_map optionally maps canonical output names to schema names, e.g.
        {"toa": "toa_ns", "pulse_width": "pulse_width_ns"}

    Exact PDW layouts are device/protocol-specific; the schema is therefore
    intentionally explicit rather than guessed.
    """
    if not isinstance(raw_data, (bytes, bytearray, memoryview)):
        raise TypeError("raw_data must be bytes-like")
    if int(offset_bytes) < 0:
        raise ValueError("offset_bytes must be >= 0")

    fields = _normalize_struct_schema(schema)
    prefix = _byteorder_prefix(byteorder)
    fmt = prefix + "".join(field[1] for field in fields)
    parser = struct.Struct(fmt)

    # Require one unpacked value per named field. This avoids ambiguous schemas.
    test_values = parser.unpack(bytes(parser.size))
    if len(test_values) != len(fields):
        raise ValueError(
            "each PDW schema field must decode to exactly one value; "
            "do not use padding-only or multi-value struct codes as named fields"
        )

    buf = memoryview(raw_data)
    if offset_bytes > len(buf):
        raise ValueError("offset_bytes exceeds input size")
    buf = buf[offset_bytes:]

    if count is None:
        if len(buf) % parser.size != 0:
            raise ValueError(
                f"PDW payload length {len(buf)} is not a multiple of record size {parser.size}"
            )
        count = len(buf) // parser.size
    else:
        count = int(count)
        if count < 0 or count * parser.size > len(buf):
            raise ValueError("invalid PDW record count")

    records = []
    for index in range(count):
        values = parser.unpack_from(buf, index * parser.size)
        raw_record = {}
        for (name, _fmt, scale, offset), value in zip(fields, values):
            if scale is not None:
                value = value * scale
            if offset is not None:
                value = value + offset
            raw_record[name] = value

        if field_map:
            record = {"raw_fields": raw_record}
            for canonical_name, source_name in field_map.items():
                record[canonical_name] = raw_record.get(source_name)
        else:
            record = raw_record
        records.append(record)

    return records


__all__ = ["parse_pdw"]


def main() -> None:
    print("=" * 60)
    print("  parse_pdw: Demonstration")
    print("=" * 60)
    raw_bytes = bytes(48)
    schema = [("toa_ns", "Q"), ("pulse_width_ns", "I")]
    res = parse_pdw(raw_bytes, schema=schema)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
