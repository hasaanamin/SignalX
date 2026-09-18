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
from .Acquisition import parse_iq
from .PDW_Parsing import parse_pdw


def parse_iqdw(
    raw_data,
    header_schema,
    *,
    iq_options,
    byteorder="little",
    header_offset_bytes=0,
    field_map=None,
    payload_length_field=None,
):
    """Parse an IQDW as [fixed binary header][IQ payload].

    iq_options are forwarded to parse_iq(), for example:
        {"layout": "IQ", "dtype": "int16", "byteorder": "little"}

    field_map maps canonical metadata names to header names. Recognized canonical
    names are: sample_rate, timestamp, sample_count, adc_resolution,
    adc_full_scale, timestamp_unit.
    """
    if not isinstance(raw_data, (bytes, bytearray, memoryview)):
        raise TypeError("raw_data must be bytes-like")
    if int(header_offset_bytes) < 0:
        raise ValueError("header_offset_bytes must be >= 0")

    fields = _normalize_struct_schema(header_schema)
    prefix = _byteorder_prefix(byteorder)
    fmt = prefix + "".join(field[1] for field in fields)
    header_struct = struct.Struct(fmt)

    start = int(header_offset_bytes)
    end = start + header_struct.size
    if end > len(raw_data):
        raise ValueError("IQDW is shorter than its declared header")

    header = parse_pdw(
        memoryview(raw_data)[start:end],
        header_schema,
        byteorder=byteorder,
        count=1,
    )[0]

    mapping = {
        "sample_rate": "sample_rate_hz" if "sample_rate_hz" in header else "sample_rate",
        "timestamp": "timestamp_ns" if "timestamp_ns" in header else "timestamp",
        "sample_count": "sample_count",
        "adc_resolution": "adc_resolution",
        "adc_full_scale": "adc_full_scale",
        "timestamp_unit": "timestamp_unit",
    }
    if field_map:
        mapping.update(field_map)

    payload = memoryview(raw_data)[end:]
    if payload_length_field is not None:
        if payload_length_field not in header:
            raise ValueError(f"payload length field '{payload_length_field}' not found in header")
        declared_payload_bytes = int(header[payload_length_field])
        if declared_payload_bytes < 0 or declared_payload_bytes > len(payload):
            raise ValueError("declared IQDW payload length is invalid")
        if declared_payload_bytes != len(payload):
            raise ValueError(
                f"IQDW payload length mismatch: header={declared_payload_bytes}, actual={len(payload)}"
            )

    options = dict(iq_options)
    for canonical in ("sample_rate", "timestamp", "adc_resolution", "adc_full_scale", "timestamp_unit"):
        source = mapping.get(canonical)
        if canonical not in options and source in header:
            options[canonical] = header[source]

    # Sensible timestamp-unit inference for a conventional *_ns header field.
    if "timestamp_unit" not in options and mapping.get("timestamp") == "timestamp_ns":
        options["timestamp_unit"] = "ns"

    record = parse_iq(payload, **options)

    count_field = mapping.get("sample_count")
    if count_field in header:
        expected = int(header[count_field])
        if record["sample_count"] != expected:
            raise ValueError(
                f"IQDW sample count mismatch: header={expected}, decoded={record['sample_count']}"
            )

    record["header"] = header
    record["record_type"] = "IQDW"
    return record


__all__ = ["parse_iqdw"]


def main() -> None:
    print("=" * 60)
    print("  parse_iqdw: Demonstration")
    print("=" * 60)
    dummy_payload = struct.pack("<II", 0x12345678, 14) + bytes(56)
    schema = [("magic", "I"), ("sample_count", "I")]
    res = parse_iqdw(dummy_payload, schema, iq_options={"layout": "IQ", "dtype": "int16"})
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
