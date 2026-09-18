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

from .IQDW_Parsing import parse_iqdw

__all__ = ["parse_iqdw"]


def main() -> None:
    print("=" * 60)
    print("  parse_iqdw: Demonstration")
    print("=" * 60)
    import struct
    dummy_payload = struct.pack("<II", 0x12345678, 14) + bytes(56)
    schema = [("magic", "I"), ("sample_count", "I")]
    res = parse_iqdw(dummy_payload, schema, iq_options={"layout": "IQ", "dtype": "int16"})
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
