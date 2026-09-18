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

from .PDW_Parsing import parse_pdw

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
