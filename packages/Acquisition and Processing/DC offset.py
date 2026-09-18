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

import importlib
import numpy as np
_canonical_dc_removal = importlib.import_module(".DC Removal", package=__package__)



def remove_dc(iq_data, **kwargs):
    """
    Compatibility wrapper delegating directly to canonical DC Removal.remove_dc.
    Preserves legacy tuple return (corrected_iq, dc_offset) if called with legacy signature.
    """
    if "return_info" not in kwargs:
        out, info = _canonical_dc_removal.remove_dc(iq_data, return_info=True, **kwargs)
        return out, info["dc_offset"]
    return _canonical_dc_removal.remove_dc(iq_data, **kwargs)


__all__ = ["remove_dc"]



def main() -> None:
    print("=" * 60)
    print("  remove_dc: Demonstration")
    print("=" * 60)
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * (np.arange(1024) / 1e6)) + 0.5
    res = remove_dc(sample_iq)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
