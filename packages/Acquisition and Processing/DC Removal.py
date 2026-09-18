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


def remove_dc(iq_data, *, method="mean", reference_slice=None, return_info=False):
    """Remove constant I/Q DC offset from a block."""
    x = _as_iq_array(iq_data)
    method = str(method).lower()

    if method == "mean":
        offset = np.mean(x)
    elif method == "median":
        offset = np.median(x.real) + 1j * np.median(x.imag)
    elif method == "reference":
        if reference_slice is None:
            raise ValueError("reference_slice is required for method='reference'")
        ref = x[slice(*reference_slice)] if isinstance(reference_slice, (tuple, list)) else x[reference_slice]
        if np.asarray(ref).size == 0:
            raise ValueError("reference_slice selects no samples")
        offset = np.mean(ref)
    else:
        raise ValueError("method must be 'mean', 'median', or 'reference'")

    y = x - offset
    info = {"operation": "dc_removal", "method": method, "dc_offset": complex(offset)}
    out = _return_like(iq_data, y, processing_entry=info)
    return (out, info) if return_info else out


__all__ = ["remove_dc"]


def main() -> None:
    print("=" * 60)
    print("  remove_dc: Demonstration")
    print("=" * 60)
    res = remove_dc(np.exp(1j * 2 * np.pi * 50e3 * (np.arange(1024)/1e6)) + 0.5)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
