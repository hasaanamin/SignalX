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

from .Sample_Rate import handle_sample_rate

__all__ = ["handle_sample_rate"]

import numpy as np


def main() -> None:
    print("=" * 60)
    print("  handle_sample_rate: Demonstration")
    print("=" * 60)
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * (np.arange(1024) / 1e6))
    res = handle_sample_rate(sample_iq, sample_rate=1e6)
    print(f"Result: {res}")
    print("=" * 60)


if __name__ == "__main__":
    main()
