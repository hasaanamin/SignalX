"""Launch the HDF5 RF pipeline desktop test bench from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signal_processing.ui.hdf5_app import main

if __name__ == "__main__":
    raise SystemExit(main())
