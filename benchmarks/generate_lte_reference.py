"""Generate independent frozen sequence vectors using pinned original srsRAN C.

Windows-only reference-generation tool requiring existing MSVC Build Tools.
The production package and ordinary tests require neither a compiler nor srsRAN.
Downloaded AGPL reference source retains its original licence notice under output/.
"""
import argparse
import cmath
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import urllib.request

import numpy as np


COMMIT = "bef8680d5f9714f3e040e6f9cbc88d7888439b6d"
URL = f"https://raw.githubusercontent.com/srsran/srsRAN_4G/{COMMIT}/lib/src/phy/sync/gen_sss.c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/lte_sync_sequences.npz"))
    parser.add_argument("--build-dir", type=Path, default=Path("output/lte_sync_20261019/reference"))
    parser.add_argument("--vsdevcmd", type=Path, default=Path("C:/Program Files/Microsoft Visual Studio/18/Community/Common7/Tools/VsDevCmd.bat"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = urllib.request.urlopen(URL, timeout=30).read()
    (build/"gen_sss.c").write_bytes(source)
    command = (f'call "{args.vsdevcmd}" -arch=x64 >nul && cl /nologo /LD /O2 '
               f'/I"{root / "lte_reference"}" /I"{build}" '
               f'"{root / "lte_reference/adapter.c"}" /link /OUT:"{build / "reference.dll"}"')
    # cmd uses its own quoting, not the CRT argv escaping applied to a list.
    subprocess.run('cmd.exe /d /s /c "'+command+'"', cwd=build, check=True)
    reference = ctypes.CDLL(str(build/"reference.dll"))
    fn = reference.reference_sss
    pointer = ctypes.POINTER(ctypes.c_float)
    fn.argtypes = [pointer, pointer, ctypes.c_uint]
    fn.restype = None
    sss = np.empty((504, 2, 62), np.int8)
    for pci in range(504):
        first, second = np.zeros(62, np.float32), np.zeros(62, np.float32)
        fn(first.ctypes.data_as(pointer), second.ctypes.data_as(pointer), pci)
        assert np.all(np.isin(first, [-1., 1.])) and np.all(np.isin(second, [-1., 1.]))
        sss[pci, 0], sss[pci, 1] = first, second
    # Separate scalar standard-equation oracle for the three PSS sequences.
    pss = np.empty((3, 62), np.complex128)
    for sector, root_value in enumerate((25, 29, 34)):
        for n in range(62):
            k = n if n < 31 else n+1
            pss[sector, n] = cmath.exp(-1j*math.pi*root_value*k*(k+1)/63)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, pss=pss, sss=sss)
    manifest = {"reference_commit": COMMIT, "reference_url": URL,
                "reference_source_sha256": hashlib.sha256(source).hexdigest(),
                "source_license": "AGPL-3.0; downloaded reference source retains original notice; not distributed in production wheel",
                "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "sss_method": "Compiled unchanged srsran_sss_generate, all 504 PCI / both half frames",
                "pss_method": "Independent scalar cmath standard-equation implementation",
                "not_verified": "Complete LTE transmitter/receiver, PBCH/MIB or RF hardware"}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
