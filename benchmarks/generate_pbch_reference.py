"""Generate BCH fixtures using unchanged pinned srsRAN C encoder/rate matcher.

External AGPL reference sources remain under ignored output/, with their notices.
Only numeric vectors and provenance are frozen under tests/fixtures. Production
does not depend on or import this tool, the native reference, or an encoder.
"""
import argparse
import binascii
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np

from benchmarks.generate_lte_reference import COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("output/lte_pbch_20261019/reference"))
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/lte_bch_reference.npz"))
    args = parser.parse_args()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    paths = {"convcoder.c": "lib/src/phy/fec/convolutional/convcoder.c",
             "parity.h": "lib/src/phy/fec/convolutional/parity.h",
             "rm_conv.c": "lib/src/phy/fec/turbo/rm_conv.c"}
    provenance = {}
    for filename, path in paths.items():
        url = f"https://raw.githubusercontent.com/srsran/srsRAN_4G/{COMMIT}/{path}"
        source = urllib.request.urlopen(url, timeout=30).read()
        (build/filename).write_bytes(source)
        provenance[filename] = {"url": url, "sha256": hashlib.sha256(source).hexdigest()}
    root = Path(__file__).resolve().parent/"lte_reference"
    command = (f'call "C:/Program Files/Microsoft Visual Studio/18/Community/Common7/Tools/VsDevCmd.bat" -arch=x64 >nul '
               f'&& cl /nologo /LD /O2 /I"{root}" /I"{build}" "{root / "bch_adapter.c"}" '
               f'/link /OUT:"{build / "bch_reference.dll"}"')
    subprocess.run('cmd.exe /d /s /c "'+command+'"', cwd=build, check=True)
    dll = ctypes.CDLL(str(build/"bch_reference.dll"))
    ptr = ctypes.POINTER(ctypes.c_uint8)
    dll.reference_bch.argtypes = [ptr, ptr, ptr]
    dll.reference_bch.restype = None
    rng = np.random.default_rng(261019)
    payloads = rng.integers(0, 2, (30, 24), dtype=np.uint8)
    # Exercise each legal bandwidth, SFN MSB extremes and all CRC masks.
    payloads[0] = 0
    payloads[1] = 1
    payloads[1, :3] = [1, 0, 1]
    for i in range(30):
        payloads[i, :3] = [(i % 6 >> k) & 1 for k in (2, 1, 0)]
    inputs, encoded, matched, ports_array = [], [], [], []
    for payload in payloads:
        checksum = binascii.crc_hqx(np.packbits(payload).tobytes(), 0)
        crc_bits = np.array([(checksum >> k) & 1 for k in range(15, -1, -1)], np.uint8)
        for ports in (1, 2, 4):
            mask = np.zeros(16, np.uint8) if ports == 1 else (np.ones(16, np.uint8) if ports == 2 else np.tile([0, 1], 8).astype(np.uint8))
            bits = np.r_[payload, crc_bits ^ mask]
            code, rate = np.empty(120, np.uint8), np.empty(1920, np.uint8)
            dll.reference_bch(bits.ctypes.data_as(ptr), code.ctypes.data_as(ptr), rate.ctypes.data_as(ptr))
            inputs.append(bits)
            encoded.append(code)
            matched.append(rate)
            ports_array.append(ports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, bits=inputs, encoded=encoded, matched=matched, ports=ports_array)
    manifest = {"commit": COMMIT, "sources": provenance,
                "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "method": "Unchanged compiled srsran_convcoder_encode and srsran_rm_conv_tx; CRC independently binascii.crc_hqx",
                "reference_license": "AGPL-3.0; original notices retained under output; no reference C in production",
                "scope": "90 BCH coding vectors, not a complete external LTE transmitter or RF recording"}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
