"""Automated builder for srsran_fec.dll using MSVC or GCC.

This script compiles the pinned srsRAN 4G C Turbo decoder, rate matcher,
and CRC modules into a high-performance Windows x64 DLL.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import sys


def build_dll(output_path: Path = None) -> Path:
    native_dir = Path(__file__).resolve().parent
    if output_path is None:
        output_name = "srsran_fec.dll" if sys.platform == "win32" else "srsran_fec.so"
        output_path = native_dir / output_name

    sources = [
        "srsran_fec_adapter.c",
        "crc.c",
        "cbsegm.c",
        "turbocoder.c",
        "turbodecoder.c",
        "turbodecoder_gen.c",
        "tc_interl_lte.c",
        "rm_turbo.c",
        "bit.c",
    ]
    src_paths = [str(native_dir / s) for s in sources]
    src_args = " ".join(f'"{s}"' for s in src_paths)

    # 1. Try GCC / Clang if available (works on Linux, macOS, MinGW)
    import shutil
    cc = shutil.which("gcc") or shutil.which("clang")
    if sys.platform != "win32" and cc:
        cmd = [
            cc, "-shared", "-fPIC", "-O2", "-std=c11", "-DSRSRAN_LTE_ONLY",
            f"-I{native_dir}", f"-I{native_dir / 'srsran'}",
            *src_paths,
            "-o", str(output_path)
        ]
        res = subprocess.run(cmd, cwd=native_dir, capture_output=True, text=True)
        if res.returncode == 0:
            return output_path

    # 2. On Windows, try MSVC VsDevCmd
    vsdev_paths = [
        Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"),
    ]
    vsdev = next((p for p in vsdev_paths if p.is_file()), None)

    if vsdev:
        cmd = (
            f'call "{vsdev}" -arch=x64 >nul && '
            f'cl /nologo /LD /O2 /std:c11 /DSRSRAN_LTE_ONLY /I"{native_dir}" /I"{native_dir / "srsran"}" '
            f'{src_args} /link /OUT:"{output_path}"'
        )
        res = subprocess.run(f'cmd.exe /d /s /c "{cmd}"', cwd=native_dir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"MSVC build failed with code {res.returncode}:\n{res.stdout}\n{res.stderr}")
        # Clean temporary link artifacts
        for ext in [".obj", ".lib", ".exp"]:
            for f in native_dir.glob(f"*{ext}"):
                try:
                    f.unlink()
                except OSError:
                    pass
        return output_path

    # 3. Fallback: try GCC on Windows if present
    if cc:
        cmd = [
            cc, "-shared", "-fPIC", "-O2", "-std=c11", "-DSRSRAN_LTE_ONLY",
            f"-I{native_dir}", f"-I{native_dir / 'srsran'}",
            *src_paths,
            "-o", str(output_path)
        ]
        res = subprocess.run(cmd, cwd=native_dir, capture_output=True, text=True)
        if res.returncode == 0:
            return output_path

    raise RuntimeError("No suitable C compiler (MSVC VsDevCmd.bat, gcc, or clang) found to compile srsran_fec.")


if __name__ == "__main__":
    out = build_dll()
    print(f"Successfully compiled srsRAN FEC DLL at: {out}")
