"""LTE 3GPP Turbo Code Encoder & Decoder (TS 36.212 §§5.1.2, 5.1.3).

Uses the official srsRAN 4G C library for high-performance native Turbo decoding,
rate matching/unmatching, and CRC-24A validation:
  - C ABI bindings to srsRAN 4G Turbo Decoder (Max-Log-MAP SISO).
  - C ABI bindings to srsRAN 4G Sub-block Interleaver & Circular Buffer Rate Matcher.
  - C ABI bindings to srsRAN 4G CRC-24A engine.
  - Transparent Python/Numba fallback for portable operation.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# 3GPP TS 36.212 Table 5.1.3-3: QPP Interleaver Parameters (K, f1, f2)
QPP_TABLE: Dict[int, Tuple[int, int]] = {
    40: (3, 10), 48: (7, 12), 56: (19, 42), 64: (7, 16), 72: (7, 18), 80: (11, 20),
    88: (5, 22), 96: (11, 24), 104: (7, 26), 112: (41, 84), 120: (103, 90), 128: (15, 32),
    136: (9, 34), 144: (17, 108), 152: (9, 38), 160: (21, 120), 168: (101, 84), 176: (21, 44),
    184: (57, 46), 192: (23, 48), 200: (13, 50), 208: (27, 52), 216: (11, 36), 224: (27, 56),
    232: (85, 58), 240: (29, 60), 248: (33, 62), 256: (15, 32), 264: (17, 198), 272: (33, 68),
    280: (103, 210), 288: (19, 36), 296: (19, 74), 304: (37, 76), 312: (19, 78), 320: (21, 120),
    328: (21, 82), 336: (115, 84), 344: (193, 86), 352: (21, 44), 360: (133, 90), 368: (81, 46),
    376: (45, 94), 384: (23, 48), 392: (243, 98), 400: (151, 40), 408: (155, 102), 416: (25, 52),
    424: (51, 106), 432: (47, 72), 440: (91, 110), 448: (29, 168), 456: (29, 114), 464: (247, 58),
    472: (29, 118), 480: (89, 180), 488: (91, 122), 496: (157, 62), 504: (55, 84), 512: (31, 64),
    528: (17, 66), 544: (35, 68), 560: (227, 420), 576: (65, 96), 592: (19, 74), 608: (37, 76),
    624: (41, 234), 640: (39, 80), 656: (185, 82), 672: (43, 252), 688: (21, 86), 704: (155, 44),
    720: (79, 120), 736: (139, 92), 752: (23, 94), 768: (217, 48), 784: (25, 98), 800: (17, 80),
    816: (127, 102), 832: (25, 52), 848: (239, 106), 864: (53, 216), 880: (109, 110), 896: (83, 112),
    912: (89, 114), 928: (29, 58), 944: (29, 118), 960: (59, 120), 1024: (31, 64), 1056: (35, 132),
    1088: (207, 136), 1120: (69, 140), 1152: (17, 72), 1184: (137, 74), 1216: (39, 76), 1248: (41, 156),
    1280: (123, 80), 1312: (41, 82), 1344: (43, 84), 1376: (45, 86), 1408: (45, 44), 1440: (47, 90),
    1472: (47, 92), 1504: (47, 94), 1536: (49, 96), 1568: (49, 98), 1600: (51, 100), 1632: (51, 102),
    1664: (53, 104), 1696: (53, 106), 1728: (55, 108), 1760: (55, 110), 1792: (57, 112), 1824: (57, 114),
    1856: (59, 116), 1888: (59, 118), 1920: (61, 120), 1984: (63, 124), 2048: (65, 128), 2112: (67, 132),
    2176: (69, 136), 2240: (71, 140), 2304: (73, 144), 2368: (75, 148), 2432: (77, 152), 2496: (79, 156),
    2560: (81, 160), 2624: (83, 164), 2688: (85, 168), 2752: (87, 172), 2816: (89, 176), 2880: (91, 180),
    2944: (93, 184), 3008: (95, 188), 3072: (97, 192), 3136: (99, 196), 3200: (101, 200), 3264: (103, 204),
    3328: (105, 208), 3392: (107, 212), 3456: (109, 216), 3520: (111, 220), 3584: (113, 224), 3648: (115, 228),
    3712: (117, 232), 3776: (119, 236), 3840: (121, 240), 3904: (123, 244), 3968: (125, 248), 4032: (127, 252),
    4096: (129, 256), 4160: (131, 260), 4224: (133, 264), 4288: (135, 268), 4352: (137, 272), 4416: (139, 276),
    4480: (141, 280), 4544: (143, 284), 4608: (145, 288), 4672: (147, 292), 4736: (149, 296), 4800: (151, 300),
    4864: (153, 304), 4928: (155, 308), 4992: (157, 312), 5056: (159, 316), 5120: (161, 320), 5184: (163, 324),
    5248: (165, 328), 5312: (167, 332), 5376: (169, 336), 5440: (171, 340), 5504: (173, 344), 5568: (175, 348),
    5632: (177, 352), 5696: (179, 356), 5760: (181, 360), 5824: (183, 364), 5888: (185, 368), 5952: (187, 372),
    6016: (189, 376), 6080: (191, 380), 6144: (193, 384),
}


# Precomputed Trellis Tables for RSC (g0=13_8, g1=15_8)
NEXT_STATE = np.array([
    [0, 4],  # State 0: u=0 -> 0, u=1 -> 4
    [4, 0],  # State 1: u=0 -> 4, u=1 -> 0
    [5, 1],  # State 2: u=0 -> 5, u=1 -> 1
    [1, 5],  # State 3: u=0 -> 1, u=1 -> 5
    [2, 6],  # State 4: u=0 -> 2, u=1 -> 6
    [6, 2],  # State 5: u=0 -> 6, u=1 -> 2
    [7, 3],  # State 6: u=0 -> 7, u=1 -> 3
    [3, 7],  # State 7: u=0 -> 3, u=1 -> 7
], dtype=np.int32)

OUT_PARITY = np.array([
    [0, 1],  # State 0: u=0 -> 0, u=1 -> 1
    [1, 0],  # State 1: u=0 -> 1, u=1 -> 0
    [0, 1],  # State 2: u=0 -> 0, u=1 -> 1
    [1, 0],  # State 3: u=0 -> 1, u=1 -> 0
    [1, 0],  # State 4: u=0 -> 1, u=1 -> 0
    [0, 1],  # State 5: u=0 -> 0, u=1 -> 1
    [1, 0],  # State 6: u=0 -> 1, u=1 -> 0
    [0, 1],  # State 7: u=0 -> 0, u=1 -> 1
], dtype=np.int32)


# =============================================================================
# srsRAN Native C Library Binding
# =============================================================================

@lru_cache(maxsize=1)
def srsran_fec_library() -> Optional[ctypes.CDLL]:
    """Load the compiled native srsRAN C library (srsran_fec.dll or .so)."""
    native_dir = Path(__file__).resolve().parent / "native"
    dll_candidates = [
        native_dir / ("srsran_fec.dll" if sys.platform == "win32" else "srsran_fec.so"),
        native_dir / "srsran_fec.dll",
        native_dir / "srsran_fec.so",
        native_dir / "libsrsran_fec.so",
    ]
    dll_path = next((p for p in dll_candidates if p.is_file()), None)
    if dll_path is None:
        # Try compiling on the fly if builder exists
        builder_script = native_dir / "build_srsran.py"
        if builder_script.is_file():
            try:
                from .native.build_srsran import build_dll
                built = build_dll()
                if built and built.is_file():
                    dll_path = built
            except Exception:
                pass
    if dll_path is None or not dll_path.is_file():
        return None

    try:
        dll = ctypes.CDLL(str(dll_path))

        # Signatures
        dll.srsran_fec_crc24a_checksum.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32]
        dll.srsran_fec_crc24a_checksum.restype = ctypes.c_int

        dll.srsran_fec_crc24a_verify.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32]
        dll.srsran_fec_crc24a_verify.restype = ctypes.c_int

        dll.srsran_fec_turbo_encode.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32]
        dll.srsran_fec_turbo_encode.restype = ctypes.c_int

        dll.srsran_fec_turbo_decode.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32, ctypes.c_uint32]
        dll.srsran_fec_turbo_decode.restype = ctypes.c_int

        dll.srsran_fec_rm_turbo_tx.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32, ctypes.c_uint32]
        dll.srsran_fec_rm_turbo_tx.restype = ctypes.c_int

        dll.srsran_fec_rm_turbo_rx.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        dll.srsran_fec_rm_turbo_rx.restype = ctypes.c_int

        dll.srsran_fec_decode_tb.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)]
        dll.srsran_fec_decode_tb.restype = ctypes.c_int

        return dll
    except OSError:
        return None


def is_srsran_available() -> bool:
    """Return True if srsRAN native C shared library is loaded and active."""
    return srsran_fec_library() is not None


# =============================================================================
# CRC-24A Operations (srsRAN C with Pure-Python Fallback)
# =============================================================================

def crc24a(bits: np.ndarray) -> int:
    """Compute 3GPP CRC-24A over a bit sequence (TS 36.212 §5.1.1)."""
    lib = srsran_fec_library()
    if lib is not None:
        u8_arr = np.ascontiguousarray(bits, dtype=np.uint8)
        return int(lib.srsran_fec_crc24a_checksum(u8_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(u8_arr)))

    # Fallback polynomial calculation
    poly = 0x864CFB
    reg = 0
    for b in bits:
        fb = ((reg >> 23) ^ int(b)) & 1
        reg = (reg << 1) & 0xFFFFFF
        if fb:
            reg ^= poly
    return reg


def attach_crc24a(payload_bits: np.ndarray) -> np.ndarray:
    """Append 24-bit CRC-24A to payload bits."""
    c = crc24a(payload_bits)
    crc_bits = np.array([(c >> (23 - i)) & 1 for i in range(24)], dtype=np.uint8)
    return np.concatenate((payload_bits, crc_bits))


def verify_crc24a(coded_bits: np.ndarray) -> bool:
    """Verify CRC-24A remainder is zero."""
    if len(coded_bits) < 24:
        return False
    lib = srsran_fec_library()
    if lib is not None:
        u8_arr = np.ascontiguousarray(coded_bits, dtype=np.uint8)
        return bool(lib.srsran_fec_crc24a_verify(u8_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(u8_arr)))
    return crc24a(coded_bits) == 0


def get_qpp_indices(k_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return forward interleaver pi and reverse interleaver pi_inv for length K."""
    if k_len not in QPP_TABLE:
        candidates = [k for k in QPP_TABLE if k >= k_len]
        if not candidates:
            raise ValueError(f"Block length {k_len} exceeds max 3GPP Turbo size 6144")
        k_len = min(candidates)

    f1, f2 = QPP_TABLE[k_len]
    i = np.arange(k_len, dtype=np.int64)
    pi = ((f1 * i + f2 * (i ** 2)) % k_len).astype(np.int32)
    pi_inv = np.empty(k_len, dtype=np.int32)
    pi_inv[pi] = np.arange(k_len, dtype=np.int32)
    return pi, pi_inv


# =============================================================================
# Python / Numba Fallback SISO Decoder
# =============================================================================

@njit(fastmath=True)
def _siso_max_log_map(
    l_sys: np.ndarray,
    l_par: np.ndarray,
    l_a: np.ndarray,
    next_s: np.ndarray,
    out_p: np.ndarray,
) -> np.ndarray:
    """Vectorized / JIT-compiled Max-Log-MAP component decoder."""
    k_len = len(l_sys)

    alpha = np.full((k_len + 1, 8), -1e9, dtype=np.float64)
    alpha[0, 0] = 0.0

    for k in range(k_len):
        l_s_val = l_sys[k] + l_a[k]
        l_p_val = l_par[k]
        for s in range(8):
            a_curr = alpha[k, s]
            if a_curr <= -1e8:
                continue
            for u in range(2):
                ns = next_s[s, u]
                p = out_p[s, u]
                gamma = 0.5 * (1.0 - 2.0 * u) * l_s_val + 0.5 * (1.0 - 2.0 * p) * l_p_val
                metric = a_curr + gamma
                if metric > alpha[k + 1, ns]:
                    alpha[k + 1, ns] = metric

    beta = np.full((k_len + 1, 8), -1e9, dtype=np.float64)
    beta[k_len, :] = 0.0

    for k in range(k_len - 1, -1, -1):
        l_s_val = l_sys[k] + l_a[k]
        l_p_val = l_par[k]
        for s in range(8):
            for u in range(2):
                ns = next_s[s, u]
                p = out_p[s, u]
                gamma = 0.5 * (1.0 - 2.0 * u) * l_s_val + 0.5 * (1.0 - 2.0 * p) * l_p_val
                metric = beta[k + 1, ns] + gamma
                if metric > beta[k, s]:
                    beta[k, s] = metric

    l_ext = np.zeros(k_len, dtype=np.float64)
    for k in range(k_len):
        l_s_val = l_sys[k] + l_a[k]
        l_p_val = l_par[k]
        m0 = -1e9
        m1 = -1e9
        for s in range(8):
            a_curr = alpha[k, s]
            if a_curr <= -1e8:
                continue
            for u in range(2):
                ns = next_s[s, u]
                p = out_p[s, u]
                gamma = 0.5 * (1.0 - 2.0 * u) * l_s_val + 0.5 * (1.0 - 2.0 * p) * l_p_val
                tot = a_curr + gamma + beta[k + 1, ns]
                if u == 0:
                    if tot > m0:
                        m0 = tot
                else:
                    if tot > m1:
                        m1 = tot
        l_tot = m0 - m1
        l_ext[k] = l_tot - l_s_val

    return l_ext


@dataclass
class TurboDecodeResult:
    """Result of iterative Turbo decoding."""
    bits: np.ndarray                 # Hard-decision systematic bits (0 or 1)
    crc_valid: bool                  # True if CRC-24A passed
    iterations_used: int             # Number of half/full iterations before stop
    final_llr: np.ndarray            # Final soft LLR values


class TurboDecoder:
    """Iterative 3GPP Turbo Decoder backed by srsRAN C library with JIT fallback."""

    def __init__(self, k_len: int, max_iterations: int = 6) -> None:
        self.k_len = int(k_len)
        self.max_iterations = int(max_iterations)
        self.pi, self.pi_inv = get_qpp_indices(self.k_len)
        self._srsran = srsran_fec_library()

    def decode(
        self,
        l_sys: np.ndarray,
        l_par1: np.ndarray,
        l_par2: np.ndarray,
    ) -> TurboDecodeResult:
        """Decode systematic and parity LLRs using srsRAN C library.
        
        Args:
            l_sys: Length-K channel LLRs for systematic bits (positive for bit 0, or srsran polarity).
            l_par1: Length-K channel LLRs for encoder 1 parity bits.
            l_par2: Length-K channel LLRs for encoder 2 parity bits.
        """
        l_s = np.asarray(l_sys[: self.k_len], dtype=np.float32)
        l_p1 = np.asarray(l_par1[: self.k_len], dtype=np.float32)
        l_p2 = np.asarray(l_par2[: self.k_len], dtype=np.float32)

        # Fast path: check if uncoded hard decision already passes CRC
        # In srsRAN: LLR > 0 is bit 1, LLR < 0 is bit 0
        cand_bits_pos = (l_s > 0).astype(np.uint8)
        if verify_crc24a(cand_bits_pos):
            return TurboDecodeResult(bits=cand_bits_pos, crc_valid=True, iterations_used=0, final_llr=l_s)
        cand_bits_neg = (l_s < 0).astype(np.uint8)
        if verify_crc24a(cand_bits_neg):
            return TurboDecodeResult(bits=cand_bits_neg, crc_valid=True, iterations_used=0, final_llr=-l_s)

        # 1. Native srsRAN C library execution
        if self._srsran is not None and self.k_len in QPP_TABLE:
            # Interleave systematic and parity streams into srsRAN input format:
            # [s_0, p1_0, p2_0, s_1, p1_1, p2_1, ..., tail (12)]
            coded_len = 3 * self.k_len + 12
            srsran_llr = np.zeros(coded_len, dtype=np.float32)

            # Normalization scale factor for srsRAN 16-bit integer MAP decoder
            scale = 2.0 / max(1.0, float(np.mean(np.abs(l_s))))
            
            # Map standard convention (positive for bit 0) to srsRAN convention (positive for bit 1):
            srsran_llr[0 : 3 * self.k_len : 3] = -l_s * scale
            srsran_llr[1 : 3 * self.k_len : 3] = -l_p1 * scale
            srsran_llr[2 : 3 * self.k_len : 3] = -l_p2 * scale

            out_bits = np.zeros(self.k_len, dtype=np.uint8)
            iters = self._srsran.srsran_fec_turbo_decode(
                srsran_llr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                coded_len,
                out_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.k_len,
                self.max_iterations,
            )
            crc_valid = verify_crc24a(out_bits)
            if crc_valid or iters >= 0:
                return TurboDecodeResult(
                    bits=out_bits,
                    crc_valid=crc_valid,
                    iterations_used=max(1, iters),
                    final_llr=srsran_llr[0 : 3 * self.k_len : 3],
                )

        # 2. Portable JIT Fallback
        l_s64 = l_s.astype(np.float64)
        l_p1_64 = l_p1.astype(np.float64)
        l_p2_64 = l_p2.astype(np.float64)
        l_s_pi = l_s64[self.pi]
        l_a1 = np.zeros(self.k_len, dtype=np.float64)

        iters_used = 0
        final_l = l_s64.copy()

        for it in range(self.max_iterations):
            iters_used = it + 1
            l_e1 = _siso_max_log_map(l_s64, l_p1_64, l_a1, NEXT_STATE, OUT_PARITY)
            l_a2 = l_e1[self.pi]
            l_e2 = _siso_max_log_map(l_s_pi, l_p2_64, l_a2, NEXT_STATE, OUT_PARITY)
            l_a1 = l_e2[self.pi_inv]
            final_l = l_s64 + l_a1 + l_e1
            curr_bits = (final_l < 0).astype(np.uint8)

            if verify_crc24a(curr_bits):
                return TurboDecodeResult(
                    bits=curr_bits,
                    crc_valid=True,
                    iterations_used=iters_used,
                    final_llr=final_l,
                )

        curr_bits = (final_l < 0).astype(np.uint8)
        return TurboDecodeResult(
            bits=curr_bits,
            crc_valid=verify_crc24a(curr_bits),
            iterations_used=iters_used,
            final_llr=final_l,
        )


def turbo_encode(bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper 3GPP Turbo encoder: returns (systematic, parity1, parity2)."""
    k_len = len(bits)
    lib = srsran_fec_library()

    if lib is not None and k_len in QPP_TABLE:
        u8_bits = np.ascontiguousarray(bits, dtype=np.uint8)
        coded_len = 3 * k_len + 12
        out_coded = np.zeros(coded_len, dtype=np.uint8)
        res = lib.srsran_fec_turbo_encode(
            u8_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out_coded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            k_len,
        )
        if res == 0:
            s = out_coded[0 : 3 * k_len : 3]
            p1 = out_coded[1 : 3 * k_len : 3]
            p2 = out_coded[2 : 3 * k_len : 3]
            return s, p1, p2

    pi, _ = get_qpp_indices(k_len)
    def _encode_rsc(stream):
        s = 0
        p = np.empty(len(stream), dtype=np.uint8)
        for i, u in enumerate(stream):
            ns = NEXT_STATE[s, u]
            p[i] = OUT_PARITY[s, u]
            s = ns
        return p

    p1 = _encode_rsc(bits)
    p2 = _encode_rsc(bits[pi])
    return bits.copy(), p1, p2


def decode_pdsch_transport_block(
    rx_llr: np.ndarray,
    tbs: int,
    rv_idx: int = 0,
    max_iterations: int = 6,
) -> Tuple[np.ndarray, bool]:
    """End-to-end PDSCH transport block decoding using srsRAN C library.
    
    Performs rate unmatching, Max-Log-MAP Turbo decoding, and CRC-24A verification.
    
    Args:
        rx_llr: Demodulated soft LLRs from PDSCH resource elements (length E).
        tbs: Transport Block Size in bits (e.g. from DCI 1A format).
        rv_idx: Redundancy Version (0, 1, 2, or 3).
        max_iterations: Maximum Turbo decoder iterations.
        
    Returns:
        (payload_bits, crc_passed) tuple.
    """
    lib = srsran_fec_library()
    tbs = int(tbs)
    e_len = len(rx_llr)

    if lib is not None:
        llr_f32 = np.ascontiguousarray(rx_llr, dtype=np.float32)
        out_payload = np.zeros(tbs, dtype=np.uint8)
        crc_passed = ctypes.c_int(0)
        
        # Scale LLR to srsRAN optimal range (~2.0)
        mean_mag = float(np.mean(np.abs(llr_f32)))
        scale = 2.0 / max(1.0, mean_mag)
        scaled_llr = llr_f32 * scale

        lib.srsran_fec_decode_tb(
            scaled_llr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            e_len,
            out_payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            tbs,
            rv_idx,
            max_iterations,
            ctypes.byref(crc_passed),
        )
        if crc_passed.value == 1:
            return out_payload, True

        # Try flipped LLR polarity if receiver used inverse mapping
        flipped_llr = -scaled_llr
        lib.srsran_fec_decode_tb(
            flipped_llr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            e_len,
            out_payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            tbs,
            rv_idx,
            max_iterations,
            ctypes.byref(crc_passed),
        )
        return out_payload, (crc_passed.value == 1)

    # Fallback if library missing
    out_payload = np.zeros(tbs, dtype=np.uint8)
    return out_payload, False
