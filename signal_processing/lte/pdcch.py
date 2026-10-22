"""LTE PDCCH (Physical Downlink Control Channel) & DCI Blind Decoder.

Compliant with 3GPP TS 36.211 §6.8 and TS 36.212 §5.3.3.

PDCCH carries Downlink Control Information (DCI) in the first CFI OFDM symbols.
For SIB1 decoding, the receiver searches the Common Search Space (CSS) at
aggregation levels 4 and 8 for DCI Format 1A scrambled with the SI-RNTI (0xFFFF).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from signal_processing.lte.bch import _trellis, crc16, gold_sequence

# Sub-block interleaver permutation (TS 36.212 Table 5.1.4-1)
PERMUTATION = np.array([
    1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
    0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
], dtype=int)

# Standard SI-RNTI (System Information RNTI)
SI_RNTI = 0xFFFF


# 3GPP TS 36.213 Table 7.1.7.2.1-1: Transport Block Size (TBS) table subset for SIB1
# Rows: ITBS (0 to 26), Columns: N_PRB (1 to 10)
TBS_TABLE: Dict[int, Dict[int, int]] = {
    0: {1: 16, 2: 32, 3: 56, 4: 88, 5: 120, 6: 152, 7: 176, 8: 208, 9: 224, 10: 256},
    1: {1: 24, 2: 56, 3: 88, 4: 144, 5: 176, 6: 208, 7: 224, 8: 256, 9: 328, 10: 344},
    2: {1: 32, 2: 72, 3: 144, 4: 176, 5: 208, 6: 256, 7: 328, 8: 392, 9: 440, 10: 488},
    3: {1: 40, 2: 104, 3: 176, 4: 208, 5: 256, 6: 328, 7: 440, 8: 504, 9: 584, 10: 648},
    4: {1: 56, 2: 120, 3: 208, 4: 256, 5: 328, 6: 408, 7: 552, 8: 632, 9: 712, 10: 776},
    5: {1: 72, 2: 144, 3: 224, 4: 328, 5: 424, 6: 504, 7: 680, 8: 776, 9: 872, 10: 968},
    6: {1: 328, 2: 176, 3: 256, 4: 392, 5: 504, 6: 600, 7: 808, 8: 936, 9: 1032, 10: 1192},
    7: {1: 104, 2: 224, 3: 328, 4: 472, 5: 584, 6: 712, 7: 968, 8: 1096, 9: 1256, 10: 1384},
    8: {1: 120, 2: 256, 3: 392, 4: 536, 5: 680, 6: 808, 7: 1096, 8: 1256, 9: 1416, 10: 1544},
    9: {1: 136, 2: 296, 3: 456, 4: 616, 5: 776, 6: 936, 7: 1256, 8: 1416, 9: 1608, 10: 1800},
}


@dataclass(frozen=True)
class DCI1AAllocation:
    """Decoded DCI Format 1A resource allocation grant for SIB1."""
    start_rb: int                    # Starting physical/virtual resource block index
    num_rbs: int                     # Number of allocated resource blocks
    mcs: int                         # Modulation & Coding Scheme index (0 to 31)
    tbs_bits: int                    # Transport Block Size in bits
    rv: int                          # Redundancy Version (0, 1, 2, or 3)
    harq_process: int                # HARQ process identifier
    is_distributed: bool             # Distributed vs Localized VRB
    rnti: int                        # Unmasked RNTI (0xFFFF for SI-RNTI)
    aggregation_level: int           # Aggregation level (4 or 8)
    cce_index: int                   # Starting CCE index in search space
    payload_bits: Tuple[int, ...]    # Unpacked DCI payload bits


def dci_1a_riv_bits(num_rb_dl: int) -> int:
    """Number of bits used for Resource Allocation Type 2 RIV (TS 36.213 §7.1.6.3)."""
    val = (num_rb_dl * (num_rb_dl + 1)) // 2
    return int(math.ceil(math.log2(val)))


def dci_1a_payload_length(num_rb_dl: int) -> int:
    """Calculate the total DCI Format 1A payload bit length for FDD (TS 36.212 §5.3.3.1.3)."""
    # 1 (format 0/1A flag) + 1 (vrb flag) + riv_bits + 5 (mcs) + 3 (harq) + 1 (ndi) + 2 (rv) + 2 (tpc)
    return 15 + dci_1a_riv_bits(num_rb_dl)


def decode_riv(riv: int, num_rb_dl: int) -> Tuple[int, int]:
    """Decode Resource Indication Value (RIV) into (start_rb, num_rbs) (TS 36.213 §7.1.6.3)."""
    length = riv // num_rb_dl + 1
    start = riv % num_rb_dl
    if length > num_rb_dl - start:
        length = num_rb_dl - length + 2
        start = num_rb_dl - start - 1
    return int(start), int(length)


def encode_riv(start_rb: int, num_rbs: int, num_rb_dl: int) -> int:
    """Encode (start_rb, num_rbs) into RIV for DCI generation."""
    if num_rbs - 1 <= num_rb_dl // 2:
        return num_rb_dl * (num_rbs - 1) + start_rb
    else:
        return num_rb_dl * (num_rb_dl - num_rbs + 1) + (num_rb_dl - 1 - start_rb)


def get_tbs(mcs: int, num_rbs: int) -> int:
    """Lookup Transport Block Size (TBS) in bits (TS 36.213 Table 7.1.7.2.1-1)."""
    i_tbs = min(mcs, 9)
    rbs = max(1, min(num_rbs, 10))
    return TBS_TABLE.get(i_tbs, {}).get(rbs, rbs * 144)


def subblock_interleave(bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """Sub-block interleaver for tail-biting convolutional code streams (TS 36.212 §5.1.4.2)."""
    k_len = len(bits)
    r_rows = int(math.ceil(k_len / 32))
    k_pi = 32 * r_rows
    n_dummy = k_pi - k_len
    flat = np.r_[np.full(n_dummy, -1, dtype=int), bits]
    matrix = flat.reshape(r_rows, 32)
    permuted = matrix[:, PERMUTATION]
    output = permuted.T.ravel()
    return output, n_dummy


def pdcch_rate_match(code_bits: np.ndarray, out_len: int) -> np.ndarray:
    """Rate match (K, 3) coded bits to output length via circular buffer (TS 36.212 §5.1.4.2)."""
    v0, _ = subblock_interleave(code_bits[:, 0])
    v1, _ = subblock_interleave(code_bits[:, 1])
    v2, _ = subblock_interleave(code_bits[:, 2])
    v12 = np.empty(len(v1) * 2, dtype=int)
    v12[0::2] = v1
    v12[1::2] = v2
    circular = np.concatenate((v0, v12))
    valid = circular[circular != -1]
    return np.resize(valid, out_len)


def pdcch_derate_match(soft_bits: np.ndarray, k_len: int) -> np.ndarray:
    """De-rate match soft bits from circular buffer back to (K, 3) soft stream."""
    r_rows = int(math.ceil(k_len / 32))
    k_pi = 32 * r_rows
    n_dummy = k_pi - k_len

    idx0 = np.r_[np.full(n_dummy, -1), np.arange(k_len)].reshape(r_rows, 32)[:, PERMUTATION].T.ravel()
    idx1 = np.r_[np.full(n_dummy, -1), np.arange(k_len)].reshape(r_rows, 32)[:, PERMUTATION].T.ravel()
    idx2 = np.r_[np.full(n_dummy, -1), np.arange(k_len)].reshape(r_rows, 32)[:, PERMUTATION].T.ravel()

    idx12 = np.empty(len(idx1) * 2, dtype=int)
    idx12[0::2] = np.where(idx1 != -1, 3 * idx1 + 1, -1)
    idx12[1::2] = np.where(idx2 != -1, 3 * idx2 + 2, -1)

    idx0_full = np.where(idx0 != -1, 3 * idx0, -1)
    circular = np.concatenate((idx0_full, idx12))
    valid = circular[circular != -1]

    soft_code = np.zeros(3 * k_len, dtype=np.float64)
    n_out = len(soft_bits)
    mapping = np.resize(valid, n_out)
    for i in range(n_out):
        dest = mapping[i]
        soft_code[dest] += soft_bits[i]

    return soft_code


def viterbi_tail_biting(soft_3k: np.ndarray, k_len: int) -> np.ndarray:
    """General K-length 64-state tail-biting Viterbi decoder (TS 36.212 §5.1.3.1)."""
    soft = np.asarray(soft_3k, dtype=np.float64).reshape(k_len, 3)
    scale = float(np.max(np.abs(soft)))
    if scale > 0:
        soft = soft / scale

    predecessors, signs = _trellis()
    metrics = np.full((64, 64), -np.inf)
    np.fill_diagonal(metrics, 0.0)
    decisions = np.empty((k_len, 64, 64), bool)

    for t in range(k_len):
        branch = np.einsum("ijk,k->ij", signs, soft[t], optimize=False)
        a = metrics[:, predecessors[0]] + branch[0]
        b = metrics[:, predecessors[1]] + branch[1]
        decisions[t] = b > a
        metrics = np.maximum(a, b)

    initial = int(np.argmax(metrics.diagonal()))
    state = initial
    bits = np.empty(k_len, dtype=np.uint8)
    for t in range(k_len - 1, -1, -1):
        bits[t] = state & 1
        state = int(predecessors[int(decisions[t, initial, state]), state])

    return bits


def pdcch_scrambling_sequence(subframe: int, pci: int, length: int) -> np.ndarray:
    """Cell-specific scrambling sequence for PDCCH (TS 36.211 §6.8.2)."""
    c_init = ((subframe) * 512 + pci) & 0x7FFFFFFF
    return gold_sequence(c_init, length)


class PDCCHDecoder:
    """Blind decoder for PDCCH Common Search Space (SI-RNTI 0xFFFF)."""

    def __init__(self, pci: int, num_rb_dl: int = 6) -> None:
        self.pci = int(pci)
        self.num_rb_dl = int(num_rb_dl)
        self.dci_payload_len = dci_1a_payload_length(self.num_rb_dl)
        self.total_coded_len = self.dci_payload_len + 16  # Payload + 16 CRC bits
        self.riv_bits = dci_1a_riv_bits(self.num_rb_dl)

    def decode_cce_candidate(
        self,
        cce_soft_bits: np.ndarray,
        aggregation_level: int = 4,
        cce_index: int = 0,
        subframe: int = 5,
    ) -> Optional[DCI1AAllocation]:
        """Decode a single CCE aggregation candidate (e.g. 288 soft bits for level 4)."""
        expected_len = aggregation_level * 72
        if len(cce_soft_bits) != expected_len:
            raise ValueError(f"Expected {expected_len} soft bits for level {aggregation_level}")

        # De-scramble soft bits
        scrambler = pdcch_scrambling_sequence(subframe, self.pci, expected_len)
        descramble_sign = 1.0 - 2.0 * scrambler.astype(np.float64)
        descrambled_soft = cce_soft_bits * descramble_sign

        # De-rate match
        soft_3k = pdcch_derate_match(descrambled_soft, self.total_coded_len)

        # Tail-biting Viterbi decode
        decoded_bits = viterbi_tail_biting(soft_3k, self.total_coded_len)

        # Separate payload and CRC
        payload = tuple(map(int, decoded_bits[: self.dci_payload_len]))
        rx_crc = int("".join(map(str, decoded_bits[self.dci_payload_len :])), 2)

        # Compute CRC-16
        calc_crc = crc16(payload)
        unmasked_rnti = rx_crc ^ calc_crc

        # Check if matched SI-RNTI (0xFFFF)
        if unmasked_rnti != SI_RNTI:
            return None

        # Unpack DCI Format 1A fields
        # Bit 0: Format 0/1A flag (must be 1 for Format 1A)
        # Bit 1: Localized/Distributed VRB flag
        # Bits 2 .. 2 + riv_bits: RIV
        # Next 5 bits: MCS
        # Next 3 bits: HARQ process
        # Next 1 bit: NDI
        # Next 2 bits: RV
        pos = 0
        fmt_flag = payload[pos]
        pos += 1
        vrb_flag = payload[pos]
        pos += 1

        riv_str = "".join(map(str, payload[pos : pos + self.riv_bits]))
        riv = int(riv_str, 2)
        pos += self.riv_bits

        mcs = int("".join(map(str, payload[pos : pos + 5])), 2)
        pos += 5

        harq = int("".join(map(str, payload[pos : pos + 3])), 2)
        pos += 3

        ndi = payload[pos]
        pos += 1

        rv = int("".join(map(str, payload[pos : pos + 2])), 2)
        pos += 2

        start_rb, num_rbs = decode_riv(riv, self.num_rb_dl)
        tbs = get_tbs(mcs, num_rbs)

        return DCI1AAllocation(
            start_rb=start_rb,
            num_rbs=num_rbs,
            mcs=mcs,
            tbs_bits=tbs,
            rv=rv,
            harq_process=harq,
            is_distributed=bool(vrb_flag),
            rnti=unmasked_rnti,
            aggregation_level=aggregation_level,
            cce_index=cce_index,
            payload_bits=payload,
        )
