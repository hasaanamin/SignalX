"""LTE PCFICH (Physical Control Format Indicator Channel) Encoder & Decoder.

Compliant with 3GPP TS 36.211 §§6.7 and TS 36.212 §5.3.4.

PCFICH is transmitted in OFDM symbol 0 of every subframe and indicates the
number of OFDM symbols (CFI = 1, 2, or 3) allocated to the control region (PDCCH).
It occupies 4 Resource Element Groups (REGs) = 16 Resource Elements (32 bits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from signal_processing.lte.bch import gold_sequence


# 3GPP TS 36.212 Table 5.3.4-1: CFI 32-bit Block Codewords
CFI_CODEWORDS: Dict[int, Tuple[int, ...]] = {
    1: (0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1),
    2: (1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0),
    3: (1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1),
    4: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
}


@dataclass(frozen=True)
class PCFICHObservation:
    """Immutable result of PCFICH decoding."""
    cfi: int                         # Detected Control Format Indicator (1, 2, or 3)
    confidence: float                # Margin between best and second-best hypothesis
    correlation_scores: Dict[int, float]  # Scores for CFI 1, 2, 3 (and 4)
    soft_symbols: np.ndarray         # 16 equalized QPSK complex symbols
    llr_bits: np.ndarray             # 32 de-scrambled LLR soft bits
    pci: int
    subframe: int
    num_rb_dl: int


def pcfich_reg_indices(pci: int, num_rb_dl: int) -> Tuple[int, int, int, int]:
    """Calculate the 4 REG indices in OFDM symbol 0 assigned to PCFICH (TS 36.211 §6.7.4).
    
    Total REGs in symbol 0 = 2 * num_rb_dl.
    """
    total_regs = 2 * num_rb_dl
    n_bar = pci % total_regs
    regs = tuple((n_bar + (i * num_rb_dl) // 2) % total_regs for i in range(4))
    return (int(regs[0]), int(regs[1]), int(regs[2]), int(regs[3]))


def pcfich_re_subcarriers(pci: int, num_rb_dl: int) -> np.ndarray:
    """Return the 16 physical subcarrier indices (0 to 12*num_rb_dl - 1) for PCFICH in symbol 0."""
    reg_indices = pcfich_reg_indices(pci, num_rb_dl)
    v_shift = pci % 6
    subcarriers = []

    for reg_idx in reg_indices:
        start_k = 6 * reg_idx
        # Within the 6 subcarriers, CRS pilots are at (v_shift % 6) and ((v_shift + 3) % 6)
        crs_offsets = {v_shift % 6, (v_shift + 3) % 6}
        reg_re_offsets = [j for j in range(6) if j not in crs_offsets]
        for offset in reg_re_offsets:
            subcarriers.append(start_k + offset)

    return np.array(subcarriers, dtype=np.intp)


def pcfich_scrambling_sequence(subframe: int, pci: int) -> np.ndarray:
    """Generate the 32-bit cell-specific Gold scrambling sequence (TS 36.211 §6.7.1)."""
    c_init = ((subframe + 1) * (2 * pci + 1) * 512 + pci) & 0x7FFFFFFF
    return gold_sequence(c_init, 32)


def pcfich_encode(
    cfi: int,
    subframe: int,
    pci: int,
    num_rb_dl: int,
    ports: int = 1,
) -> np.ndarray:
    """Test-only transmitter synthesis: Generates 16 QPSK symbols or grid subcarriers."""
    if cfi not in CFI_CODEWORDS:
        raise ValueError(f"Invalid CFI {cfi}; must be 1, 2, 3, or 4")

    bits = np.array(CFI_CODEWORDS[cfi], dtype=np.uint8)
    scrambler = pcfich_scrambling_sequence(subframe, pci)
    scrambled = (bits ^ scrambler)

    # QPSK modulation: d(i) = ((1 - 2*b0) + 1j*(1 - 2*b1)) / sqrt(2)
    b0 = scrambled[0::2]
    b1 = scrambled[1::2]
    symbols = ((1.0 - 2.0 * b0) + 1j * (1.0 - 2.0 * b1)) / np.sqrt(2.0)
    return symbols


class PCFICHDecoder:
    """Decodes PCFICH to extract the Control Format Indicator (CFI: 1, 2, or 3)."""

    def __init__(self, pci: int, num_rb_dl: int = 6) -> None:
        self.pci = int(pci)
        self.num_rb_dl = int(num_rb_dl)
        self.re_indices = pcfich_re_subcarriers(self.pci, self.num_rb_dl)

        # Precompute bipolar representations of CFI codewords: (1 - 2*b)
        self.cfi_bipolar: Dict[int, np.ndarray] = {
            cfi: (1.0 - 2.0 * np.array(cw, dtype=np.float64))
            for cfi, cw in CFI_CODEWORDS.items()
        }

    def decode(
        self,
        symbol0_subcarriers: np.ndarray,
        channel_estimates: Optional[np.ndarray] = None,
        subframe: int = 5,
        ports: int = 1,
    ) -> PCFICHObservation:
        """Decode PCFICH from the received symbol 0 subcarriers.
        
        Args:
            symbol0_subcarriers: Complex array of subcarriers in OFDM symbol 0.
                Length must be >= max(self.re_indices) + 1 (e.g. 72 for 6 RBs, 600 for 50 RBs).
            channel_estimates: Optional channel response H(f) across symbol 0 subcarriers.
                If None, unitary channel is assumed.
            subframe: Subframe index (0 to 9, default 5 for SIB subframe).
            ports: Number of antenna ports (1, 2, or 4).
        """
        received = np.asarray(symbol0_subcarriers, dtype=np.complex128)[self.re_indices]

        if channel_estimates is not None:
            h = np.asarray(channel_estimates, dtype=np.complex128)[self.re_indices]
            h_power = np.abs(h) ** 2
            reg = 0.01 * float(np.mean(h_power)) + 1e-12
            equalized = (received * h.conj()) / (h_power + reg)
        else:
            equalized = received

        # Demap QPSK soft symbols -> 32 LLRs (positive for bit 0)
        llrs = np.empty(32, dtype=np.float64)
        llrs[0::2] = np.sqrt(2.0) * equalized.real
        llrs[1::2] = np.sqrt(2.0) * equalized.imag

        # De-scramble soft bits with Gold sequence
        scrambler = pcfich_scrambling_sequence(subframe, self.pci)
        descramble_sign = 1.0 - 2.0 * scrambler.astype(np.float64)
        descrambled_llrs = llrs * descramble_sign

        # Correlate with candidate CFI block codewords
        scores: Dict[int, float] = {}
        for cfi, bipolar_cw in self.cfi_bipolar.items():
            scores[cfi] = float(np.dot(descrambled_llrs, bipolar_cw))

        # Best hypothesis among standard CFIs (1, 2, 3)
        candidate_cfis = [1, 2, 3] if self.num_rb_dl > 10 else [1, 2, 3, 4]
        sorted_candidates = sorted(candidate_cfis, key=lambda c: scores[c], reverse=True)
        best_cfi = sorted_candidates[0]
        second_best_cfi = sorted_candidates[1]
        confidence = scores[best_cfi] - scores[second_best_cfi]

        return PCFICHObservation(
            cfi=best_cfi,
            confidence=confidence,
            correlation_scores=scores,
            soft_symbols=equalized,
            llr_bits=descrambled_llrs,
            pci=self.pci,
            subframe=subframe,
            num_rb_dl=self.num_rb_dl,
        )
