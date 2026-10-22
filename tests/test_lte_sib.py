"""Unit tests for LTE SIB1 (System Information Block 1) Receiver & Parser.

Validates 3GPP TS 36.331 SIB1 bitstream unmarshaling, PLMN operator lookup,
cell identity decomposition, and SIB1Receiver orchestration.
"""

import numpy as np
import pytest

from signal_processing.lte.pcfich import pcfich_encode, pcfich_re_subcarriers
from signal_processing.lte.pdcch import dci_1a_payload_length
from signal_processing.lte.sib import (
    LTESIB1Observation,
    SIB1Receiver,
    pack_sib1_bits,
    parse_sib1_bits,
)
from signal_processing.lte.turbo import attach_crc24a, turbo_encode


def test_sib1_bit_packing_and_parsing_att():
    """Verify bitfield packing and parsing for AT&T (310-410)."""
    mcc = "310"
    mnc = "410"
    tac = 0x1234
    eci = 0x0123456  # 28 bits
    band = 4

    packed = pack_sib1_bits(
        mcc=mcc, mnc=mnc, tac=tac, cell_identity=eci, freq_band=band, target_len=120
    )
    assert len(packed) == 120

    obs = parse_sib1_bits(packed, pci=42, sfn=840, subframe=5, cfi=2, start_rb=0, num_rbs=6)

    assert isinstance(obs, LTESIB1Observation)
    assert obs.mcc == mcc
    assert obs.mnc == mnc
    assert "AT&T" in obs.operator_name
    assert obs.tracking_area_code == tac
    assert obs.cell_identity == eci
    assert obs.enodeb_id == (eci >> 8)
    assert obs.sector_id == (eci & 0xFF)
    assert obs.frequency_band_indicator == band
    assert obs.pci == 42
    assert obs.sfn == 840
    assert obs.subframe == 5


def test_sib1_bit_packing_and_parsing_vodafone():
    """Verify bitfield packing and parsing for Vodafone UK (234-15)."""
    mcc = "234"
    mnc = "15"
    tac = 45678
    eci = 0x7654321
    band = 20

    packed = pack_sib1_bits(
        mcc=mcc, mnc=mnc, tac=tac, cell_identity=eci, freq_band=band, target_len=120
    )
    obs = parse_sib1_bits(packed, pci=174)

    assert obs.mcc == mcc
    assert obs.mnc == mnc
    assert "Vodafone" in obs.operator_name
    assert obs.tracking_area_code == tac
    assert obs.cell_identity == eci
    assert obs.frequency_band_indicator == band


def test_sib1_receiver_end_to_end():
    """Verify SIB1Receiver decodes PCFICH and PDSCH subframe into SIB1 observation."""
    pci = 42
    num_rb = 6
    receiver = SIB1Receiver(pci=pci, num_rb_dl=num_rb)

    # 1. Synthesize subframe 5 grid (14 symbols x 72 subcarriers)
    grid = np.zeros((14, 12 * num_rb), dtype=np.complex128)

    # Place PCFICH in Symbol 0 (CFI = 2)
    pcfich_syms = pcfich_encode(cfi=2, subframe=5, pci=pci, num_rb_dl=num_rb)
    re_indices = pcfich_re_subcarriers(pci, num_rb)
    grid[0, re_indices] = pcfich_syms

    # Generate synthetic SIB1 payload bits
    sib_bits = pack_sib1_bits(mcc="310", mnc="260", tac=9999, cell_identity=0x3ABCDEF, target_len=120)
    coded = attach_crc24a(sib_bits)

    # Modulate into QPSK symbols in symbol 2..13
    qpsk_syms = ((1.0 - 2.0 * coded[0::2]) + 1j * (1.0 - 2.0 * coded[1::2])) / np.sqrt(2.0)
    # Fill PDSCH region in grid
    idx = 0
    for l in range(2, 14):
        for k in range(12 * num_rb):
            if idx < len(qpsk_syms):
                grid[l, k] = qpsk_syms[idx]
                idx += 1

    obs = receiver.decode_from_subframe(grid, sfn=936, subframe=5)
    assert obs is not None
    assert obs.mcc == "310"
    assert obs.mnc == "260"
    assert "T-Mobile" in obs.operator_name
    assert obs.cfi == 2
    assert obs.crc_valid is True
    assert obs.pci == pci
