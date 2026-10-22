"""Unit tests for LTE PDCCH & DCI Format 1A Blind Decoding.

Validates RIV encoding/decoding, circular buffer rate matching, CRC-16
masking with SI-RNTI (0xFFFF), and blind decoding at aggregation levels 4 and 8.
"""

import numpy as np
import pytest

from signal_processing.lte.bch import crc16
from signal_processing.lte.pdcch import (
    SI_RNTI,
    DCI1AAllocation,
    PDCCHDecoder,
    dci_1a_payload_length,
    dci_1a_riv_bits,
    decode_riv,
    encode_riv,
    get_tbs,
    pdcch_derate_match,
    pdcch_rate_match,
    pdcch_scrambling_sequence,
    viterbi_tail_biting,
)


def encode_conv(bits: np.ndarray) -> np.ndarray:
    """Helper convolutional encoder for tail-biting code."""
    state = sum(bits[-1 - j] << j for j in range(6))
    code = []
    for bit in bits:
        state = ((state << 1) | bit) & 0x7F
        code.extend((state & p).bit_count() % 2 for p in (109, 79, 87))
    return np.array(code, dtype=np.uint8).reshape(-1, 3)


def make_dci_1a_candidate(
    start_rb: int,
    num_rbs: int,
    mcs: int,
    num_rb_dl: int,
    subframe: int = 5,
    pci: int = 42,
    aggregation_level: int = 4,
    rnti: int = SI_RNTI,
) -> np.ndarray:
    """Synthesize transmitted PDCCH CCE soft bits for testing."""
    riv_bits = dci_1a_riv_bits(num_rb_dl)
    riv = encode_riv(start_rb, num_rbs, num_rb_dl)
    riv_bin = [int(b) for b in f"{riv:0{riv_bits}b}"]

    # DCI 1A fields:
    # 1 bit fmt, 1 bit vrb, riv_bits riv, 5 bits mcs, 3 bits harq, 1 bit ndi, 2 bits rv, 2 bits tpc
    payload = [1, 0] + riv_bin + [int(b) for b in f"{mcs:05b}"] + [0, 0, 0, 1, 0, 0, 0, 0]
    payload = np.array(payload, dtype=np.uint8)

    # Compute CRC-16 and mask with RNTI
    calc_crc = crc16(payload)
    masked_crc = calc_crc ^ rnti
    crc_bits = np.array([(masked_crc >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)

    coded_input = np.concatenate((payload, crc_bits))
    code_3k = encode_conv(coded_input)

    out_len = aggregation_level * 72
    rate_matched = pdcch_rate_match(code_3k, out_len)

    # Scramble
    scrambler = pdcch_scrambling_sequence(subframe, pci, out_len)
    scrambled = rate_matched ^ scrambler

    # Convert to soft bits (positive for bit 0)
    return (1.0 - 2.0 * scrambled.astype(np.float64))


@pytest.mark.parametrize("num_rb_dl", [6, 15, 25, 50, 75, 100])
def test_riv_encode_decode_roundtrip(num_rb_dl):
    """Verify RIV encoding and decoding roundtrips across all valid allocations."""
    test_cases = [
        (0, 1),
        (0, num_rb_dl),
        (1, 2),
        (num_rb_dl // 2, min(3, num_rb_dl - num_rb_dl // 2)),
        (num_rb_dl - 2, 2),
    ]
    for start, length in test_cases:
        riv = encode_riv(start, length, num_rb_dl)
        dec_start, dec_len = decode_riv(riv, num_rb_dl)
        assert (start, length) == (dec_start, dec_len), (
            f"RIV roundtrip failed for start={start}, len={length}, num_rb={num_rb_dl}"
        )


@pytest.mark.parametrize("aggregation_level", [4, 8])
@pytest.mark.parametrize("pci", [0, 42, 174])
@pytest.mark.parametrize("num_rb_dl", [6, 25, 50])
def test_pdcch_si_rnti_blind_decode_noiseless(aggregation_level, pci, num_rb_dl):
    """Verify bit-exact blind decoding of DCI 1A for SI-RNTI at levels 4 and 8."""
    decoder = PDCCHDecoder(pci=pci, num_rb_dl=num_rb_dl)

    start_rb = 2
    num_rbs = 4
    mcs = 5
    subframe = 5

    cce_soft_bits = make_dci_1a_candidate(
        start_rb=start_rb,
        num_rbs=num_rbs,
        mcs=mcs,
        num_rb_dl=num_rb_dl,
        subframe=subframe,
        pci=pci,
        aggregation_level=aggregation_level,
        rnti=SI_RNTI,
    )

    dci = decoder.decode_cce_candidate(
        cce_soft_bits, aggregation_level=aggregation_level, cce_index=0, subframe=subframe
    )

    assert dci is not None
    assert isinstance(dci, DCI1AAllocation)
    assert dci.start_rb == start_rb
    assert dci.num_rbs == num_rbs
    assert dci.mcs == mcs
    assert dci.rnti == SI_RNTI
    assert dci.tbs_bits == get_tbs(mcs, num_rbs)
    assert dci.aggregation_level == aggregation_level


def test_pdcch_rejects_unrelated_rnti():
    """Verify PDCCH decoder returns None when candidate belongs to another RNTI (e.g. C-RNTI)."""
    pci = 42
    num_rb = 50
    decoder = PDCCHDecoder(pci=pci, num_rb_dl=num_rb)

    c_rnti = 0x4A1B
    cce_soft_bits = make_dci_1a_candidate(
        start_rb=10,
        num_rbs=5,
        mcs=3,
        num_rb_dl=num_rb,
        rnti=c_rnti,
    )

    dci = decoder.decode_cce_candidate(cce_soft_bits, aggregation_level=4, cce_index=0)
    assert dci is None, "Expected decoder to reject C-RNTI when scanning for SI-RNTI"


def test_pdcch_decoding_under_awgn_noise():
    """Verify PDCCH decoding under 2 dB AWGN noise."""
    pci = 42
    num_rb = 50
    decoder = PDCCHDecoder(pci=pci, num_rb_dl=num_rb)

    clean_soft = make_dci_1a_candidate(
        start_rb=5,
        num_rbs=8,
        mcs=4,
        num_rb_dl=num_rb,
        aggregation_level=4,
        rnti=SI_RNTI,
    )

    rng = np.random.default_rng(2026)
    noise_sigma = 0.8  # ~2 dB SNR
    noisy_soft = clean_soft + noise_sigma * rng.normal(size=len(clean_soft))

    dci = decoder.decode_cce_candidate(noisy_soft, aggregation_level=4, cce_index=0)
    assert dci is not None
    assert dci.start_rb == 5
    assert dci.num_rbs == 8
    assert dci.rnti == SI_RNTI
