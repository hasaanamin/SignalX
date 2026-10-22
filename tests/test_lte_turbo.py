"""Unit tests for 3GPP Turbo Code Encoder & Decoder (TS 36.212 §§5.1.2, 5.1.3).

Validates QPP interleaver, CRC-24A attachment/verification, and iterative
Max-Log-MAP decoding with early stopping under noiseless and AWGN channels.
"""

import numpy as np
import pytest

from signal_processing.lte.turbo import (
    TurboDecoder,
    attach_crc24a,
    crc24a,
    get_qpp_indices,
    turbo_encode,
    verify_crc24a,
)


def test_crc24a_validation():
    """Verify CRC-24A attachment and remainder validation."""
    payload = np.random.randint(0, 2, 120, dtype=np.uint8)
    coded = attach_crc24a(payload)
    assert len(coded) == 120 + 24
    assert verify_crc24a(coded) is True

    # Single bit corruption should invalidate CRC
    corrupted = coded.copy()
    corrupted[10] ^= 1
    assert verify_crc24a(corrupted) is False


@pytest.mark.parametrize("k_len", [40, 56, 80, 120, 144, 208, 328, 512])
def test_qpp_interleaver_permutation(k_len):
    """Verify QPP interleaver produces a strict 1-to-1 bijection (permutation)."""
    pi, pi_inv = get_qpp_indices(k_len)
    assert len(pi) == k_len
    assert len(np.unique(pi)) == k_len
    assert np.all(pi >= 0)
    assert np.all(pi < k_len)

    # Invertibility check
    identity = pi_inv[pi]
    assert np.array_equal(identity, np.arange(k_len))


@pytest.mark.parametrize("payload_len", [40, 80, 120, 208])
def test_turbo_decoder_noiseless(payload_len):
    """Verify bit-exact Turbo decoding with early stopping on noiseless input."""
    k_len = payload_len + 24  # Add CRC-24A
    raw_payload = np.random.randint(0, 2, payload_len, dtype=np.uint8)
    bits = attach_crc24a(raw_payload)

    sys_b, p1_b, p2_b = turbo_encode(bits)
    l_sys = 1.0 - 2.0 * sys_b.astype(float)
    l_p1 = 1.0 - 2.0 * p1_b.astype(float)
    l_p2 = 1.0 - 2.0 * p2_b.astype(float)

    decoder = TurboDecoder(k_len, max_iterations=6)
    res = decoder.decode(l_sys, l_p1, l_p2)

    assert res.crc_valid is True
    assert np.array_equal(res.bits, bits)
    assert np.array_equal(res.bits[:payload_len], raw_payload)
    assert res.iterations_used <= 1  # Should stop immediately on clean input


def test_turbo_decoder_under_awgn_noise():
    """Verify iterative decoding convergence under AWGN noise (0.5 sigma)."""
    payload_len = 80
    k_len = payload_len + 24
    bits = attach_crc24a(np.random.randint(0, 2, payload_len, dtype=np.uint8))

    sys_b, p1_b, p2_b = turbo_encode(bits)
    l_sys = 1.0 - 2.0 * sys_b.astype(float)
    l_p1 = 1.0 - 2.0 * p1_b.astype(float)
    l_p2 = 1.0 - 2.0 * p2_b.astype(float)

    rng = np.random.default_rng(2026)
    sigma = 0.5
    l_sys_noisy = l_sys + sigma * rng.normal(size=k_len)
    l_p1_noisy = l_p1 + sigma * rng.normal(size=k_len)
    l_p2_noisy = l_p2 + sigma * rng.normal(size=k_len)

    decoder = TurboDecoder(k_len, max_iterations=6)
    res = decoder.decode(l_sys_noisy, l_p1_noisy, l_p2_noisy)

    assert res.crc_valid is True
    assert np.array_equal(res.bits, bits)


def test_srsran_fec_library_status():
    """Verify srsRAN C shared library is compiled, loaded, and available."""
    from signal_processing.lte.turbo import is_srsran_available, srsran_fec_library
    assert is_srsran_available() is True
    assert srsran_fec_library() is not None


@pytest.mark.parametrize("tbs,E", [(120, 288), (208, 432), (328, 720)])
def test_srsran_end_to_end_transport_block_decode(tbs, E):
    """Verify srsRAN C end-to-end transport block rate matching, turbo decoding, and CRC-24A."""
    import ctypes
    from signal_processing.lte.turbo import srsran_fec_library, decode_pdsch_transport_block

    lib = srsran_fec_library()
    assert lib is not None

    rng = np.random.default_rng(tbs)
    tx_payload = rng.integers(0, 2, tbs, dtype=np.uint8)

    # Attach CRC-24A
    K = tbs + 24
    crc_val = lib.srsran_fec_crc24a_checksum(tx_payload.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), tbs)
    crc_bits = np.array([(crc_val >> (23 - i)) & 1 for i in range(24)], dtype=np.uint8)
    cb_bits = np.concatenate([tx_payload, crc_bits])

    # Encode with Turbo
    coded_len = 3 * K + 12
    coded = np.zeros(coded_len, dtype=np.uint8)
    lib.srsran_fec_turbo_encode(cb_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), coded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), K)

    # Rate match to E bits
    matched = np.zeros(E, dtype=np.uint8)
    lib.srsran_fec_rm_turbo_tx(coded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), coded_len, matched.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), E, 0)

    # Map to soft LLRs (+2.0 for bit 1, -2.0 for bit 0)
    rx_llr = np.where(matched == 1, 2.0, -2.0).astype(np.float32)

    decoded_payload, crc_valid = decode_pdsch_transport_block(rx_llr, tbs, rv_idx=0)
    assert crc_valid is True
    assert np.array_equal(decoded_payload, tx_payload)

