"""Unit tests for LTE PCFICH (Physical Control Format Indicator Channel).

Validates 3GPP TS 36.211 §6.7 resource mapping, scrambling, QPSK modulation,
and block decoding for CFI values 1, 2, and 3 under clean and noisy channels.
"""

import numpy as np
import pytest

from signal_processing.lte.pcfich import (
    CFI_CODEWORDS,
    PCFICHDecoder,
    pcfich_encode,
    pcfich_re_subcarriers,
    pcfich_reg_indices,
    pcfich_scrambling_sequence,
)


def test_cfi_codewords_hamming_distance():
    """Verify minimum Hamming distance between 3GPP Table 5.3.4-1 codewords is >= 20."""
    cfis = [1, 2, 3, 4]
    for i in range(len(cfis)):
        for j in range(i + 1, len(cfis)):
            cw1 = CFI_CODEWORDS[cfis[i]]
            cw2 = CFI_CODEWORDS[cfis[j]]
            dist = sum(a != b for a, b in zip(cw1, cw2))
            assert dist >= 20, f"Hamming distance between CFI {cfis[i]} and {cfis[j]} is {dist} < 20"


@pytest.mark.parametrize("pci", [0, 1, 42, 174, 503])
@pytest.mark.parametrize("num_rb_dl", [6, 15, 25, 50, 75, 100])
def test_pcfich_re_mapping_and_pilot_collision(pci, num_rb_dl):
    """Verify exactly 16 subcarriers are mapped and none collide with CRS pilots."""
    subcarriers = pcfich_re_subcarriers(pci, num_rb_dl)
    assert len(subcarriers) == 16
    assert len(np.unique(subcarriers)) == 16
    assert np.all(subcarriers >= 0)
    assert np.all(subcarriers < 12 * num_rb_dl)

    # Check collision with CRS pilots in symbol 0:
    v_shift = pci % 6
    for k in subcarriers:
        assert (k % 6) != (v_shift % 6), f"Subcarrier {k} collides with CRS port 0"
        assert (k % 6) != ((v_shift + 3) % 6), f"Subcarrier {k} collides with CRS port 1"


@pytest.mark.parametrize("cfi", [1, 2, 3])
@pytest.mark.parametrize("pci", [0, 42, 174, 503])
@pytest.mark.parametrize("subframe", [0, 1, 5, 9])
def test_pcfich_noiseless_decoding_all_cfi_and_pcis(cfi, pci, subframe):
    """Verify 100% bit-exact noiseless CFI recovery across cell IDs and subframes."""
    num_rb = 50
    symbols = pcfich_encode(cfi, subframe, pci, num_rb)
    assert len(symbols) == 16

    # Place symbols on physical grid
    grid_sym0 = np.zeros(12 * num_rb, dtype=np.complex128)
    re_indices = pcfich_re_subcarriers(pci, num_rb)
    grid_sym0[re_indices] = symbols

    decoder = PCFICHDecoder(pci, num_rb_dl=num_rb)
    res = decoder.decode(grid_sym0, subframe=subframe)

    assert res.cfi == cfi
    assert res.confidence > 20.0
    assert res.pci == pci
    assert res.subframe == subframe


@pytest.mark.parametrize("cfi", [1, 2, 3])
def test_pcfich_decoding_under_awgn_noise(cfi):
    """Verify robust decoding under 0 dB SNR AWGN noise over 100 Monte Carlo runs."""
    pci = 42
    num_rb = 25
    subframe = 5
    decoder = PCFICHDecoder(pci, num_rb_dl=num_rb)
    re_indices = pcfich_re_subcarriers(pci, num_rb)
    symbols = pcfich_encode(cfi, subframe, pci, num_rb)

    rng = np.random.default_rng(2026)
    noise_sigma = 1.0 / np.sqrt(2.0)  # ~0 dB SNR relative to unit power symbols

    correct = 0
    runs = 100
    for _ in range(runs):
        noisy_symbols = symbols + noise_sigma * (rng.normal(size=16) + 1j * rng.normal(size=16))
        grid = np.zeros(12 * num_rb, dtype=np.complex128)
        grid[re_indices] = noisy_symbols

        obs = decoder.decode(grid, subframe=subframe)
        if obs.cfi == cfi:
            correct += 1

    assert correct == runs, f"Expected 100% decoding at 0 dB SNR due to d_min=21, got {correct}/{runs}"


def test_pcfich_decoding_with_channel_fading():
    """Verify decoding with Rayleigh channel fading and equalizer compensation."""
    pci = 174
    num_rb = 50
    subframe = 5
    cfi = 2

    symbols = pcfich_encode(cfi, subframe, pci, num_rb)
    re_indices = pcfich_re_subcarriers(pci, num_rb)

    # Complex fading channel H(k)
    rng = np.random.default_rng(42)
    h_grid = (rng.normal(size=12 * num_rb) + 1j * rng.normal(size=12 * num_rb)) / np.sqrt(2.0)

    grid = np.zeros(12 * num_rb, dtype=np.complex128)
    grid[re_indices] = symbols * h_grid[re_indices]

    decoder = PCFICHDecoder(pci, num_rb_dl=num_rb)
    obs = decoder.decode(grid, channel_estimates=h_grid, subframe=subframe)

    assert obs.cfi == cfi
    assert obs.confidence > 10.0
