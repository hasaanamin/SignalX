"""Independent sync-bearing OFDM test IQ; not a complete LTE transmitter.

Sequences come from the compiled external reference fixture. Resource assembly,
CP and impairment injection do not import the production LTE implementation.
"""
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


REFERENCE_PATH = Path(__file__).with_name("fixtures")/"lte_sync_sequences.npz"


def waveform(pci=42, *, frames=3, rate=2_000_000., cfo=0., snr_db=None,
             onset=431, seed=771, multipath=None, tail=0, with_sync=True):
    with np.load(REFERENCE_PATH, allow_pickle=False) as reference:
        pss = reference["pss"][pci % 3]
        sss = reference["sss"][pci].copy()
    rng = np.random.default_rng(seed)
    symbols = []
    data_bins = np.r_[np.arange(-36, 0), np.arange(1, 37)] % 128
    sync_bins = np.r_[np.arange(-31, 0), np.arange(1, 32)] % 128
    for frame in range(frames):
        for symbol in range(140):
            grid = np.zeros(128, np.complex128)
            grid[data_bins] = ((2*rng.integers(0, 2, 72)-1)+1j*(2*rng.integers(0, 2, 72)-1))/np.sqrt(2)
            if with_sync and symbol in (5, 6, 75, 76):
                grid[:] = 0
                grid[sync_bins] = pss if symbol in (6, 76) else sss[int(symbol == 75)]
            useful = np.fft.ifft(grid)*np.sqrt(128)
            cp = 10 if symbol % 7 == 0 else 9
            symbols.append(np.r_[useful[-cp:], useful])
    native = np.concatenate(symbols)
    assert len(native) == frames*19200
    signal = resample_poly(native, 25, 24) if rate == 2_000_000 else native
    if multipath is not None:
        signal = np.convolve(signal, multipath, mode="full")[:len(signal)]
    signal_power = float(np.mean(np.abs(signal)**2))
    iq = np.r_[np.zeros(onset), signal, np.zeros(tail)].astype(np.complex128)
    iq *= np.exp(2j*np.pi*cfo*np.arange(len(iq))/rate)
    if snr_db is not None:
        sigma = np.sqrt(signal_power/10**(snr_db/10)/2)
        iq += sigma*(rng.normal(size=len(iq))+1j*rng.normal(size=len(iq)))
    return iq.astype(np.complex64)
