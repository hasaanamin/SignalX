"""FDD normal-CP PBCH: CRS estimation, transmit-diversity and BCH recovery.

Only the central six resource blocks are needed. Four-port decoding uses the
two Alamouti pairs (0,2) and (1,3), not a spatial-multiplexing assumption.
"""
from functools import lru_cache

import numpy as np
from scipy.fft import fft

from .bch import gold_sequence, recover_bch
from .mib import LTEMIB


CENTRAL_BINS = np.r_[np.arange(-36, 0), np.arange(1, 37)] % 128
CENTRAL_BINS.flags.writeable = False
_SLOT_TIME = np.arange(960)/1_920_000.
_CP_INDICES = np.array([position+k for position,cp in
    ((0,10),(138,9),(275,9),(412,9),(549,9),(686,9),(823,9)) for k in range(3,cp)])
_GRID_INDICES = np.array((8,145,282,419,556))[:,None]+np.arange(128)
_CHANNEL_FREQUENCIES = np.r_[np.arange(-36,0),np.arange(1,37)]
_CHANNEL_SMOOTH = np.ones(3)
_CHANNEL_DENOMINATOR = np.convolve(np.ones(12),_CHANNEL_SMOOTH,'same')
for _constant in (_SLOT_TIME,_CP_INDICES,_GRID_INDICES,_CHANNEL_FREQUENCIES,_CHANNEL_SMOOTH,_CHANNEL_DENOMINATOR):
    _constant.flags.writeable = False


@lru_cache(maxsize=16)
def _crs(pci, symbol):
    # Slot ns=1 of subframe zero, normal CP. Selecting the central 6 RB
    # cancels the unknown system bandwidth in m'=m+110-N_RB.
    initialization = 1024*(14+symbol+1)*(2*pci+1)+2*pci+1
    bits = gold_sequence(initialization, 440).astype(float)
    result = ((1-2*bits[208:232:2])+1j*(1-2*bits[209:232:2]))/np.sqrt(2)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=16)
def pbch_resources(pci):
    # All four CRS ports are reserved irrespective of actual antenna count.
    result = np.array([(l, k) for l in range(4) for k in range(72)
                     if l >= 2 or k % 3 != pci % 3], dtype=np.intp).T
    return np.frombuffer(result.tobytes(),dtype=result.dtype).reshape(result.shape)


def _channel(grid, pci, port):
    symbols = (0, 4) if port < 2 else (1,)
    estimates = []
    for l in symbols:
        v = ((0 if l == 0 else 3) if port == 0 else
             (3 if l == 0 else 0) if port == 1 else (3 if port == 2 else 0))
        indices = np.arange((v+pci) % 6, 72, 6)
        pilot = grid[l, indices]*_crs(pci, l).conj()
        # Mild frequency smoothing; interpolation uses physical frequencies,
        # including the missing DC carrier, rather than treating DC as an RE.
        pilot = np.convolve(pilot, _CHANNEL_SMOOTH, "same")/_CHANNEL_DENOMINATOR
        frequencies = _CHANNEL_FREQUENCIES
        estimates.append(np.interp(frequencies, frequencies[indices], pilot))
    if len(estimates) == 1:
        return np.tile(estimates[0], (4, 1))
    return np.array([(1-l/4)*estimates[0]+(l/4)*estimates[1] for l in range(4)])


def _soft_symbols(received, channels, ports):
    """Channel-weighted QPSK soft symbols; avoids division at deep fades."""
    if ports == 1:
        return received*channels[0].conj()
    result = np.zeros(240, np.complex128)
    pairs = ((0, 1, np.arange(0, 240, 2)),) if ports == 2 else (
        (0, 2, np.arange(0, 240, 4)), (1, 3, np.arange(2, 240, 4)))
    for a, b, first in pairs:
        # y0 = (ha0*d0 - hb0*conj(d1))/sqrt(2)
        # conj(y1) = (conj(hb1)*d0 + conj(ha1)*conj(d1))/sqrt(2).
        aa, ab = channels[a, first], -channels[b, first]
        ba, bb = channels[b, first+1].conj(), channels[a, first+1].conj()
        y0, y1 = received[first], received[first+1].conj()
        determinant = aa*bb-ab*ba
        power0 = np.abs(ab)**2+np.abs(bb)**2
        power1 = np.abs(aa)**2+np.abs(ba)**2
        good0, good1 = power0 > 0, power1 > 0
        # ZF estimate multiplied by inverse noise variance. Singular/faded
        # pairs yield erasures, not invented finite channel/noise measurements.
        result[first[good0]] = ((bb*y0-ab*y1)*determinant.conj())[good0]/power0[good0]*np.sqrt(2)
        result[first[good1]+1] = (((-ba*y0+aa*y1)*determinant.conj())[good1]/power1[good1]*np.sqrt(2)).conj()
    return result


def slot_soft_bits(grid, pci):
    """Internal five-symbol central grid -> one 480-LLR array per port hypothesis."""
    coordinates = pbch_resources(pci)
    received = grid[tuple(coordinates)]
    channels = np.array([_channel(grid, pci, p)[tuple(coordinates)] for p in range(4)])
    result = {}
    for ports in (1, 2, 4):
        equalized = _soft_symbols(received, channels, ports)
        soft = np.column_stack((equalized.real, equalized.imag)).ravel()
        # A common scale is harmless; normalize per hypothesis/frame to keep
        # representable extreme amplitudes safe. Combining is equal-frame
        # weighting, not a claim of optimal receiver-noise likelihoods.
        maximum = float(np.max(np.abs(soft)))
        result[ports] = soft/maximum if maximum > 0 else soft
    return result


class PBCHDecoder:
    """Bounded soft-combining decoder for chronological received slot grids.

    Public input is five (l=0..4) central-72-carrier OFDM symbols from slot 1
    of subframe 0, with CFO already corrected. At most four frames are held.
    Return (MIB, antenna_ports, current_frame_mod4, combined_frame_count) or
    None for an ordinary CRC/semantic failure. No invalid MIB is published.
    """

    def __init__(self, pci, *, backend='auto', threads=1):
        if type(pci) is not int or not 0 <= pci < 504:
            raise ValueError("Invalid PCI")
        self.pci = pci
        from ._native import initialize
        self.backend = initialize(backend, threads)
        self.threads = threads
        self._frames = []
        self.attempted_frames = self.decoded_frames = 0

    @property
    def retained_frames(self):
        return len(self._frames)

    @property
    def latest_frame_start(self):
        return self._frames[-1][0] if self._frames else None

    def decode(self, grid, frame_start):
        grid = np.asarray(grid)
        if grid.shape != (5, 72) or not np.issubdtype(grid.dtype, np.complexfloating) or not np.all(np.isfinite(grid)):
            raise ValueError("PBCH grid must be finite complex (5,72)")
        if type(frame_start) is not int or frame_start < 0:
            raise ValueError("Invalid positioned frame")
        if self._frames and frame_start <= self._frames[-1][0]:
            raise ValueError("PBCH frames must be strictly chronological")
        grid = grid.astype(np.complex128)
        scale = max(float(np.max(np.abs(grid.real))), float(np.max(np.abs(grid.imag))))
        if scale > 0:
            grid = grid/scale
        # An explicit nonconsecutive frame cannot contribute to one codeword.
        frames = self._frames if not self._frames or frame_start == self._frames[-1][0]+19200 else []
        soft = slot_soft_bits(grid, self.pci)
        frames = (frames+[(frame_start, soft)])[-4:]
        self.attempted_frames += 1
        # Prefer the newest frame alone. If it fails, combine only sequences
        # lying entirely within the hypothesized 40-ms BCH block.
        for count in range(1, len(frames)+1):
            found = []
            batch = None
            if self.backend == 'numba' and self.threads == 1:
                from ._native import _recover_batch
                from .bch import _pbch_signs, rate_match_indices, _trellis
                inputs = np.array([[f[1][p] for p in (1,2,4)] for f in frames[-count:]])
                batch = _recover_batch(inputs, _pbch_signs(self.pci), rate_match_indices(), *_trellis())
            for ports in (1, 2, 4):
                for quarter in range(count-1, 4):
                    if batch is None:
                        bits = recover_bch([f[1][ports] for f in frames[-count:]], self.pci, quarter, ports,
                                           backend=self.backend, threads=self.threads)
                    else:
                        port_index = (1,2,4).index(ports)
                        bits = tuple(map(int,batch[1][port_index,quarter])) if batch[0][port_index,quarter] else None
                    if bits is None:
                        continue
                    try:
                        mib = LTEMIB(bits)
                    except ValueError:
                        continue
                    found.append((mib, ports, quarter, count))
            if found:
                # Ambiguous CRC-valid configurations are not silently selected.
                self._frames = frames
                if len(found) != 1:
                    return None
                self.decoded_frames += 1
                return found[0]
        self._frames = frames
        return None


def extract_slot(samples, frame_start, base, frequency_offset_hz, *, average_cfo=False):
    """Internal native-rate IQ extraction; caller guarantees a complete slot.

    CP-derived residual CFO uses only this already received slot. FFT starts
    two samples inside the CP to tolerate the synchronizer's ±1-sample peak;
    CRS estimates absorb the resulting deterministic subcarrier phase slope.
    """
    start = frame_start+960-base
    slot = np.asarray(samples[start:start+960], np.complex128).copy()
    t = _SLOT_TIME
    slot *= np.exp(-2j*np.pi*frequency_offset_hz*t)
    position = 0
    correlation = 0j
    if average_cfo:
        # More received CP pairs reduce the variance of the residual-CFO
        # estimate at low SNR. Use <=20 slots, all complete and in this epoch.
        # Start at the last available slot to avoid making earlier decoding
        # depend on the caller's unprocessed/next transport-block suffix.
        last_slot = (base+len(samples)-frame_start)//960-1
        first_slot = max(last_slot-19, (base-frame_start+959)//960)
        indices = frame_start+np.arange(first_slot,last_slot+1)[:,None]*960-base+_CP_INDICES
        # Same <=20 complete received slots; gather only their CP pairs, not
        # the full IQ history. Promote stored complex64 before multiplying.
        correlation = np.vdot(samples[indices].astype(np.complex128),
                              samples[indices+128].astype(np.complex128))
        correlation *= np.exp(-2j*np.pi*frequency_offset_hz*128/1_920_000.)
    else:
        correlation = np.vdot(slot[_CP_INDICES],slot[_CP_INDICES+128])
    residual = float(np.angle(correlation)*1_920_000/(2*np.pi*128)) if correlation != 0 else 0.
    slot *= np.exp(-2j*np.pi*residual*t)
    grid = fft(slot[_GRID_INDICES],axis=-1,workers=1)[:,CENTRAL_BINS]/np.sqrt(128)
    return grid, frequency_offset_hz+residual
