"""Independent PBCH test assembly from frozen external BCH encoder vectors.

No production LTE functions are imported. This is a synthetic central-six-RB
FDD/normal-CP waveform, not a full eNodeB or a hardware acquisition.
"""
from pathlib import Path
import binascii

import numpy as np
from scipy.signal import resample_poly


PATH = Path(__file__).with_name("fixtures")/"lte_bch_reference.npz"


def gold(initial, length):
    a, b, values = 1, int(initial), []
    for n in range(1600+length):
        if n >= 1600:
            values.append((a ^ b) & 1)
        a = (a >> 1) | (((a ^ (a >> 3)) & 1) << 30)
        b = (b >> 1) | (((b ^ (b >> 1) ^ (b >> 2) ^ (b >> 3)) & 1) << 30)
    return np.array(values)


def fixture(index=4, ports=1):
    with np.load(PATH, allow_pickle=False) as data:
        row = 3*index+(1, 2, 4).index(ports)
        return data["bits"][row, :24].copy(), data["matched"][row].copy()


def encode_payload(payload, ports):
    """Test-only scalar encoder, independently checked against all 90 C vectors.

    Used only to advance SFN in longer synthetic recordings. The original first
    block still uses frozen external codewords, not a production encoder.
    """
    checksum = binascii.crc_hqx(np.packbits(payload).tobytes(), 0)
    mask = [0]*16 if ports == 1 else ([1]*16 if ports == 2 else [0, 1]*8)
    bits = list(map(int, payload))+[((checksum >> (15-i)) & 1) ^ mask[i] for i in range(16)]
    state = sum(bits[-1-j] << j for j in range(6))
    code = []
    for bit in bits:
        state = (state << 1) | bit
        code.extend((state & p).bit_count() % 2 for p in (109, 79, 87))
    collection = []
    for stream in range(3):
        matrix = np.r_[np.full(24, -1), np.array(code[stream::3])].reshape(2, 32)
        for column in range(32):
            permuted = int(f"{column:05b}"[::-1], 2) ^ 1
            collection.extend(int(v) for v in matrix[:, permuted] if v != -1)
    return np.array((collection*16), np.uint8)


def frame_fixture(index, ports, frame, first_quarter=0):
    payload, code = fixture(index, ports)
    advance = (first_quarter+frame)//4
    if advance:
        msb = sum(int(payload[6+i]) << (7-i) for i in range(8))
        payload[6:14] = [((msb+advance) % 256 >> k) & 1 for k in range(7, -1, -1)]
        code = encode_payload(payload, ports)
    return payload, code


def slot_grids(pci, ports, *, index=4, quarter=0, frame=0, first_quarter=0):
    payload, code = frame_fixture(index, ports, frame, first_quarter)
    scrambled = (code.astype(int)+gold(pci, 1920)) % 2
    q = scrambled[480*quarter:480*(quarter+1)].reshape(240, 2)
    symbols = ((1-2*q[:, 0])+1j*(1-2*q[:, 1]))/np.sqrt(2)
    transmitted = np.zeros((ports, 240), complex)
    if ports == 1:
        transmitted[0] = symbols
    else:
        step = ports
        pairs = [(0, 1, 0)] if ports == 2 else [(0, 2, 0), (1, 3, 2)]
        for a, b, offset in pairs:
            for i in range(offset, 240, step):
                transmitted[a, i], transmitted[a, i+1] = symbols[i], symbols[i+1]
                transmitted[b, i], transmitted[b, i+1] = -symbols[i+1].conjugate(), symbols[i].conjugate()
        transmitted /= np.sqrt(2)
    grids = np.zeros((ports, 7, 72), complex)
    positions = [(l, k) for l in range(4) for k in range(72) if l > 1 or (k-pci) % 3]
    for port in range(ports):
        for i, (l, k) in enumerate(positions):
            grids[port, l, k] = transmitted[port, i]
        for l in ((0, 4) if port < 2 else (1,)):
            v = ((0 if l == 0 else 3) if port == 0 else
                 (3 if l == 0 else 0) if port == 1 else (3 if port == 2 else 0))
            c = gold(1024*(7*2+l+1)*(2*pci+1)+2*pci+1, 440)
            for m in range(12):
                k = 6*m+(v+pci) % 6
                mprime = m+104
                grids[port, l, k] = ((1-2*c[2*mprime])+1j*(1-2*c[2*mprime+1]))/np.sqrt(2)
    return payload, grids


def waveform(pci=42, *, ports=1, index=4, frames=4, first_quarter=0,
             rate=2_000_000., onset=431, cfo=0., snr_db=None, seed=771,
             channel=None, corrupt_pbch=False):
    with np.load(Path(__file__).with_name("fixtures")/"lte_sync_sequences.npz", allow_pickle=False) as reference:
        pss, sss = reference["pss"][pci % 3], reference["sss"][pci]
    rng = np.random.default_rng(seed)
    carriers = np.r_[np.arange(-36, 0), np.arange(1, 37)] % 128
    sync_carriers = np.r_[np.arange(-31, 0), np.arange(1, 32)] % 128
    if channel is None:
        channel = [1, .6+.3j, -.2+.7j, .5-.4j][:ports]
    parts = []
    for frame in range(frames):
        _, grids = slot_grids(pci, ports, index=index, quarter=(first_quarter+frame) % 4,
                              frame=frame, first_quarter=first_quarter)
        for symbol in range(140):
            grid = np.zeros(128, complex)
            grid[carriers] = ((2*rng.integers(0, 2, 72)-1)+1j*(2*rng.integers(0, 2, 72)-1))/np.sqrt(2)
            if 7 <= symbol < 14:
                grid[:] = 0
                combined = sum(channel[p]*grids[p, symbol-7] for p in range(ports))
                if corrupt_pbch:
                    for k in range(72):
                        if symbol-7 > 1 or (k-pci) % 3:
                            combined[k] = rng.normal()+1j*rng.normal()
                grid[carriers] = combined
            elif symbol in (5, 6, 75, 76):
                grid[:] = 0
                grid[sync_carriers] = pss if symbol in (6, 76) else sss[int(symbol == 75)]
                grid *= channel[0]
            useful = np.fft.ifft(grid)*np.sqrt(128)
            cp = 10 if symbol % 7 == 0 else 9
            parts.append(np.r_[useful[-cp:], useful])
    native = np.concatenate(parts)
    signal = resample_poly(native, 25, 24) if rate == 2_000_000 else native
    power = float(np.mean(np.abs(signal)**2))
    iq = np.r_[np.zeros(onset), signal].astype(complex)
    iq *= np.exp(2j*np.pi*cfo*np.arange(len(iq))/rate)
    if snr_db is not None:
        iq += np.sqrt(power/10**(snr_db/10)/2)*(rng.normal(size=len(iq))+1j*rng.normal(size=len(iq)))
    return iq.astype(np.complex64)
