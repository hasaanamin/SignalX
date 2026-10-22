"""LTE BCH decoding primitives (TS 36.212 §§5.1, 5.3.1).

Soft-bit convention is positive for bit zero. No transmitter/test truth is used.
The tail-biting decoder evaluates all 64 initial states, with equal start/end
state enforced; it is not a zero-terminated Viterbi approximation.
"""
from functools import lru_cache
import operator

import numpy as np


def gold_sequence(initialization: int, length: int):
    """TS 36.211 §7.2, Nc=1600. Internal bounded callers supply validated sizes."""
    initialization, length = operator.index(initialization), operator.index(length)
    # Advance 31 GF(2) bits per Python iteration. The low 28 output bits
    # depend only on the current register; the final three additionally use
    # already-known low output bits. This is the exact recurrence, not a
    # statistical substitute, cached IQ or a PCI-specific generated answer.
    mask, low = (1 << 31)-1, (1 << 28)-1
    a, b, bits = 1, initialization & ((1 << 31)-1), 0
    count = 1600+length
    for offset in range(0, count, 31):
        bits |= (a ^ b) << offset
        aa = a | (((a ^ (a >> 3)) & low) << 31)
        bb = b | (((b ^ (b >> 1) ^ (b >> 2) ^ (b >> 3)) & low) << 31)
        a = (aa ^ (aa >> 3)) & mask
        b = (bb ^ (bb >> 1) ^ (bb >> 2) ^ (bb >> 3)) & mask
    size = ((count+30)//31*31+7)//8
    raw = np.frombuffer(bits.to_bytes(size, 'little'), dtype=np.uint8)
    return np.unpackbits(raw, bitorder='little')[1600:count].copy()


def crc16(bits):
    """MSB-first CRC-16, g(D)=D^16+D^12+D^5+1, zero initialization."""
    register = 0
    for bit in bits:
        feedback = (register >> 15) ^ int(bit)
        register = (register << 1) & 65535
        if feedback:
            register ^= 0x1021
    return register


@lru_cache(maxsize=1)
def rate_match_indices():
    # 40 bits/stream, two 32-column rows, 24 leading NULL positions.
    permutation = (1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
                   0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30)
    collection = [3*(32*r+c-24)+s for s in range(3) for c in permutation
                  for r in range(2) if 32*r+c >= 24]
    result = np.resize(np.array(collection, np.intp), 1920)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=1)
def _trellis():
    destination = np.arange(64)
    predecessors = np.stack((destination >> 1, (destination >> 1)+32))
    registers = (predecessors << 1) | (destination & 1)
    signs = np.array([[[1-2*(int(reg & p).bit_count() % 2) for p in (0x6d, 0x4f, 0x57)]
                       for reg in row] for row in registers], float)
    predecessors.flags.writeable = signs.flags.writeable = False
    return predecessors, signs


def decode_tail_biting(soft, *, backend='auto', threads=1):
    if backend not in ('auto', 'reference', 'numba') or type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('Invalid BCH execution backend/thread count')
    soft = np.asarray(soft, dtype=np.float64)
    if soft.shape != (120,) or not np.all(np.isfinite(soft)):
        raise ValueError("BCH requires 120 finite soft code bits")
    # Scaling prevents accumulated scores overflowing without altering decisions.
    scale = float(np.max(np.abs(soft)))
    if scale == 0:
        return None
    soft = soft.reshape(40, 3)/scale
    predecessors, signs = _trellis()
    if backend != 'reference':
        from ._native import initialize, run
        if initialize(backend, threads) == 'numba':
            return run(soft, predecessors, signs, threads)
    elif type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('Numba threads must be an integer from 1 to 4')
    metrics = np.full((64, 64), -np.inf)
    np.fill_diagonal(metrics, 0.)
    decisions = np.empty((40, 64, 64), bool)
    for t in range(40):
        branch = np.einsum("ijk,k->ij", signs, soft[t], optimize=False)
        a = metrics[:, predecessors[0]]+branch[0]
        b = metrics[:, predecessors[1]]+branch[1]
        decisions[t] = b > a
        metrics = np.maximum(a, b)
    initial = int(np.argmax(metrics.diagonal()))
    state = initial
    bits = np.empty(40, np.uint8)
    for t in range(39, -1, -1):
        bits[t] = state & 1
        state = int(predecessors[int(decisions[t, initial, state]), state])
    assert state == initial
    return bits


@lru_cache(maxsize=16)
def _pbch_signs(pci):
    signs = 1-2*gold_sequence(pci, 1920).astype(float)
    signs.flags.writeable = False
    return signs


def recover_bch(frames, pci, current_quarter, ports, *, backend='auto', threads=1):
    """Recover one block from <=4 chronological, same-block PBCH quarters.

    frames contains (480,) soft arrays, ending at current_quarter. Gaps and
    40-ms boundaries must be excluded by the caller before soft combination.
    """
    signs = _pbch_signs(pci)
    recovered = np.zeros(120)
    first = current_quarter-len(frames)+1
    for quarter, soft in enumerate(frames, first):
        indices = slice(quarter*480, (quarter+1)*480)
        recovered += np.bincount(rate_match_indices()[indices],
                                 weights=soft*signs[indices], minlength=120)
    bits = decode_tail_biting(recovered, backend=backend, threads=threads)
    if bits is None:
        return None
    mask = np.zeros(16, np.uint8) if ports == 1 else (
        np.ones(16, np.uint8) if ports == 2 else np.tile([0, 1], 8).astype(np.uint8))
    received_crc = bits[24:] ^ mask
    if crc16(np.r_[bits[:24], received_crc]) != 0:
        return None
    return tuple(int(b) for b in bits[:24])
