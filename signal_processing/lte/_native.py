"""Optional exact BCH kernels. No fast math or stream/lifecycle state.

One native thread is the default: these small jobs rarely justify a team.
Explicit 2..4-thread execution parallelizes independent initial trellis states,
not time steps, and restores the caller's thread-local Numba mask afterwards.
"""
from functools import lru_cache
import threading

import numpy as np

try:
    import numba
except ImportError:
    numba = None


_warm_lock = threading.Lock()


if numba is not None:
    @numba.njit(cache=True,nogil=True)
    def _group_power_slice(powers,groups,offset,count,capacity,mirror_limit):
        mirrored = min(count,max(0,mirror_limit-offset))
        for group in range(3):
            row = group*7
            for j in range(mirrored):
                k = offset+j
                head = (((powers[row,k]+powers[row+1,k])+powers[row+2,k])+powers[row+3,k])+powers[row+4,k]
                tail = powers[row+5,k]+powers[row+6,k]
                groups[2*group,k] = groups[2*group,k+capacity] = head
                groups[2*group+1,k] = groups[2*group+1,k+capacity] = tail
            for j in range(mirrored,count):
                k = offset+j
                groups[2*group,k] = (((powers[row,k]+powers[row+1,k])+powers[row+2,k])+powers[row+3,k])+powers[row+4,k]
                groups[2*group+1,k] = powers[row+5,k]+powers[row+6,k]

    @numba.njit(cache=True,nogil=True)
    def _subthreshold_tone(samples,templates,threshold):
        """Prove every received 128-sample correlation is below threshold.

        For ANY z, write x[k+1]=z*x[k]+e[k]. Summation by parts gives
        dot(x,t)=x[0]*a(z)+dot(e,b(z)); Cauchy-Schwarz bounds its remainder.
        Bounds use the entire received region, so they cover every window and
        every nonnegative incoherent integration of those windows. Failure to
        prove the bound always retains ordinary full search.
        """
        if len(samples) < 128:
            return False
        begin = max(0,len(samples)-512)
        lag = 0j
        for j in range(begin,len(samples)-1):
            lag += samples[j+1]*samples[j].conjugate()
        z = lag/abs(lag) if abs(lag) > 0 else 1+0j
        energy,error = 0.,0.
        for j in range(begin,len(samples)-1):
            value = samples[j]
            residual = samples[j+1]-z*value
            energy += value.real*value.real+value.imag*value.imag
            error += residual.real*residual.real+residual.imag*residual.imag
        if error > 1e-4*energy:
            return False  # Cheap rejection of non-tonal input, not a decision.
        minimum,maximum,total_error = np.inf,0.,0.
        for j in range(len(samples)):
            value = samples[j]
            power = value.real*value.real+value.imag*value.imag
            minimum = min(minimum,power)
            maximum = max(maximum,power)
            if j+1 < len(samples):
                residual = samples[j+1]-z*value
                total_error += residual.real*residual.real+residual.imag*residual.imag
        if maximum == 0:
            return True
        if minimum == 0:
            return False
        scale = np.sqrt(62*128*minimum)*(np.sqrt(threshold)-1e-8)
        first,remainder = np.sqrt(maximum),np.sqrt(total_error)
        for branch in range(len(templates)):
            tail = 0j
            tail_energy = 0.
            for k in range(127,0,-1):
                tail = templates[branch,k]+z*tail
                tail_energy += tail.real*tail.real+tail.imag*tail.imag
            a = templates[branch,0]+z*tail
            if first*abs(a)+remainder*np.sqrt(tail_energy) >= scale:
                return False
        return True

    @numba.njit(cache=True,nogil=True,inline="never")
    def _subspace_project(powers,energies,start,stop,repeats,capacity,energy,projections,maxima):
        count = stop-start
        energy[:count] = 0.
        projections[:,:count] = 0.
        for repeat in range(repeats):
            offset = (start-repeat*9600) % capacity
            for j in range(count):
                energy[j] += energies[offset+j]
            if len(powers) == 6:
                for group in range(6):
                    for j in range(count):
                        projections[group,j] += powers[group,offset+j]
            else:
                for group in range(3):
                    row = group*7
                    for j in range(count):
                        k = offset+j
                        total = (((powers[row,k]+powers[row+1,k])+powers[row+2,k])
                                 +powers[row+3,k])+powers[row+4,k]
                        projections[2*group,j] += total
                        projections[2*group+1,j] += powers[row+5,k]+powers[row+6,k]
        maxima[:] = 0.
        for j in range(count):
            energy[j] = 1/(62*energy[j]) if energy[j] > 0 else 0.
        for group in range(6):
            for j in range(count):
                projections[group,j] = np.sqrt(projections[group,j]*energy[j])
            maximum = 0.
            for j in range(count):
                maximum = max(maximum,projections[group,j])
            maxima[group] = maximum
        return energy,projections,maxima

    @numba.njit(cache=True,nogil=True,inline="never")
    def _subspace_select(energy,projections,maxima,cutoffs,templates,count,result,selected_branches,selected_positions,previous_count,candidates,candidate_counts):
        # Only the preceding selected entries can be nonzero. Clearing an entire
        # 27-by-query result and scanning an equally large boolean mask wastes
        # bandwidth for the usual subthreshold input. Preserve branch/time order.
        for i in range(previous_count[0]):
            result[selected_branches[i],selected_positions[i]] = 0.
        refined = 0
        # All nine CFO branches share a PSS family's projected powers. A bound
        # using the largest nonnegative coefficients and smallest cutoff covers
        # every branch. Scan positions once per family, then refine only its
        # surviving positions with each branch's original bound.
        for group in range(3):
            upper_a,upper_b,lower = 0.,0.,np.inf
            for branch in range(group,len(templates),3):
                if (cutoffs[branch,0]*maxima[2*group]
                        +cutoffs[branch,1]*maxima[2*group+1] < cutoffs[branch,2]):
                    continue
                upper_a = max(upper_a,cutoffs[branch,0])
                upper_b = max(upper_b,cutoffs[branch,1])
                lower = min(lower,cutoffs[branch,2])
            candidate_counts[group] = 0
            if upper_a*maxima[2*group]+upper_b*maxima[2*group+1] < lower:
                continue
            for j in range(count):
                if (energy[j] > 0 and upper_a*projections[2*group,j]
                        +upper_b*projections[2*group+1,j] >= lower):
                    candidates[group,candidate_counts[group]] = j
                    candidate_counts[group] += 1
        for branch in range(len(templates)):
            group = branch % 3
            a,b,threshold = cutoffs[branch,0],cutoffs[branch,1],cutoffs[branch,2]
            if a*maxima[2*group]+b*maxima[2*group+1] < threshold:
                continue
            for i in range(candidate_counts[group]):
                j = candidates[group,i]
                if a*projections[2*group,j]+b*projections[2*group+1,j] >= threshold:
                    selected_branches[refined] = branch
                    selected_positions[refined] = j
                    refined += 1
        previous_count[0] = refined
        return result,selected_branches,selected_positions,refined

    @numba.njit(cache=True,nogil=True,inline="never")
    def _subspace_dots(samples,base,start,repeats,energy,templates,result,selected_branches,selected_positions,refined,accumulated,real,imag,maximum):
        accumulated[:refined] = 0.
        for repeat in range(repeats):
            real[:refined] = 0.
            imag[:refined] = 0.
            origin = start-base-repeat*9600
            # Independent dot products run across columns; each retains its
            # original chronological 128-term accumulation, without fast math.
            for k in range(128):
                for i in range(refined):
                    value = samples[origin+selected_positions[i]+k]
                    template = templates[selected_branches[i],k]
                    real[i] += value.real*template.real-value.imag*template.imag
                    imag[i] += value.real*template.imag+value.imag*template.real
            for i in range(refined):
                accumulated[i] += real[i]*real[i]+imag[i]*imag[i]
        maximum[0] = 0.
        for i in range(refined):
            branch,j = selected_branches[i],selected_positions[i]
            result[branch,j] = accumulated[i]*energy[j]
            maximum[0] = max(maximum[0],result[branch,j])
        return result,refined


    def _subspace_refine(samples,base,powers,energies,start,stop,repeats,capacity,cutoffs,templates,workspace,limit=None):
        energy,projections,maxima = _subspace_project(powers,energies,start,stop,repeats,capacity,*workspace[:3])
        result,branches,positions,count = _subspace_select(energy,projections,maxima,cutoffs,templates,stop-start,
                                                          *workspace[3:7],*workspace[10:12])
        if limit is not None and count > limit:
            return None,count
        _subspace_dots(samples,base,start,repeats,energy,templates,result,branches,positions,count,
                       *workspace[7:10],workspace[-1])
        return result[:,:stop-start],count

    @numba.njit(cache=True,nogil=True)
    def _repair_small_powers(samples,energy,templates,powers,start,capacity,mirror_limit):
        # FFT roundoff from a loud part of a tile must not create a detection
        # in a much quieter received window. Recompute only these rare windows
        # directly. This also gives exact zero powers for zero-energy windows.
        threshold = energy.max()*1e-6
        for j in range(len(energy)):
            if energy[j] == 0 or energy[j] < threshold:
                position = (start+j) % capacity
                for branch in range(len(templates)):
                    dot = 0j
                    if energy[j] > 0:
                        for k in range(128):
                            dot += samples[j+k]*templates[branch,k]
                    value = dot.real*dot.real+dot.imag*dot.imag
                    powers[branch,position] = value
                    if position < mirror_limit:
                        powers[branch,position+capacity] = value

    @numba.njit(cache=True,nogil=True)
    def _window_energy(samples):
        squared = np.empty(len(samples))
        for j in range(len(samples)):
            squared[j] = samples[j].real*samples[j].real+samples[j].imag*samples[j].imag
        # Seven nonnegative dyadic reductions replace 128 passes. Advancing
        # left-to-right reads each future term before it is modified; no running
        # subtraction can erase a quiet window after a much louder one.
        for stride in numba.literal_unroll((1,2,4,8,16,32,64)):
            for j in range(len(samples)-2*stride+1):
                squared[j] += squared[j+stride]
        return squared[:len(samples)-127]

    @numba.njit(cache=True,nogil=True)
    def _ring_scores(powers, energies, start, stop, repeats, capacity):
        count = stop-start
        result = np.empty((len(powers),count))
        energy = np.zeros(count)
        offsets = np.empty(repeats,np.int64)
        for repeat in range(repeats):
            offset = (start-repeat*9600) % capacity
            offsets[repeat] = offset
            for j in range(count):
                energy[j] += energies[offset+j]
        for j in range(count):
            energy[j] = 1/(62*energy[j]) if energy[j] > 0 else 0.
        if repeats == 4:
            a,b,c,d = offsets[0],offsets[1],offsets[2],offsets[3]
            for branch in range(len(powers)):
                for j in range(count):
                    total = ((powers[branch,a+j]+powers[branch,b+j])
                             +powers[branch,c+j])+powers[branch,d+j]
                    result[branch,j] = total*energy[j]
            return result
        for branch in range(len(powers)):
            for j in range(count):
                result[branch,j] = 0.
            for repeat in range(repeats):
                offset = offsets[repeat]
                for j in range(count):
                    result[branch,j] += powers[branch,offset+j]
            for j in range(count):
                result[branch,j] *= energy[j]
        return result

    @numba.njit(cache=True, nogil=True)
    def _correlation_powers(correlations, destination, offset, count, capacity, source_offset, mirror_limit):
        mirrored = min(count,max(0,mirror_limit-offset))
        for branch in range(len(correlations)):
            for j in range(mirrored):
                value = correlations[branch, source_offset+j]
                real, imag = value.real, value.imag
                power = real*real+imag*imag
                destination[branch, offset+j] = power
                destination[branch, offset+capacity+j] = power
            for j in range(mirrored,count):
                value = correlations[branch, source_offset+j]
                real, imag = value.real, value.imag
                destination[branch, offset+j] = real*real+imag*imag

    @numba.njit(cache=True, nogil=True)
    def _frequency_dots(samples, templates):
        """Small exact complex reductions without a BLAS thread-team dispatch."""
        result = np.empty((len(samples),len(templates)),np.complex128)
        for repeat in range(len(samples)):
            for frequency in range(len(templates)):
                total = 0j
                for j in range(128):
                    total += samples[repeat,j]*templates[frequency,j]
                result[repeat,frequency] = total
        return result

    @numba.njit(cache=True, nogil=True)
    def _tracking_scores(samples, positions, repeats, templates):
        """Direct received-window reductions; no prefix subtraction/fast math."""
        scores = np.zeros((len(positions),len(templates)),np.float64)
        for k in range(len(positions)):
            energy = 0.
            powers = np.zeros(len(templates))
            for repeat in range(repeats):
                start = positions[k]-repeat*9600
                local_energy = 0.
                for j in range(128):
                    value = samples[start+j]
                    local_energy += value.real*value.real+value.imag*value.imag
                energy += local_energy
                for branch in range(len(templates)):
                    dot = 0j
                    for j in range(128):
                        dot += samples[start+j]*templates[branch,j]
                    powers[branch] += dot.real*dot.real+dot.imag*dot.imag
            if energy > 0:
                for branch in range(len(templates)):
                    scores[k,branch] = powers[branch]/(62*energy)
        return scores

    @numba.njit(cache=True, nogil=True)
    def _sss_scores(bank, equalized):
        output = np.empty((len(equalized),len(bank)),np.complex128)
        for repeat in range(len(equalized)):
            for hypothesis in range(len(bank)):
                total = 0j
                for j in range(62):
                    total += bank[hypothesis,j]*equalized[repeat,j]
                output[repeat,hypothesis] = total
        return output

    @numba.njit(cache=True, nogil=True)
    def _integrated_sss(spectra, bins, pss, bank):
        """Received PSS/SSS channel smoothing and all 336 SSS hypotheses.

        Accumulate in chronological repeat order, with the existing alternating
        half-frame permutation. Each channel half has its own truncated 11-bin
        moving average; no smoothing crosses DC. No fast math or BLAS team.
        """
        repeats = spectra.shape[1]
        accumulated = np.zeros(336, np.complex128)
        total_energy = 0.
        for repeat in range(repeats):
            channel = np.empty(62, np.complex128)
            equalized = np.empty(62, np.complex128)
            for j in range(62):
                channel[j] = spectra[0, repeat, bins[j]]*pss[j].conjugate()
            for j in range(62):
                base = 0 if j < 31 else 31
                lo, hi = max(base, j-5), min(base+31, j+6)
                smooth = 0j
                for k in range(lo, hi):
                    smooth += channel[k]
                value = spectra[1, repeat, bins[j]]*(smooth/(hi-lo)).conjugate()
                equalized[j] = value
                total_energy += value.real*value.real+value.imag*value.imag
            for hypothesis in range(336):
                index = hypothesis ^ (repeat % 2)
                dot = 0j
                for j in range(62):
                    dot += bank[index, j]*equalized[j]
                accumulated[hypothesis] += dot
        scores = np.zeros(336)
        if total_energy > 0:
            for j in range(336):
                value = accumulated[j]
                scores[j] = (value.real*value.real+value.imag*value.imag)/(repeats*62*total_energy)
        return scores

    @numba.njit(cache=True, nogil=True)
    def _branches(soft, signs):
        result = np.empty((40, 2, 64), np.float64)
        for t in range(40):
            for p in range(2):
                for s in range(64):
                    result[t, p, s] = (signs[p, s, 0]*soft[t, 0]
                        + signs[p, s, 1]*soft[t, 1] + signs[p, s, 2]*soft[t, 2])
        return result

    @numba.njit(cache=True, nogil=True)
    def _state(branches, predecessors, initial, decisions):
        current = np.full(64, -np.inf)
        following = np.empty(64, np.float64)
        current[initial] = 0.
        for t in range(40):
            for state in range(64):
                a = current[predecessors[0, state]]+branches[t, 0, state]
                b = current[predecessors[1, state]]+branches[t, 1, state]
                choose_b = b > a  # Exact reference tie rule, no approximate TB.
                decisions[t, initial, state] = choose_b
                following[state] = b if choose_b else a
            current, following = following, current
        return current[initial]

    @numba.njit(cache=True, nogil=True)
    def _trace(scores, decisions, predecessors):
        initial = 0
        for state in range(1, 64):
            if scores[state] > scores[initial]:
                initial = state
        state = initial
        bits = np.empty(40, np.uint8)
        for t in range(39, -1, -1):
            bits[t] = state & 1
            state = predecessors[int(decisions[t, initial, state]), state]
        assert state == initial
        return bits

    @numba.njit(cache=True, nogil=True)
    def _serial(soft, predecessors, signs):
        branches = _branches(soft, signs)
        # Destination-major metrics make the 64 independent initial states a
        # contiguous inner loop. LLVM can SIMD this without gather loads or
        # launching a thread team for each tiny trellis. Still exact 64-state TB.
        current = np.full((64, 64), -np.inf)
        following = np.empty((64, 64), np.float64)
        for initial in range(64):
            current[initial, initial] = 0.
        decisions = np.empty((40, 64, 64), np.bool_)  # time, destination, initial
        for t in range(40):
            for state in range(64):
                p0, p1 = predecessors[0, state], predecessors[1, state]
                b0, b1 = branches[t, 0, state], branches[t, 1, state]
                for initial in range(64):
                    a = current[p0, initial]+b0
                    b = current[p1, initial]+b1
                    choose_b = b > a
                    decisions[t, state, initial] = choose_b
                    following[state, initial] = b if choose_b else a
            current, following = following, current
        scores = np.empty(64, np.float64)
        for initial in range(64):
            scores[initial] = current[initial, initial]
        return _trace(scores, decisions.transpose(0, 2, 1), predecessors)

    @numba.njit(cache=True, nogil=True)
    def _closed_candidates(branches, predecessors):
        # Free-start Viterbi is an upper bound for EVERY closed-start path
        # ending in the same state. Obtain one feasible closed-path lower bound
        # and discard only states whose upper bound is STRICTLY smaller.
        current = np.zeros(64)
        following = np.empty(64)
        origins = np.arange(64)
        next_origins = np.empty(64, np.int64)
        for t in range(40):
            for state in range(64):
                p0, p1 = predecessors[0, state], predecessors[1, state]
                a = current[p0]+branches[t, 0, state]
                b = current[p1]+branches[t, 1, state]
                following[state] = b if b > a else a
                next_origins[state] = origins[p1] if b > a else origins[p0]
            current, following = following, current
            origins, next_origins = next_origins, origins
        upper = current.copy()
        initial = int(np.argmax(upper))
        # A free-start survivor whose origin equals its destination is itself
        # a feasible closed path. Its score is an additional rigorous lower
        # bound; when the global best survivor closes, no forced-start pass is
        # needed. This changes neither tie rules nor the exact TB optimum.
        closed_lower = -np.inf
        for state in range(64):
            if origins[state] == state and upper[state] > closed_lower:
                closed_lower = upper[state]
        if origins[initial] == initial:
            return np.flatnonzero(upper >= upper[initial])
        current[:] = -np.inf
        current[initial] = 0.
        for t in range(40):
            for state in range(64):
                a = current[predecessors[0, state]]+branches[t, 0, state]
                b = current[predecessors[1, state]]+branches[t, 1, state]
                following[state] = b if b > a else a
            current, following = following, current
        lower = max(current[initial], closed_lower)
        return np.flatnonzero(upper >= lower)

    @numba.njit(cache=True, nogil=True)
    def _serial_bounded(soft, predecessors, signs):
        branches = _branches(soft, signs)
        candidates = _closed_candidates(branches, predecessors)
        n = len(candidates)
        if n > 32:
            return _serial(soft, predecessors, signs)  # Keep the SIMD reference fallback.
        current = np.full((64, n), -np.inf)
        following = np.empty((64, n))
        decisions = np.empty((40, 64, n), np.bool_)
        for k in range(n):
            current[candidates[k], k] = 0.
        for t in range(40):
            for state in range(64):
                p0, p1 = predecessors[0, state], predecessors[1, state]
                b0, b1 = branches[t, 0, state], branches[t, 1, state]
                for k in range(n):
                    a = current[p0, k]+b0
                    b = current[p1, k]+b1
                    choose_b = b > a
                    decisions[t, state, k] = choose_b
                    following[state, k] = b if choose_b else a
            current, following = following, current
        # Candidates are ascending; strict > preserves the reference's first
        # argmax and predecessor tie rules, including quantized inputs.
        best = 0
        for k in range(1, n):
            if current[candidates[k], k] > current[candidates[best], best]:
                best = k
        initial = state = candidates[best]
        bits = np.empty(40, np.uint8)
        for t in range(39, -1, -1):
            bits[t] = state & 1
            state = predecessors[int(decisions[t, state, best]), state]
        assert state == initial
        return bits

    @numba.njit(cache=True, nogil=True)
    def _recover_batch(frames, scrambler, indices, predecessors, signs):
        """All existing port/quarter hypotheses, with exact ordered recovery.

        Frame-local accumulation then chronological addition matches bincount
        followed by += in recover_bch. No hypothesis, tie or CRC is omitted.
        """
        count = len(frames)
        valid = np.zeros((3, 4), np.bool_)
        payloads = np.zeros((3, 4, 24), np.uint8)
        for port in range(3):
            for quarter in range(count-1, 4):
                recovered = np.zeros(120)
                for frame in range(count):
                    offset = (quarter-count+1+frame)*480
                    local = np.zeros(120)
                    for k in range(480):
                        local[indices[offset+k]] += frames[frame, port, k]*scrambler[offset+k]
                    recovered += local
                scale = np.max(np.abs(recovered))
                if scale == 0:
                    continue
                bits = _serial_bounded((recovered/scale).reshape(40, 3), predecessors, signs)
                register = 0
                for k in range(40):
                    bit = int(bits[k])
                    if k >= 24:
                        if port == 1:
                            bit ^= 1
                        elif port == 2:
                            bit ^= (k-24) % 2
                    feedback = (register >> 15) ^ bit
                    register = (register << 1) & 65535
                    if feedback:
                        register ^= 0x1021
                if register == 0:
                    valid[port, quarter] = True
                    payloads[port, quarter] = bits[:24]
        return valid, payloads

    @numba.njit(cache=True, nogil=True, parallel=True)
    def _parallel(soft, predecessors, signs):
        branches = _branches(soft, signs)
        decisions = np.empty((40, 64, 64), np.bool_)
        scores = np.empty(64, np.float64)
        for initial in numba.prange(64):
            scores[initial] = _state(branches, predecessors, initial, decisions)
        return _trace(scores, decisions, predecessors)


def run(soft, predecessors, signs, threads=1):
    if numba is None:
        raise ImportError('Numba acceleration requires signal-processing[acceleration]')
    if type(threads) is not int or not 1 <= threads <= min(4, numba.config.NUMBA_NUM_THREADS):
        raise ValueError('Numba threads must be an available integer from 1 to 4')
    if threads == 1:
        return _serial_bounded(soft, predecessors, signs)
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        return _parallel(soft, predecessors, signs)
    finally:
        numba.set_num_threads(previous)


def initialize(backend='auto', threads=1):
    if backend not in ('auto', 'reference', 'numba'):
        raise ValueError('BCH backend must be auto, reference or numba')
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('Numba threads must be an integer from 1 to 4')
    return _initialize_cached(backend, threads)


def initialize_sync(backend='auto'):
    """Pre-warm pure synchronization kernels before accepting positioned IQ."""
    if backend not in ('auto','numpy','numba'):
        raise ValueError('Sync backend must be auto, numpy or numba')
    if backend == 'numpy' or (backend == 'auto' and numba is None):
        return 'numpy'
    if numba is None:
        raise ImportError('Numba sync requires signal-processing[acceleration]')
    _warm_sync()
    return 'numba'


@lru_cache(maxsize=1)
def _warm_sync():
    _correlation_powers(np.zeros((1,512),np.complex128),np.empty((1,1024)),0,385,512,127,385)
    with _warm_lock:
        _subspace_refine(np.frombuffer(bytes(40000*16),np.complex128),0,
                         np.zeros((21,2048)),np.ones(2048),30000,30008,4,1024,
                         np.ones((27,3)),np.ones((27,128),np.complex128),subspace_workspace(8,27))
        _repair_small_powers(np.frombuffer(bytes(512*16),np.complex128),np.ones(385),
                             np.ones((1,128),np.complex128),np.empty((1,1024)),0,512,385)
        _window_energy(np.frombuffer(bytes(512*16),np.complex128))
        _group_power_slice(np.zeros((21,1024)),np.empty((6,1024)),0,385,512,385)
        _subthreshold_tone(np.frombuffer(bytes(512*16),np.complex128),np.ones((27,128),np.complex128),.1)
        _ring_scores(np.zeros((1,1024)),np.zeros(1024),0,64,1,512)
        # Same readonly/memory-layout signatures used by immutable IQ history
        # and standards banks. No first-input compilation or stream advancement.
        samples = np.frombuffer(bytes(40000*16),np.complex128)
        bank = np.frombuffer(np.ones((336,62)).tobytes(),np.float64).reshape(336,62)
        _tracking_scores(samples,np.array([30000]),4,np.ones((27,128),np.complex128))
        _sss_scores(bank,np.ones((4,62),np.complex128))
        from .sequences import SYNC_BINS, pss_sequence
        _integrated_sss(np.ones((2,4,128),np.complex128),SYNC_BINS,pss_sequence(0),bank)
        templates=np.frombuffer(bytes(31*128*16),np.complex128).reshape(31,128)
        _frequency_dots(np.ones((4,128),np.complex128),templates)


@lru_cache(maxsize=4)
def _initialize_cached(backend, threads):
    if backend == 'reference' or (backend == 'auto' and numba is None):
        return 'reference'
    from .bch import _trellis
    with _warm_lock:
        run(np.ones((40, 3)), *_trellis(), threads)
        if threads == 1:
            from .bch import _pbch_signs, rate_match_indices
            _recover_batch(np.zeros((1,3,480)), _pbch_signs(0), rate_match_indices(), *_trellis())
        if threads > 1 and numba.threading_layer() == 'workqueue':
            raise ValueError('Concurrent parallel BCH requires the thread-safe omp or tbb layer')
    return 'numba'


def diagnostics(backend='auto', threads=1):
    selected = initialize(backend, threads)
    return dict(backend=selected, numba_version=numba.__version__ if numba else None,
                active_kernel_threads=threads if selected == 'numba' else 1,
                numba_pool_capacity=numba.config.NUMBA_NUM_THREADS if numba else 0,
                parallel=selected == 'numba' and threads > 1,
                dynamic_queue_scaling=False,
                threading_layer=numba.threading_layer() if selected == 'numba' and threads > 1 else None)


def subspace_workspace(count,branches):
    """Bounded per-receiver scratch, allocated before the source is released."""
    return (np.empty(count),np.empty((6,count)),np.empty(6),
            np.zeros((branches,count)),np.empty(branches*count,np.int64),
            np.empty(branches*count,np.int64),np.zeros(1,np.int64),
            np.empty(branches*count),np.empty(branches*count),np.empty(branches*count),
            np.empty((3,count),np.int64),np.empty(3,np.int64),np.zeros(1))
