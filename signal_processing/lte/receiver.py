"""Bounded continuous PSS/SSS acquisition; no truth or energy gate is used."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict, deque
from fractions import Fraction
import threading

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import find_peaks

from signal_processing.core import IQBlock, StreamInfo, StreamConsistencyError
from signal_processing.core.stream_compatibility import require_stream
from signal_processing.core.timebase import sample_offset_ns
from signal_processing.sources.streaming import SourceBusy, SourceEnvelope
from .channel import LTEChannel
from ._history import IQHistory
from .config import LTESyncConfig
from .contracts import LTESyncObservation
from .sequences import SYNC_BINS, pss_sequence, sss_hypotheses, sync_symbol


@dataclass
class _Cell:
    last_pss: int
    frame: int
    frequency: float
    count: int
    score: float


class LTESynchronizer:
    """Single-reader FDD normal-CP synchronization, initially strongest cell.

    Receive 2 MS/s or native 1.92 MS/s positioned IQ. Fixed internal search
    chunks make decisions independent of transport partitioning. Returned
    observations are immutable; repeated confirmed observations are tracking
    updates, not distinct RF bursts. LTEBroadcastReceiver adds PBCH/MIB decoding.
    """

    def __init__(self, stream_info: StreamInfo, config: LTESyncConfig | None = None, *, search_backend='bounded', fft_workers=1, sync_backend='auto', blind_prefilter=True):
        if not isinstance(stream_info, StreamInfo):
            raise TypeError("Expected StreamInfo")
        if search_backend not in ('bounded', 'reference'):
            raise ValueError('Search backend must be bounded or reference')
        if sync_backend not in ('auto','numpy','numba'):
            raise ValueError('Sync backend must be auto, numpy or numba')
        if type(fft_workers) is not int or not 1 <= fft_workers <= 4:
            raise ValueError('FFT workers must be an integer from 1 to 4')
        if type(blind_prefilter) is not bool:
            raise ValueError('blind_prefilter must be bool')
        self._fft_workers = fft_workers
        self._search_backend = search_backend
        self._config = config or LTESyncConfig(input_sample_rate_hz=stream_info.sample_rate_hz)
        if not isinstance(self.config, LTESyncConfig) or stream_info.sample_rate_hz != self.config.input_sample_rate_hz:
            raise StreamConsistencyError("Stream/configuration sampling rate mismatch")
        self._search_query_width = self.config.search_chunk_samples+2*self.config.timing_tolerance_samples+1
        self._initial_stream = self._stream = stream_info
        self._channel = LTEChannel(stream_info.sample_rate_hz)
        self._pss = np.array([pss_sequence(n) for n in range(3)])
        self._symbols = np.array([sync_symbol(s) for s in self._pss])
        max_f, step = self.config.max_frequency_offset_hz, self.config.frequency_step_hz
        steps = int(max_f/step)
        frequencies = np.unique(np.r_[np.arange(-steps, steps+1)*step, -max_f, max_f])
        self._branches = [(n, float(f)) for f in frequencies for n in range(3)]
        t = np.arange(128)/1_920_000.
        templates = np.array([self._symbols[n]*np.exp(2j*np.pi*f*t) for n, f in self._branches])
        self._kernels = templates[:, ::-1].conj()
        self._templates = templates
        self._templates_conjugate = templates.conj()
        self._symbol_time = t
        self._smooth_kernel = np.ones(11)
        self._smooth_denominator = np.convolve(np.ones(31), self._smooth_kernel, 'same')
        self._sss_indices = (np.arange(336), np.arange(336) ^ 1)
        self._frequency_cache = OrderedDict()
        self._sss = [sss_hypotheses(n) for n in range(3)]
        self._sync_backend = 'numpy'
        if sync_backend != 'numpy':
            from ._native import initialize_sync
            self._sync_backend = initialize_sync(sync_backend)
        self._search_kernels = self._kernels
        self._power_templates_conjugate = self._templates_conjugate
        self._subspace_cutoffs = None
        if (blind_prefilter and self._sync_backend == 'numba' and search_backend == 'bounded'
                and self.config.tracking and len(frequencies) == 9):
            # Seven orthonormal directions cover each nine-CFO template family.
            # Residual bounds retain EVERY potentially qualifying full filter.
            bases,cutoffs = [],np.empty((len(self._branches),3))
            for nid in range(3):
                indices = np.arange(nid,len(self._branches),3)
                templates = self._templates[indices]/np.sqrt(62)
                basis = np.linalg.svd(templates,full_matrices=False)[2][:7]
                coefficients = templates@basis.conj().T
                residuals = np.linalg.norm(templates-coefficients@basis,axis=1)+1e-12
                norms = np.linalg.norm(coefficients[:,:5],axis=1)+1e-12
                tail_norms = np.linalg.norm(coefficients[:,5:],axis=1)+1e-12
                remaining = np.maximum(0,np.sqrt(self.config.pss_threshold)-residuals-1e-8)
                cutoffs[indices,0] = norms
                cutoffs[indices,1] = tail_norms
                cutoffs[indices,2] = remaining
                bases.extend(basis*np.sqrt(62))
            basis = np.array(bases)
            self._power_templates_conjugate = basis.conj()
            self._search_kernels = basis[:,::-1].conj()
            self._subspace_cutoffs = cutoffs
        self._subspace_workspace = None
        if self._subspace_cutoffs is not None:
            from ._native import subspace_workspace
            self._subspace_workspace = subspace_workspace(self._search_query_width,len(self._branches))
        self._refined_blind_windows = self._possible_blind_windows = 0
        self._fft_cache = {}
        self._stream_fft = (fft(self._search_kernels,n=512,axis=-1,workers=1)
                            if self._sync_backend == 'numba' and search_backend == 'bounded' else None)
        if self._stream_fft is not None:
            ifft(self._stream_fft,axis=-1,workers=1)
        self._fft_bank = None
        if self._stream_fft is not None:
            from ._fft_bank import optional_bank
            self._fft_bank = optional_bank(self._search_kernels)
        # Dense narrowband inputs can defeat a conservative subspace bound.
        # Prepare the exact full bank before acquisition too; switching uses
        # only retained received IQ and never changes a detection threshold.
        self._screen_configuration = (self._search_kernels,self._power_templates_conjugate,
                                      self._subspace_cutoffs,self._stream_fft,self._fft_bank)
        self._full_stream_fft = self._full_fft_bank = None
        if self._subspace_cutoffs is not None:
            self._full_stream_fft = fft(self._kernels,n=512,axis=-1,workers=1)
            ifft(self._full_stream_fft,axis=-1,workers=1)
            self._full_fft_bank = optional_bank(self._kernels)
        self._dense_search_fallbacks = 0
        self._tone_rejected_chunks = 0
        if search_backend == 'bounded':
            # Only immutable configuration kernels; no samples/decisions/clock
            # advance. Make FFT pool/plans ready before acquisition starts.
            initial = 137+self._channel.settling_samples+(self.config.integration_half_frames-1)*9600
            chunk = self.config.search_chunk_samples
            horizon = ((initial+127)//chunk+1)*chunk
            for count in (horizon-127-initial, chunk):
                nfft = next_fast_len(count+254)
                self._fft_cache[nfft] = fft(self._kernels, n=nfft, axis=-1, workers=fft_workers)
                # Ready the inverse transform too, using configuration only.
                ifft(self._fft_cache[nfft],axis=-1,workers=fft_workers)
        self._gate = threading.Lock()
        self._epoch = self._gaps = self._lock_losses = self._observation_count = 0
        self._blind_searches = self._tracking_searches = 0
        self._reset_stream()

    @property
    def config(self):
        return self._config

    @property
    def execution(self):
        """Detached runtime selection; configuration/statistic identity is separate."""
        return dict(sync_backend=self._sync_backend,search_backend=self._search_backend,
                    fft_workers=self._fft_workers,
                    blind_fft_backend='fftw' if self._fft_bank is not None else 'scipy',
                    blind_fft_dispatch=getattr(self._fft_bank,'dispatch','python' if self._fft_bank is not None else None),
                    blind_prefilter='bounded_subspace_v1' if self._subspace_cutoffs is not None else None)

    def _reset_stream(self):
        (self._search_kernels,self._power_templates_conjugate,self._subspace_cutoffs,
         self._stream_fft,self._fft_bank) = self._screen_configuration
        self._channel.reset()
        if not hasattr(self, '_history'):
            capacity = 300+(self.config.integration_half_frames-1)*9600+self.config.search_chunk_samples+16384
            self._history = IQHistory(capacity)
        self._history.reset()
        self._buffer = self._history.view()
        self._base = self._processed = self._available = 0
        self._next_search = 137+self._channel.settling_samples+(self.config.integration_half_frames-1)*9600
        self._origin = self._previous = self._anchor = None
        self._received_end = self._prepared_end = None
        self._source_metadata = self._source_configuration = None
        self._api = None
        self._cells = {}
        self._locked = None
        self._tracking_missed = False
        self._blind_preparation_horizon = self._tone_rejected_horizon = None
        self._state = "SEARCHING"
        self._finished = False
        self._score_chunks = deque()
        self._score_end = 137+self._channel.settling_samples
        self._power_ring_capacity = ((self.config.integration_half_frames-1)*9600
                                    +2*self.config.search_chunk_samples+512)
        if not hasattr(self,'_power_ring'):
            # Only the largest query needs mirroring, rather than a second
            # whole history. Keeping this working set small matters on laptops.
            ring_width = (self._power_ring_capacity+self._search_query_width
                          if self._stream_fft is not None else 0)
            self._power_storage = np.empty((len(self._branches),ring_width))
            self._energy_ring = np.empty(ring_width)
            self._group_power_ring = np.empty((6 if self._screen_configuration[2] is not None else 0,ring_width))
        self._power_ring = self._power_storage[:len(self._search_kernels)]
        self._power_ring_start = self._score_end

    def reset(self):
        if not self._gate.acquire(False):
            raise SourceBusy("Cannot reset an active synchronizer")
        try:
            self._epoch += 1
            self._stream = self._initial_stream
            self._reset_stream()
        finally:
            self._gate.release()

    def snapshot(self):
        if not self._gate.acquire(False):
            raise SourceBusy("Snapshot requires the receiver between input calls")
        try:
            return {"state": self._state, "pci": self._locked, "epoch": self._epoch,
                    "profile_id": self.config.profile_id,
                    "input_rate_hz": self._stream.sample_rate_hz, "processing_rate_hz": 1_920_000,
                    "received_input_end": self._received_end, "prepared_input_end": self._prepared_end,
                    "processed_samples": self._processed, "retained_processing_samples": len(self._buffer),
                    "retained_resampler_samples": self._channel.retained_samples,
                    "candidate_cells": len(self._cells), "discontinuities": self._gaps,
                    "retained_search_power_samples": sum(b-a for a,b,_,_ in self._score_chunks),
                    "retained_search_power_bytes": sum(p.nbytes+e.nbytes for _,_,p,e in self._score_chunks),
                    "search_power_allocation_bytes": self._power_storage.nbytes+self._energy_ring.nbytes,
                    "search_group_power_allocation_bytes": self._group_power_ring.nbytes,
                    "dense_search_fallbacks": self._dense_search_fallbacks,
                    "tone_rejected_search_chunks": self._tone_rejected_chunks,
                    "search_workspace_bytes": sum(a.nbytes for a in self._subspace_workspace) if self._subspace_workspace is not None else 0,
                    "refined_blind_windows": self._refined_blind_windows,
                    "possible_blind_windows": self._possible_blind_windows,
                    "lock_losses": self._lock_losses, "observations": self._observation_count,
                    "blind_searches": self._blind_searches, "tracking_searches": self._tracking_searches,
                    "frequency_hypotheses": len(self._branches)//3,
                    "finished": self._finished, **self._extra_snapshot()}
        finally:
            self._gate.release()

    def _extra_snapshot(self):
        return {}

    def _begin_output(self):
        """Extension hooks execute under the same single-reader lifecycle gate."""

    def _finish_output(self, emitted):
        return tuple(emitted)

    def _attach_output_failure(self, exc):
        pass

    def process_envelope(self, envelope: SourceEnvelope):
        if not isinstance(envelope, SourceEnvelope):
            raise TypeError("Expected SourceEnvelope")
        return self._process(envelope.block, envelope)

    def process_block(self, block: IQBlock):
        return self._process(block, None)

    def _preflight(self, block, envelope):
        if not isinstance(block, IQBlock):
            raise TypeError("Expected IQBlock")
        if self._state == "FAULTED" or self._finished:
            raise RuntimeError("Reset required after fault/EOF")
        if block.num_samples > self.config.max_input_samples:
            raise ValueError("Block exceeds bounded LTE input capacity; partition it before admission")
        candidate = require_stream(block.stream, self._stream,
                                   bind_center=self._previous is None)
        api = "envelope" if envelope is not None else "block"
        if self._api is not None and self._api != api:
            raise ValueError("Do not mix envelope/block admission within one receiver run")
        if envelope is not None and self._source_metadata is not None and envelope.metadata != self._source_metadata:
            raise StreamConsistencyError("Source/gain/configuration epoch changed; reset/reconfigure explicitly")
        gap = block.dropped_samples_before > 0 or block.overflow
        if self._previous is not None:
            p = self._previous
            if block.sequence_number <= p.sequence_number or block.start_sample != p.end_sample+block.dropped_samples_before:
                raise StreamConsistencyError("Discontinuous IQ requires explicit truthful loss markers")
        if not gap and self._anchor is not None and block.timestamp_ns is not None:
            index, stamp = self._anchor
            expected = stamp+sample_offset_ns(block.start_sample-index, candidate.sample_rate_hz)
            if abs(expected-block.timestamp_ns) > 1:
                raise StreamConsistencyError("Receiver timestamp contradicts sample timing (1 ns tolerance)")
        return candidate, api, gap

    def _process(self, block, envelope):
        if not self._gate.acquire(False):
            raise SourceBusy("LTE synchronizer supports one ordered caller")
        try:
            candidate, api, gap = self._preflight(block, envelope)
            self._begin_output()
            # All predictable rejection above is atomic, including pending IQ.
            self._received_end = block.end_sample
            emitted = []
            consumed = 0
            try:
                if gap:
                    # Same chronological finalization as EOF; do not throw away
                    # a complete real PSS/SSS pair in the final short search chunk.
                    self._process_ready(emitted, final=True)
                    if self._locked is not None:
                        self._lock_losses += 1
                    self._epoch += 1
                    self._gaps += 1
                    self._reset_stream()
                self._stream = candidate
                self._api = api
                if envelope is not None:
                    self._source_metadata = envelope.metadata
                    self._source_configuration = envelope.configuration_id
                if self._origin is None:
                    self._origin = block.start_sample
                if block.timestamp_ns is not None and self._anchor is None:
                    self._anchor = (block.start_sample, block.timestamp_ns)
                self._received_end = block.end_sample
                for start in range(0, block.num_samples, 16384):
                    end = min(block.num_samples, start+16384)
                    _, prepared = self._channel._process_received(block.iq[start:end])
                    self._prepared_end = block.start_sample+end
                    consumed = end
                    self._history.append(prepared)
                    self._buffer = self._history.view()
                    self._available += len(prepared)
                    self._process_ready(emitted)
                self._previous = block
            except Exception as exc:
                self._state = "FAULTED"
                if consumed:
                    self._previous = block.slice_received(0, consumed)
                exc.completed_observations = tuple(emitted)
                exc.received_input_end = self._received_end
                exc.prepared_input_end = self._prepared_end
                self._attach_output_failure(exc)
                raise
            return self._finish_output(emitted)
        finally:
            self._gate.release()

    def _process_ready(self, emitted, final=False):
        while self._processed+self.config.search_chunk_samples <= self._available or (final and self._processed < self._available):
            horizon = min(self._processed+self.config.search_chunk_samples, self._available)
            if self._stream_fft is not None and (self._locked is None or self._tracking_missed):
                self._prepare_blind_search(horizon)
            self._search(horizon, emitted, final=final)
            self._processed = horizon
            keep = max(0, horizon-300-(self.config.integration_half_frames-1)*9600)
            self._history.discard_before(keep)
            self._buffer = self._history.view()
            self._base = keep
            while self._score_chunks and self._score_chunks[0][1] <= keep:
                self._score_chunks.popleft()

    def flush(self):
        if not self._gate.acquire(False):
            raise SourceBusy("Cannot flush active receiver")
        try:
            if self._state == "FAULTED":
                raise RuntimeError("Reset required after receiver fault")
            self._begin_output()
            if self._finished:
                return self._finish_output(())
            emitted = []
            try:
                self._process_ready(emitted, final=True)
            except Exception as exc:
                self._state = "FAULTED"
                exc.completed_observations = tuple(emitted)
                self._attach_output_failure(exc)
                raise
            self._finished = True
            return self._finish_output(emitted)
        finally:
            self._gate.release()

    def _search(self, horizon, emitted, final=False):
        stop = horizon-127
        start = self._next_search
        self._expire(horizon)
        if stop <= start:
            return
        x = self._buffer[:horizon-self._base]
        can_track = (self.config.tracking and self._locked is not None
                     and not self._tracking_missed and not final)
        tracking = self._tracking_candidates(x,start,stop) if can_track else None
        if tracking is None:
            if can_track:
                self._tracking_missed = True
            # Rebuild only missing RECEIVED correlations once after a tracking
            # miss, then keep continuous blind search until fresh confirmation.
            # Avoid the four whole-history FFT banks formerly charged per miss.
            if (self._stream_fft is not None and self._score_end < stop
                    and self._blind_preparation_horizon != horizon):
                self._prepare_blind_search(horizon)
            candidates = self._blind_candidates(x, start, stop)
            next_search = stop
        else:
            candidates, next_search = tracking
        freshly_confirmed = set()
        for p, nid, coarse in candidates:
            decoded = self._decode(x, p-self._base, nid, coarse)
            if decoded is None:
                continue
            pci, half, frequency, pmetric, smetric, ratio = decoded
            previous = self._cells.get(pci)
            if previous is not None and p-previous.last_pss < 64:
                continue
            frame = p-832-half*9600
            count = 1
            if previous is not None:
                delta = p-previous.last_pss
                distance = abs((frame-previous.frame+9600) % 19200-9600)
                periods = round(delta/9600)
                if (periods >= 1 and abs(delta-periods*9600) <= self.config.timing_tolerance_samples
                        and distance <= self.config.timing_tolerance_samples
                        and abs(frequency-previous.frequency) <= self.config.frequency_step_hz):
                    count = min(self.config.confirmations, previous.count+1)
            if pci not in self._cells and len(self._cells) == self.config.max_cells:
                oldest = min(self._cells, key=lambda key: self._cells[key].last_pss)
                del self._cells[oldest]
            self._cells[pci] = _Cell(p, frame, frequency, count, smetric)
            if count >= self.config.confirmations:
                freshly_confirmed.add(pci)
            self._observation_count += 1
            raw_pss = self._origin+self._channel.input_position(p)
            raw_frame = self._origin+self._channel.input_position(frame)
            emitted.append(LTESyncObservation(pci,
                "SYNCHRONIZED" if count >= self.config.confirmations else "CANDIDATE",
                self._epoch, self.config.profile_id, self._stream.stream_id,
                p, frame, Fraction(raw_pss), Fraction(raw_frame),
                self._origin <= raw_frame < self._prepared_end,
                float(frequency), float(pmetric), float(smetric), float(ratio), count,
                self._prepared_end, self._source_metadata, self._source_configuration))
        self._next_search = next_search
        self._update_state()
        if self._locked in freshly_confirmed:
            self._tracking_missed = False
            self._restore_screen_bank()

    def _restore_screen_bank(self):
        if self._subspace_cutoffs is None and self._screen_configuration[2] is not None:
            # Locked tracking or a proved subthreshold region needs no full
            # bank. Invalidate its history before restoring the sparse bank.
            (self._search_kernels,self._power_templates_conjugate,self._subspace_cutoffs,
             self._stream_fft,self._fft_bank) = self._screen_configuration
            self._power_ring = self._power_storage[:len(self._search_kernels)]
            self._score_chunks.clear()
            self._score_end = self._power_ring_start = self._base

    def _tracking_candidates(self, x, start, stop):
        if self._search_backend == 'reference' or self._sync_backend == 'numpy':
            return self._tracking_candidates_reference(x, start, stop)
        return self._tracking_candidates_native(x, start, stop)

    def _tracking_candidates_native(self, x, start, stop):
        """Same branches/received windows as reference, with batched reductions."""
        self._tracking_searches += 1
        cell = self._cells[self._locked]
        tol = self.config.timing_tolerance_samples
        period = max(1, (start-tol-cell.last_pss+9599)//9600)
        expected = cell.last_pss+period*9600
        candidates, next_search = [], stop
        while expected-tol < stop:
            if expected+tol >= stop:
                next_search = max(start, expected-tol)
                break
            positions = np.arange(max(start, expected-tol), min(stop, expected+tol+1))
            from ._native import _tracking_scores
            scores = _tracking_scores(x,positions-self._base,self.config.integration_half_frames,self._templates_conjugate)
            i, branch = np.unravel_index(int(scores.argmax()), scores.shape)
            if scores[i,branch] < self.config.pss_threshold:
                return None
            nid, coarse = self._branches[branch]
            candidates.append((int(positions[i]), nid, coarse))
            expected += 9600
        return candidates, next_search

    def _tracking_candidates_reference(self, x, start, stop):
        """Search only predicted times after independently repeated confirmation.

        Evaluate all original PSS/CFO branches with the same normalized statistic.
        Loss or a missing expected peak falls back to blind acquisition. The
        tracking=False profile retains full search as a numerical reference.
        """
        self._tracking_searches += 1
        cell = self._cells[self._locked]
        tol = self.config.timing_tolerance_samples
        period = max(1, (start-tol-cell.last_pss+9599)//9600)
        expected = cell.last_pss+period*9600
        candidates = []
        next_search = stop
        while expected-tol < stop:
            if expected+tol >= stop:
                next_search = max(start, expected-tol)
                break  # Keep enough overlap to choose the complete peak next time.
            positions = np.arange(max(start, expected-tol), min(stop, expected+tol+1))
            windows = np.array([x[p-self._base:p-self._base+128] for p in positions])
            energy = np.einsum("ij,ij->i", windows.conj(), windows, optimize=False).real
            dots = np.einsum("ij,kj->ik", windows, self._templates_conjugate, optimize=False)
            scores = np.zeros_like(dots.real)
            positive = energy > 0
            scores[positive] = np.abs(dots[positive])**2/(62*energy[positive, None])
            if self.config.integration_half_frames > 1:
                powers = np.abs(dots)**2
                for repeat in range(1, self.config.integration_half_frames):
                    past = np.array([x[p-self._base-repeat*9600:p-self._base-repeat*9600+128] for p in positions])
                    energy += np.einsum("ij,ij->i", past.conj(), past, optimize=False).real
                    powers += np.abs(np.einsum("ij,kj->ik", past, self._templates_conjugate, optimize=False))**2
                scores[:] = 0
                positive = energy > 0
                scores[positive] = powers[positive]/(62*energy[positive, None])
            i, branch = np.unravel_index(int(scores.argmax()), scores.shape)
            if scores[i, branch] < self.config.pss_threshold:
                return None
            nid, coarse = self._branches[branch]
            candidates.append((int(positions[i]), nid, coarse))
            expected += 9600
        return candidates, next_search

    def _blind_candidates(self, x, start, stop):
        self._blind_searches += 1
        screened_maximum = None
        if self._tone_rejected_horizon == stop+127:
            return []
        if self._search_backend == 'bounded':
            scores = (self._cached_blind_scores(start,stop,prune=True)
                      if self._stream_fft is not None and self.config.tracking else None)
            if scores is None:
                scores = self._bounded_scores(x, start, stop)
            elif self._subspace_cutoffs is not None:
                screened_maximum = self._subspace_workspace[-1][0]
        else:
            scores = self._reference_scores(x, start, stop)
        if screened_maximum is not None and screened_maximum < self.config.pss_threshold:
            return []
        if self.config.integration_half_frames > 1:
            if screened_maximum is None and scores.max() < self.config.pss_threshold:
                return []
            # Zadoff-Chu delay/frequency ambiguities require branch diversity.
            hypotheses = []
            for branch, values in enumerate(scores):
                indices = find_peaks(np.r_[-np.inf, values, -np.inf],
                    height=self.config.pss_threshold, distance=64)[0]-1
                hypotheses.extend((float(values[idx]), int(idx), branch) for idx in indices)
            selected = []
            for _, idx, branch in sorted(hypotheses, reverse=True):
                nid, coarse = self._branches[branch]
                if any(nid == n and abs(start+idx-p) <= 2 for p, n, f in selected):
                    continue
                selected.append((start+idx, nid, coarse))
                if len(selected) == self.config.max_candidates:
                    break
            return sorted(selected)
        best = scores.max(axis=0)
        branches = scores.argmax(axis=0)
        peaks = find_peaks(np.r_[-np.inf, best, -np.inf], height=self.config.pss_threshold, distance=64)[0]-1
        if len(peaks):
            peaks = peaks[np.argsort(best[peaks])[-self.config.max_candidates:]]
        return [(start+int(idx), *self._branches[int(branches[idx])]) for idx in sorted(peaks)]

    def _bounded_scores(self, x, start, stop):
        """Same linear correlations, only on the regions actually observed.

        Each repetition needs [start-r*9600, stop-r*9600+127). Transforming
        the entire 30k history for every 4k search wastes both work and memory.
        No correlation/result history or running-subtraction power is retained.
        """
        if self._stream_fft is not None:
            cached = self._cached_blind_scores(start,stop)
            if cached is not None:
                return cached
        count = stop-start
        energy = np.zeros(count)
        powers = np.zeros((len(self._branches), count))
        for repeat in range(self.config.integration_half_frames):
            begin = start-self._base-repeat*9600
            region = x[begin:stop-self._base-repeat*9600+127]
            nfft = next_fast_len(len(region)+127)
            if nfft not in self._fft_cache:
                if len(self._fft_cache) == 2:
                    self._fft_cache.pop(next(iter(self._fft_cache)))
                self._fft_cache[nfft] = fft(self._kernels, n=nfft, axis=-1, workers=self._fft_workers)
            transformed = fft(region, n=nfft, workers=1)
            corr = ifft(self._fft_cache[nfft]*transformed, axis=-1, workers=self._fft_workers)
            powers += np.abs(corr[:, 127:127+count])**2
            energy += np.convolve(region.real*region.real+region.imag*region.imag,
                                  np.ones(128), mode='valid')
        scores = np.zeros_like(powers)
        positive = energy > 0
        scores[:, positive] = powers[:, positive]/(62*energy[positive])
        return scores

    def _prepare_blind_search(self,horizon):
        from ._native import _subthreshold_tone
        begin = max(self._base,self._next_search-(self.config.integration_half_frames-1)*9600)
        region = self._buffer[begin-self._base:horizon-self._base]
        rejected = _subthreshold_tone(region,self._templates_conjugate,self.config.pss_threshold)
        self._blind_preparation_horizon = horizon
        self._tone_rejected_horizon = horizon if rejected else None
        if rejected:
            self._tone_rejected_chunks += 1
            self._restore_screen_bank()
        else:
            self._prepare_blind_powers(horizon)

    def _prepare_blind_powers(self, horizon):
        """Overlap-save each received position ONCE while searching.

        The four-half-frame statistic remains an incoherent sum of exactly the
        same correlations. Preparing its old contributions as IQ arrives avoids
        concentrating four FFT banks at first eligibility. Small FFT tiles keep
        the working set bounded, including across tracking/lock-loss transitions.
        """
        start = max(self._score_end,self._base)
        stop = horizon-127
        if stop <= start:
            return
        count = stop-start
        x = self._buffer
        capacity = self._power_ring_capacity
        region = x[start-self._base:stop-self._base+127]
        from ._native import _correlation_powers, _window_energy
        energy = _window_energy(region)
        if self._fft_bank is not None:
            self._fft_bank.write_powers(region,start,self._power_ring,capacity,
                                        self._search_query_width)
        else:
            for offset in range(0,count,385):
                n = min(385,count-offset)
                begin = start+offset-self._base
                tile = x[begin:begin+n+127]
                transformed = fft(tile,n=512,workers=1)
                correlations = ifft(self._stream_fft*transformed,axis=-1,workers=1,
                                    overwrite_x=True)
                position = (start+offset) % capacity
                first = min(n,capacity-position)
                _correlation_powers(correlations,self._power_ring,position,first,capacity,127,self._search_query_width)
                if first < n:
                    _correlation_powers(correlations,self._power_ring,0,n-first,capacity,127+first,self._search_query_width)
        if start != self._score_end:
            self._power_ring_start = start
        position = start % capacity
        first = min(count,capacity-position)
        for source,destination,length in ((0,position,first),(first,0,count-first)):
            if not length:
                continue
            self._energy_ring[destination:destination+length] = energy[source:source+length]
            mirrored = min(length,max(0,self._search_query_width-destination))
            self._energy_ring[destination+capacity:destination+capacity+mirrored] = energy[source:source+mirrored]
        # A lock-loss rebuild can span the retained half-frames; each borrowed
        # diagnostic view still fits the mirrored query suffix at a wrap.
        for offset in range(0,count,self.config.search_chunk_samples):
            length = min(self.config.search_chunk_samples,count-offset)
            position = (start+offset) % capacity
            self._score_chunks.append((start+offset,start+offset+length,
                self._power_ring[:,position:position+length],
                self._energy_ring[position:position+length]))
        from ._native import _repair_small_powers
        _repair_small_powers(x[start-self._base:stop-self._base+127],energy,
                             self._power_templates_conjugate,self._power_ring,start,capacity,
                             self._search_query_width)
        if self._subspace_cutoffs is not None:
            from ._native import _group_power_slice
            position = start % capacity
            first = min(count,capacity-position)
            for destination,length in ((position,first),(0,count-first)):
                if length:
                    _group_power_slice(self._power_ring,self._group_power_ring,destination,length,
                                       capacity,self._search_query_width)
        self._power_ring_start = max(self._power_ring_start,stop-capacity)
        self._score_end = stop

    def _cached_blind_scores(self,start,stop,*,prune=False):
        # Absolute coverage is invalidated across lock-time gaps and resets.
        # Borrowed chunk views are diagnostics, never a fallback for stale IQ.
        if (start-(self.config.integration_half_frames-1)*9600 >= self._power_ring_start
                and stop <= self._score_end and stop-start <= self._search_query_width):
            if self._subspace_cutoffs is not None:
                if not prune:
                    from ._native import _tracking_scores
                    positions = np.arange(start-self._base,stop-self._base)
                    return _tracking_scores(self._buffer,positions,self.config.integration_half_frames,
                                            self._templates_conjugate).T
                from ._native import _subspace_refine
                scores,count = _subspace_refine(self._buffer,self._base,self._group_power_ring,
                    self._energy_ring,start,stop,self.config.integration_half_frames,
                    self._power_ring_capacity,self._subspace_cutoffs,self._templates_conjugate,self._subspace_workspace,
                    limit=max(64,(stop-start)//32))
                if scores is None:
                    rebuild_end = max(stop,self._score_end)
                    self._search_kernels = self._kernels
                    self._power_templates_conjugate = self._templates_conjugate
                    self._subspace_cutoffs = None
                    self._stream_fft,self._fft_bank = self._full_stream_fft,self._full_fft_bank
                    self._power_ring = self._power_storage
                    self._score_chunks.clear()
                    self._score_end = self._power_ring_start = self._base
                    self._dense_search_fallbacks += 1
                    self._prepare_blind_powers(rebuild_end+127)
                    return self._cached_blind_scores(start,stop,prune=prune)
                self._refined_blind_windows += count
                self._possible_blind_windows += scores.size
                return scores
            from ._native import _ring_scores
            return _ring_scores(self._power_ring,self._energy_ring,start,stop,
                                self.config.integration_half_frames,self._power_ring_capacity)
        return None

    def _reference_scores(self, x, start, stop):
        """Original whole-history FFT reference for numerical verification."""
        nfft = next_fast_len(len(x)+127)
        if nfft not in self._fft_cache:
            if len(self._fft_cache) == 2:
                self._fft_cache.pop(next(iter(self._fft_cache)))
            self._fft_cache[nfft] = fft(self._kernels, n=nfft, axis=-1, workers=self._fft_workers)
        transformed = fft(x, n=nfft, workers=1)
        corr = ifft(self._fft_cache[nfft]*transformed, axis=-1, workers=self._fft_workers)
        positions = np.arange(start-self._base, stop-self._base)
        # Direct nonnegative reductions avoid cumulative-subtraction cancellation.
        all_energy = np.convolve(x.real*x.real+x.imag*x.imag, np.ones(128), mode="valid")
        energy = all_energy[positions].copy()
        scores = np.zeros((len(self._branches), len(positions)))
        positive = energy > 0
        scores[:, positive] = np.abs(corr[:, positions[positive]+127])**2/(62*energy[positive])
        if self.config.integration_half_frames > 1:
            powers = np.abs(corr[:, positions+127])**2
            for repeat in range(1, self.config.integration_half_frames):
                past = positions-repeat*9600
                energy += all_energy[past]
                powers += np.abs(corr[:, past+127])**2
            scores[:] = 0
            positive = energy > 0
            scores[:, positive] = powers[:, positive]/(62*energy[positive])
        return scores

    def _expire(self, horizon):
        for pci in list(self._cells):
            if horizon-self._cells[pci].last_pss > self.config.loss_timeout_half_frames*9600:
                del self._cells[pci]
        self._update_state()

    def _update_state(self):
        confirmed = [pci for pci, c in self._cells.items() if c.count >= self.config.confirmations]
        locked = max(confirmed, key=lambda pci: self._cells[pci].score) if confirmed else None
        if self._locked is not None and locked is None:
            self._lock_losses += 1
        self._locked = locked
        self._state = "SYNCHRONIZED" if locked is not None else ("CANDIDATE" if self._cells else "SEARCHING")

    def _decode(self, x, p, nid, coarse):
        if self.config.integration_half_frames > 1:
            return self._decode_integrated(x, p, nid, coarse)
        y = x[p:p+128]
        template = self._symbols[nid]
        limit = self.config.max_frequency_offset_hz
        lo = max(-limit, coarse-self.config.frequency_step_hz)
        hi = min(limit, coarse+self.config.frequency_step_hz)
        t = self._symbol_time
        grid, templates = self._frequency_templates(float(coarse), False)
        dots = templates @ (y*template.conj())
        frequency = grid[int(np.argmax(np.abs(dots)))]
        fine, templates = self._frequency_templates(float(frequency), True)
        dots = templates @ (y*template.conj())
        index = int(np.argmax(np.abs(dots)))
        frequency = fine[index]
        pmetric = float(abs(dots[index])**2/(62*np.vdot(y, y).real))
        if pmetric < self.config.pss_threshold:
            return None
        yp = fft(y*np.exp(-2j*np.pi*frequency*t), workers=1)[SYNC_BINS]
        ys = fft(x[p-137:p-9]*np.exp(-2j*np.pi*frequency*(t-137/1_920_000.)), workers=1)[SYNC_BINS]
        channel = yp*self._pss[nid].conj()
        # Smooth each side of DC separately; normalize edges, never floor power.
        for a, b in ((0, 31), (31, 62)):
            channel[a:b] = np.convolve(channel[a:b], np.ones(5), mode="same")/np.convolve(np.ones(31), np.ones(5), mode="same")
        power = np.abs(channel)**2
        regularizer = .05*float(np.mean(power))
        if regularizer == 0:
            return None
        equalized = ys*channel.conj()/(power+regularizer)
        norm = float(np.vdot(equalized, equalized).real)*62
        if norm == 0:
            return None
        dots = np.einsum("ij,j->i", self._sss[nid], equalized, optimize=False)
        scores = np.abs(dots)**2/norm
        order = np.argpartition(scores, -2)[-2:]
        index = int(order[np.argmax(scores[order])])
        second = float(np.min(scores[order]))
        score = float(scores[index])
        ratio = score/max(second, np.finfo(float).tiny)
        if score < self.config.sss_threshold or ratio < self.config.sss_peak_ratio:
            return None
        group, half = divmod(index, 2)
        return 3*group+nid, half, frequency, min(pmetric, 1.), min(score, 1.), ratio

    def _frequency_templates(self, center, fine):
        # Configuration-only kernels, never received IQ/correlations/results.
        # Bounded even under a long stream of arbitrary candidate frequencies.
        key = (center, fine)
        found = self._frequency_cache.get(key)
        if found is not None:
            self._frequency_cache.move_to_end(key)
            return found
        width, count = (500, 11) if fine else (self.config.frequency_step_hz, 31)
        limit = self.config.max_frequency_offset_hz
        grid = np.linspace(max(-limit, center-width), min(limit, center+width), count)
        templates = np.exp(-2j*np.pi*grid[:, None]*self._symbol_time)
        grid.flags.writeable = templates.flags.writeable = False
        if len(self._frequency_cache) == 32:
            self._frequency_cache.popitem(last=False)
        self._frequency_cache[key] = grid, templates
        return grid, templates

    def _decode_integrated(self, x, p, nid, coarse):
        repeats = self.config.integration_half_frames
        positions = p-9600*np.arange(repeats)
        if positions[-1] < 137:
            return None
        y = np.array([x[q:q+128] for q in positions])
        t = self._symbol_time
        limit = self.config.max_frequency_offset_hz
        grid, templates = self._frequency_templates(float(coarse), False)
        mixed = y*self._symbols[nid].conj()
        native = self._sync_backend == 'numba' and self._search_backend != 'reference'
        if native:
            from ._native import _frequency_dots
            dots = _frequency_dots(mixed,templates)
        else:
            dots = mixed @ templates.T
        frequency = float(grid[np.argmax(np.sum(np.abs(dots)**2, axis=0))])
        fine, templates = self._frequency_templates(frequency, True)
        dots = _frequency_dots(mixed,templates) if native else mixed @ templates.T
        index = int(np.argmax(np.sum(np.abs(dots)**2, axis=0)))
        frequency = float(fine[index])
        energy = float(np.sum(np.abs(y)**2))
        if energy == 0:
            return None
        pmetric = float(np.sum(np.abs(dots[:, index])**2)/(62*energy))
        if pmetric < self.config.pss_threshold:
            return None
        if native:
            # One bounded FFT dispatch and one native channel/SSS reduction.
            # Every repeated received pair and original hypothesis is retained.
            from ._native import _integrated_sss
            pairs = np.empty((2, repeats, 128), np.complex128)
            pairs[0] = y*np.exp(-2j*np.pi*frequency*t)
            rotation = np.exp(-2j*np.pi*frequency*(t-137/1_920_000.))
            for repeat, q in enumerate(positions):
                pairs[1, repeat] = x[q-137:q-9]*rotation
            spectra = fft(pairs, axis=-1, workers=1, overwrite_x=True)
            scores = _integrated_sss(spectra,SYNC_BINS,self._pss[nid],self._sss[nid])
        else:
            scores = self._integrated_sss_reference(x, y, positions, nid, frequency)
        order = np.argpartition(scores, -2)[-2:]
        index = int(order[np.argmax(scores[order])])
        score, second = float(scores[index]), float(np.min(scores[order]))
        ratio = score/max(second, np.finfo(float).tiny)
        if score < self.config.sss_threshold or ratio < self.config.sss_peak_ratio:
            return None
        group, half = divmod(index, 2)
        return 3*group+nid, half, frequency, min(pmetric, 1.), min(score, 1.), ratio

    def _integrated_sss_reference(self, x, y, positions, nid, frequency):
        t = self._symbol_time
        repeats = len(positions)
        accumulated = np.zeros(336, np.complex128)
        total_energy = 0.
        for repeat, q in enumerate(positions):
            yp = fft(y[repeat]*np.exp(-2j*np.pi*frequency*t), workers=1)[SYNC_BINS]
            ys = fft(x[q-137:q-9]*np.exp(-2j*np.pi*frequency*(t-137/1_920_000.)), workers=1)[SYNC_BINS]
            channel = yp*self._pss[nid].conj()
            for a, b in ((0, 31), (31, 62)):
                channel[a:b] = np.convolve(channel[a:b], self._smooth_kernel, "same")/self._smooth_denominator
            equalized = ys*channel.conj()
            dots = np.einsum("ij,j->i", self._sss[nid], equalized, optimize=False)
            # SSS alternates its two sequences every half-frame. Channel phase
            # is cancelled separately at each received PSS/SSS pair before
            # coherent accumulation; no 5-ms carrier-phase assumption is made.
            accumulated += dots[self._sss_indices[repeat % 2]]
            total_energy += float(np.vdot(equalized, equalized).real)
        return (np.abs(accumulated)**2/(repeats*62*total_energy)
                if total_energy > 0 else np.zeros(336))
