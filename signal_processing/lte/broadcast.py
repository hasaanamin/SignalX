"""Additive bounded LTE synchronization + PBCH/MIB streaming receiver."""
import hashlib
import time

import numpy as np

from .mib import LTEBroadcastBatch, LTEPBCHObservation
from .pbch import PBCHDecoder, extract_slot
from .receiver import LTESynchronizer
from ._history import IQHistory


class LTEBroadcastReceiver(LTESynchronizer):
    """Same positioned-input/preflight contracts as LTESynchronizer.

    process_block/process_envelope/flush return LTEBroadcastBatch. MIBs are
    per received frame, not a global result history. CRC failures produce no
    MIB; counters remain inspectable. IQ retention <=24000+search chunk and
    at most max_cells decoders, each with four 3x480 soft-bit arrays. Raw gaps
    finalize eligible old-prefix work, then clear all cross-gap combining.
    """

    def __init__(self, stream_info, config=None, *, bch_backend='auto', numba_threads=1, search_backend='bounded', fft_workers=1, measure_timing=False, sync_backend='auto', decode_cadence='every_frame', blind_prefilter=True):
        if decode_cadence not in ('every_frame','bch_block'):
            raise ValueError('decode_cadence must be every_frame or bch_block')
        self._decode_cadence = decode_cadence
        if type(measure_timing) is not bool:
            raise ValueError('measure_timing must be bool')
        self._measure_timing = measure_timing
        self._stage_timing = {name:dict(calls=0,seconds=0.,maximum_seconds=0.)
                              for name in ('synchronization','pbch_decode')}
        # Compilation/readiness completes before callers start acquiring IQ.
        # Runtime execution controls do not alter thresholds/profile mathematics.
        from ._native import diagnostics
        self._execution = diagnostics(bch_backend, numba_threads)
        self._bch_backend = self._execution['backend']
        self._numba_threads = numba_threads
        super().__init__(stream_info, config, search_backend=search_backend, fft_workers=fft_workers,sync_backend=sync_backend,blind_prefilter=blind_prefilter)
        self._execution.update(super().execution)
        self._execution['sync_backend'] = self._sync_backend
        self._execution['decode_cadence'] = decode_cadence
        self._execution['search_backend'] = search_backend
        self._execution['fft_workers'] = fft_workers
        self._execution['bch_hypotheses'] = ('native_batch' if self._bch_backend == 'numba'
                                            and numba_threads == 1 else 'per_word')

    @property
    def execution(self):
        """Detached execution diagnostics; mutation cannot change the receiver."""
        return dict(self._execution)

    @property
    def stage_timing(self):
        """Bounded lifetime aggregate timers (diagnostic; enabled explicitly).

        Synchronization and slot extraction/PBCH decoding are disjoint stages.
        Reset/gaps do not erase totals. No IQ or per-frame timing history kept.
        Read between input calls, like snapshot().
        """
        return {name:dict(value) for name,value in self._stage_timing.items()}

    def _record_stage(self, name, started):
        elapsed = time.perf_counter()-started
        entry = self._stage_timing[name]
        entry['calls'] += 1
        entry['seconds'] += elapsed
        entry['maximum_seconds'] = max(entry['maximum_seconds'],elapsed)

    def _reset_stream(self):
        super()._reset_stream()
        if not hasattr(self, '_broadcast_history'):
            history = self.config.integration_half_frames*9600 if self.config.integration_half_frames > 1 else 0
            self._broadcast_history = IQHistory(24000+history+self.config.search_chunk_samples)
        self._broadcast_history.reset()
        self._broadcast_buffer = self._broadcast_history.view()
        self._broadcast_base = self._broadcast_end = 0
        self._broadcast_targets = {}
        self._broadcast_decoders = {}
        # At most max_cells entries: only a CRC-validated quarter establishes
        # the current 40-ms codeword boundary. Gaps/reset erase that knowledge.
        self._decoded_block_end = {}
        self._last_broadcast_frame = {}
        self._repetitions_avoided = 0
        # A gap resets raw state but must preserve completed output from this
        # call's old prefix. The base invokes _begin_output only on admission.
        if not hasattr(self, "_broadcast_completed"):
            self._broadcast_completed = []
        self._broadcast_attempts = self._broadcast_successes = 0

    @property
    def broadcast_profile_id(self):
        cached = getattr(self, '_broadcast_identity', None)
        if cached is not None:
            return cached
        value = self.config.profile_id+"|pbch_fdd_normal_crs_sfbc_exact_tb_v1|retention24000|combine4|frame_association_v1"
        if self.config.integration_half_frames > 1:
            value += "|bounded_retrospective_frames|received_cp20_v1"
        if self._decode_cadence == 'bch_block':
            value += '|decode_once_per_crc_validated_bch_block_v1'
        self._broadcast_identity = "lte_broadcast_"+hashlib.sha256(value.encode()).hexdigest()[:20]
        return self._broadcast_identity

    def _search(self, horizon, emitted, final=False):
        begin = len(emitted)
        started = time.perf_counter() if self._measure_timing else None
        super()._search(horizon, emitted, final)
        if started is not None:
            self._record_stage('synchronization',started)
        self._broadcast_history.append(self._buffer[self._broadcast_end-self._base:horizon-self._base])
        self._broadcast_buffer = self._broadcast_history.view()
        self._broadcast_end = horizon
        for observation in emitted[begin:]:
            if observation.state != "SYNCHRONIZED" or observation.processing_frame_start < 0:
                continue
            frame = observation.processing_frame_start
            history = self.config.integration_half_frames*9600 if self.config.integration_half_frames > 1 else 0
            # Acquisition can complete after earlier PBCH slots arrived. Decode
            # retained real IQ retrospectively, never manufacture sync records.
            for previous in range(frame, max(-1, frame-history-1), -19200):
                if previous+960 < self._broadcast_base:
                    continue
                key = (observation.pci, previous)
                decoder = self._broadcast_decoders.get(observation.pci)
                if decoder is not None and self._frame_identity(decoder, previous) is None:
                    continue
                self._broadcast_targets[key] = observation
        for key, observation in sorted(tuple(self._broadcast_targets.items()), key=lambda item: item[0][1]):
            pci, frame = key
            if frame+960 < self._broadcast_base:
                del self._broadcast_targets[key]
                continue
            if frame+1920 > horizon:
                continue
            del self._broadcast_targets[key]
            if pci not in self._broadcast_decoders:
                if len(self._broadcast_decoders) == self.config.max_cells:
                    oldest = next(iter(self._broadcast_decoders))
                    del self._broadcast_decoders[oldest]
                    self._decoded_block_end.pop(oldest,None)
                    self._last_broadcast_frame.pop(oldest,None)
                self._broadcast_decoders[pci] = PBCHDecoder(pci, backend=self._bch_backend,
                                                          threads=self._numba_threads)
            decoder = self._broadcast_decoders[pci]
            frame_identity = self._frame_identity(decoder, frame)
            if frame_identity is None:
                continue  # a refined half-frame observation of a decoded frame
            if self._decode_cadence == 'bch_block':
                self._last_broadcast_frame[pci] = frame_identity
            if (self._decode_cadence == 'bch_block'
                    and frame_identity < self._decoded_block_end.get(pci,-1)):
                self._repetitions_avoided += 1
                continue  # already decoded THIS codeword, not an unprocessed new one
            started = time.perf_counter() if self._measure_timing else None
            grid, frequency = extract_slot(self._broadcast_buffer, frame, self._broadcast_base,
                                            observation.frequency_offset_hz,
                                            average_cfo=self.config.integration_half_frames > 1)
            result = decoder.decode(grid, frame_identity)
            if started is not None:
                self._record_stage('pbch_decode',started)
            self._broadcast_attempts += 1
            if result is not None:
                mib, ports, quarter, count = result
                if self._decode_cadence == 'bch_block':
                    self._decoded_block_end[pci] = frame_identity+(4-quarter)*19200
                self._broadcast_successes += 1
                self._broadcast_completed.append(LTEPBCHObservation(mib, pci, ports, quarter, count,
                    self._epoch, observation.stream_id, self.broadcast_profile_id, frame,
                    self._origin+self._channel.input_position(frame), frequency, self._prepared_end,
                    observation.source_metadata, observation.source_configuration_id))
        history = self.config.integration_half_frames*9600 if self.config.integration_half_frames > 1 else 0
        keep = max(0, horizon-24000-history)
        self._broadcast_history.discard_before(keep)
        self._broadcast_buffer = self._broadcast_history.view()
        self._broadcast_base = keep

    def _frame_identity(self, decoder, measured):
        last = decoder.latest_frame_start
        if self._decode_cadence == 'bch_block':
            considered = self._last_broadcast_frame.get(decoder.pci)
            if considered is not None:
                last = max(last,considered) if last is not None else considered
        if last is None:
            return measured
        # Timing refinements are not new radio frames. Keep extraction at the
        # measured position but associate repeated observations and combining
        # with the existing 10-ms lattice, within the declared sync tolerance.
        periods = (measured-last+9600)//19200
        if periods <= 0:
            return None
        predicted = last+periods*19200
        if abs(predicted-measured) <= self.config.timing_tolerance_samples:
            return predicted
        return measured  # discontinuous phase: decoder clears combining history

    def _begin_output(self):
        self._broadcast_completed = []

    def _finish_output(self, sync):
        mibs = tuple(self._broadcast_completed)
        self._broadcast_completed.clear()
        return LTEBroadcastBatch(sync, mibs)

    def _attach_output_failure(self, exc):
        exc.completed_mibs = tuple(self._broadcast_completed)
        self._broadcast_completed.clear()

    def _extra_snapshot(self):
        return dict(broadcast_profile_id=self.broadcast_profile_id,
                    pbch_attempted_frames=self._broadcast_attempts, pbch_decoded_frames=self._broadcast_successes,
                    decode_cadence=self._decode_cadence, decoded_repetitions_avoided=self._repetitions_avoided,
                    retained_broadcast_samples=len(self._broadcast_buffer),
                    pending_broadcast_frames=len(self._broadcast_targets),
                    retained_bch_frames=sum(d.retained_frames for d in self._broadcast_decoders.values()))
