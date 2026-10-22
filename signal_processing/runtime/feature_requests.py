"""Explicit, bounded feature requests, independent of IQ admission.

No capture/history is retained here. Callers own the immutable capture they
select; only accepted jobs enter the existing bounded asynchronous worker.
"""
from dataclasses import dataclass, replace
import math
import threading
import time

from signal_processing.capture.sidecar import CaptureSidecar
from .worker import FeatureAnalysisWorker, WorkerState


@dataclass(frozen=True)
class FeatureRequestSnapshot:
    ui_requests: int
    classification_requests: int
    accepted: int
    throttled_ui: int
    rejected_capacity: int


class FeatureRequestGate:
    """Non-waiting UI/classification requests to a bounded async worker.

    UI requests are limited to at most ``ui_rate_hz`` (maximum 60). This is an
    admission limit, not a 60-Hz computation/delivery guarantee. The renderer
    should refresh the latest completed result independently. Classification
    requests bypass the UI rate limit, NOT worker capacity. Overload and UI
    throttling are immediate returned outcomes; neither calls a sink. Accepted
    work retains the worker's polling/callback, reset, and shutdown contracts.
    Slow analysis/sinks cannot make this gate wait for capacity.
    """

    def __init__(self, worker: FeatureAnalysisWorker, *, ui_rate_hz: float = 60.) -> None:
        if not isinstance(worker, FeatureAnalysisWorker):
            raise TypeError('worker must be FeatureAnalysisWorker')
        if worker.synchronous:
            raise ValueError('On-demand feature requests require an asynchronous worker')
        if (isinstance(ui_rate_hz, bool) or not isinstance(ui_rate_hz, (int, float))
                or not math.isfinite(ui_rate_hz) or not 0 < ui_rate_hz <= 60):
            raise ValueError('ui_rate_hz must be finite and in (0, 60]')
        self._worker = worker
        self._ui_rate_hz = float(ui_rate_hz)
        self._lock = threading.Lock()
        self._next_ui = 0.
        self._ui = self._classification = self._accepted = self._throttled = self._rejected = 0

    @property
    def worker(self) -> FeatureAnalysisWorker:
        return self._worker

    @property
    def ui_rate_hz(self) -> float:
        return self._ui_rate_hz

    def request(self, capture: CaptureSidecar, *, purpose: str = 'classification') -> CaptureSidecar:
        """Return pending iff accepted; never wait for analysis or a free slot.

        A throttled UI request returns ``skipped`` / ``UI request rate limited``.
        Capacity rejection returns ``dropped_overload`` from the worker. These
        are unaccepted requests, not missing terminal callbacks. Re-requesting
        a capture is intentional new work; this gate does not deduplicate IDs.
        """
        if not isinstance(capture, CaptureSidecar):
            raise TypeError('capture must be CaptureSidecar')
        if purpose not in ('ui', 'classification'):
            raise ValueError('purpose must be ui or classification')
        with self._lock:
            if self.worker.state != WorkerState.OPEN:
                raise RuntimeError('Cannot request features after worker closing/reset shutdown')
            now = time.monotonic()
            if purpose == 'ui':
                self._ui += 1
                if now < self._next_ui:
                    self._throttled += 1
                    return replace(capture,features=None,analysis_result=None,feature_status='skipped',
                                   feature_reason='UI request rate limited; job not accepted')
            else:
                self._classification += 1
            result = self.worker.submit(capture)
            if result.feature_status == 'pending':
                self._accepted += 1
                if purpose == 'ui':
                    self._next_ui = now+1./self.ui_rate_hz
            else:
                self._rejected += 1
            return result

    def snapshot(self) -> FeatureRequestSnapshot:
        with self._lock:
            return FeatureRequestSnapshot(self._ui, self._classification, self._accepted,
                                          self._throttled, self._rejected)
