"""Bounded, ordered publication owned by one acquisition caller."""
import queue
import threading
import time


class OrderedPublisher:
    def __init__(self,callback,capacity=64):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Publication capacity must be positive')
        self._callback = callback
        self._capacity = capacity
        self._queue = queue.Queue(maxsize=capacity)
        self._slots = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()
        self._error = None
        self._closing = self._sentinel = False
        self._submitted = self._published = self._failed = self._rejected = self._peak = 0
        self._thread = threading.Thread(target=self._work,name='ordered-lte-output',daemon=True)
        self._thread.start()

    def submit(self,value):
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f'Publication failed: {self._error}')
            if self._closing:
                raise RuntimeError('Publication is closing')
            if not self._slots.acquire(False):
                self._rejected += 1
                raise RuntimeError('Bounded publication capacity exhausted')
            self._queue.put_nowait((value,))
            self._submitted += 1
            self._peak = max(self._peak,self._submitted-self._published-self._failed)

    def _work(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                failed = self._error is not None
                if not failed:
                    try:
                        self._callback(item[0])
                    except BaseException as exc:
                        with self._lock:
                            self._error = f'{type(exc).__name__}: {exc}'
                        failed = True
                with self._lock:
                    if failed:
                        self._failed += 1
                    else:
                        self._published += 1
                self._slots.release()
            finally:
                self._queue.task_done()

    def finish(self,timeout=5.):
        deadline = time.monotonic()+timeout
        with self._lock:
            self._closing = True
        if not self._sentinel:
            try:
                self._queue.put(None,timeout=max(0.,deadline-time.monotonic()))
            except queue.Full:
                raise TimeoutError('Publication remains blocked') from None
            self._sentinel = True
        self._thread.join(max(0.,deadline-time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError('Publication remains blocked')
        if self._error is not None:
            raise RuntimeError(f'Publication failed: {self._error}')

    def snapshot(self):
        with self._lock:
            return dict(capacity=self._capacity,submitted=self._submitted,published=self._published,
                failed=self._failed,rejected=self._rejected,
                outstanding=self._submitted-self._published-self._failed,peak_outstanding=self._peak,
                writer_alive=self._thread.is_alive(),error=self._error)
