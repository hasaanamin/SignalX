"""Monotonic source scheduling without a coarse Event timeout timer.

Cancellation is polled at most every 50 ms while sleeping. Wake-up is NOT a
deadline guarantee: callers must measure lateness and account expired samples.
No OS timer-resolution or interpreter-global setting is changed.
"""
import math
import time


def wait_until(deadline: float, cancellation=None) -> bool:
    """Wait for an absolute perf_counter deadline, or return False on cancellation."""
    if isinstance(deadline, bool) or not isinstance(deadline, (int,float)) or not math.isfinite(deadline):
        raise ValueError('deadline must be a finite monotonic-clock value')
    while cancellation is None or not cancellation.is_set():
        remaining=deadline-time.perf_counter()
        if remaining<=0: return True
        time.sleep(min(remaining,.05))
    return False
