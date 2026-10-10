"""Primary-host kernel notification: no job storage and no child-owned lock."""
import multiprocessing
import sys

import pytest


def _signal(event):
    try:
        event.set()
    finally:
        event.close()


if sys.platform == 'win32':
    def test_spawned_signal_is_coalesced_and_child_close_preserves_parent():
        from signal_processing.runtime._readiness import WindowsReadiness
        event = WindowsReadiness()
        child = multiprocessing.get_context('spawn').Process(target=_signal, args=(event,))
        try:
            assert not event.wait(0)
            child.start()
            assert event.wait(5)
            child.join(5)
            assert child.exitcode == 0
            assert event.wait(0)  # manual reset, not consumptive counting
            event.clear()
            assert not event.wait(0)
            for _ in range(1000): event.set()
            event.clear()
            assert not event.wait(0)
        finally:
            if child.is_alive(): child.terminate(); child.join(5)
            child.close()
            event.close()
        event.close()
        with pytest.raises(OSError, match='closed'): event.set()
