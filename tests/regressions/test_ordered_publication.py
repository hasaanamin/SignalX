"""Real bounded writer ownership, ordering, failure and drain checks."""
import threading

import pytest

from signal_processing.runtime._publication import OrderedPublisher


def test_slow_writer_never_blocks_submit_and_capacity_includes_active_item():
    entered,release = threading.Event(),threading.Event()
    received = []
    def sink(value):
        entered.set()
        assert release.wait(5)
        received.append(value)
    writer = OrderedPublisher(sink,capacity=2)
    try:
        writer.submit(0)
        assert entered.wait(5)
        writer.submit(1)
        with pytest.raises(RuntimeError,match='capacity'):
            writer.submit(2)
        assert writer.snapshot()['outstanding'] == writer.snapshot()['peak_outstanding'] == 2
    finally:
        release.set()
        writer.finish()
    writer.finish()
    assert received == [0,1]
    assert writer.snapshot()['published'] == 2
    assert writer.snapshot()['rejected'] == 1
    assert writer.snapshot()['outstanding'] == 0 and not writer.snapshot()['writer_alive']


def test_writer_failure_accounts_for_every_accepted_item_and_is_observable():
    release = threading.Event()
    received = []
    def sink(value):
        assert release.wait(5)
        if value == 1:
            raise OSError('real publication failure')
        received.append(value)
    writer = OrderedPublisher(sink,capacity=3)
    for value in range(3):
        writer.submit(value)
    release.set()
    with pytest.raises(RuntimeError,match='real publication failure'):
        writer.finish()
    state = writer.snapshot()
    assert received == [0] and state['published'] == 1 and state['failed'] == 2
    assert state['submitted'] == 3 and state['outstanding'] == 0 and not state['writer_alive']
    with pytest.raises(RuntimeError,match='real publication failure'):
        writer.submit(3)


def test_blocked_writer_timeout_preserves_ownership_and_can_finish_after_release():
    entered,release = threading.Event(),threading.Event()
    def sink(value):
        entered.set()
        assert release.wait(5)
    writer = OrderedPublisher(sink,capacity=1)
    try:
        writer.submit(1)
        assert entered.wait(5)
        with pytest.raises(TimeoutError):
            writer.finish(timeout=.001)
        assert writer.snapshot()['outstanding'] == 1 and writer.snapshot()['writer_alive']
        with pytest.raises(RuntimeError,match='closing'):
            writer.submit(2)
    finally:
        release.set()
        writer.finish()
    assert writer.snapshot()['published'] == 1 and writer.snapshot()['outstanding'] == 0
