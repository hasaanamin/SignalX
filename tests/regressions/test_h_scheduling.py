"""The finer timer cannot conceal oversleep or ignore cancellation."""
from types import SimpleNamespace
import threading
from unittest.mock import patch
import pytest
from signal_processing.runtime.scheduling import wait_until


def test_overshoot_is_not_relabelled_as_timely():
    now=[10.];sleeps=[]
    def sleep(n):sleeps.append(n);now[0]+=n+.02
    with patch('signal_processing.runtime.scheduling.time',SimpleNamespace(perf_counter=lambda:now[0],sleep=sleep)):
        assert wait_until(10.001024)
    assert now[0]>10.002048 and len(sleeps)==1


def test_cancellation_before_sleep_and_during_bounded_sleep():
    event=threading.Event();event.set()
    assert not wait_until(1e20,event)
    event.clear();now=[10.];sleeps=[]
    def sleep(n):sleeps.append(n);now[0]+=n;event.set()
    with patch('signal_processing.runtime.scheduling.time',SimpleNamespace(perf_counter=lambda:now[0],sleep=sleep)):
        assert not wait_until(11.,event)
    assert sleeps==[.05]


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),'1'])
def test_invalid_deadline(value):
    with pytest.raises(ValueError):wait_until(value)
