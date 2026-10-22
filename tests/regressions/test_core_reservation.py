"""Real native idle/atomic stop, engine lifetime and scoped thread cleanup."""
import gc
import platform
import sys
import threading

import pytest

from signal_processing.runtime._core_reservation import core_reservation
from signal_processing.runtime._performance import performance_scope

NATIVE = sys.platform == 'win32' and platform.machine().lower() in ('amd64','x86_64')


def test_disabled_and_non_smt_policy_creates_no_thread():
    before = set(threading.enumerate())
    with core_reservation([0,1],enabled=False) as disabled:
        assert not disabled['applied'] and disabled['reason'] == 'disabled'
    with core_reservation([0]) as single:
        assert not single['applied']
    assert disabled['stopped'] and single['stopped']
    assert set(threading.enumerate()) == before


@pytest.mark.parametrize('invalid',[[1,0],[0,0],[-1,0],[0,64],[False,1],[]])
def test_invalid_core_indices_fail_before_starting_thread(invalid):
    with pytest.raises(ValueError,match='Core CPUs'):
        with core_reservation(invalid):
            raise AssertionError('Invalid CPU policy entered')


@pytest.mark.skipif(not NATIVE,reason='Windows x64 native SMT reservation')
@pytest.mark.parametrize('failure',[False,True])
def test_native_reservation_is_gil_free_and_restores_after_exception_and_cache_clear(failure):
    import ctypes
    from signal_processing.runtime._core_reservation import _native_idle
    before = set(threading.enumerate())
    with performance_scope() as dsp:
        core = dsp['preferred_core_cpus']
        if core is None or len(core) != 2:
            pytest.skip('Target has no automatically selected SMT pair')
        try:
            with core_reservation(core) as policy:
                assert policy['applied'] and not policy['stopped']
                assert policy['cpus'] == core[1:]
                assert policy['worker_policy']['thread_relative_priority'] == 2
                assert len(set(threading.enumerate())-before) == 1
                # The active context must retain its engine/pointers even if
                # another caller clears/rebuilds the process-local JIT cache.
                _native_idle.cache_clear()
                gc.collect()
                rebuilt = _native_idle()
                flag = ctypes.c_int32(0)
                rebuilt[2](ctypes.byref(flag))
                assert flag.value == 1
                rebuilt[1](ctypes.byref(flag))  # Already stopped: returns directly.
                assert sum(range(1000)) == 499500  # Native idle has released the GIL.
                if failure:
                    raise RuntimeError('real owner failure')
        except RuntimeError as exc:
            assert failure and str(exc) == 'real owner failure'
        assert policy['stopped'] and policy['affinity_restored'] and policy['error'] is None
        assert policy['worker_policy']['scope_released']
        assert policy['worker_policy']['thread_priority_restored']
    assert set(threading.enumerate()) == before


@pytest.mark.skipif(not NATIVE,reason='Windows x64 native SMT reservation')
def test_worker_start_failure_is_propagated_and_joined(monkeypatch):
    import signal_processing.runtime._core_reservation as reservation_module
    before = set(threading.enumerate())
    def fail(mask):
        raise OSError('injected affinity failure')
    monkeypatch.setattr(reservation_module,'_set_affinity',fail)
    with pytest.raises(RuntimeError,match='injected affinity failure'):
        with core_reservation([0,1]):
            raise AssertionError('Failed reservation entered')
    assert set(threading.enumerate()) == before


@pytest.mark.skipif(not NATIVE,reason='Windows x64 native SMT reservation')
def test_early_close_is_idempotent_and_diagnostics_remain_serializable():
    import json
    with performance_scope() as dsp:
        core = dsp['preferred_core_cpus']
        if core is None or len(core) != 2:
            pytest.skip('Target has no automatically selected SMT pair')
        with core_reservation(core) as policy:
            assert policy['applied']
            policy.close()
            policy.close()
            assert policy['stopped'] and policy['affinity_restored']
            assert json.loads(json.dumps(policy))['worker_policy']['thread_priority_restored']
