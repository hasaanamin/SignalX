"""Observe Windows process policy restoration, including nested/failed runs."""
import sys

import pytest

from signal_processing.runtime._performance import performance_scope


def state(thread=False):
    import ctypes
    from ctypes import wintypes
    class State(ctypes.Structure):
        _fields_ = [('version',wintypes.ULONG),('control',wintypes.ULONG),('selected',wintypes.ULONG)]
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    kernel.GetProcessInformation.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.GetThreadInformation.argtypes = kernel.GetProcessInformation.argtypes
    value = State(1,0,0)
    assert (kernel.GetThreadInformation(kernel.GetCurrentThread(),3,ctypes.byref(value),ctypes.sizeof(value)) if thread
            else kernel.GetProcessInformation(kernel.GetCurrentProcess(),4,ctypes.byref(value),ctypes.sizeof(value)))
    return value.control,value.selected


def thread_affinity():
    import ctypes
    from ctypes import wintypes
    class Group(ctypes.Structure):
        _fields_ = [('Mask',ctypes.c_size_t),('Group',wintypes.WORD),('Reserved',wintypes.WORD*3)]
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    kernel.GetThreadGroupAffinity.argtypes = [wintypes.HANDLE,ctypes.POINTER(Group)]
    value = Group()
    assert kernel.GetThreadGroupAffinity(kernel.GetCurrentThread(),ctypes.byref(value))
    return value.Mask,value.Group


def thread_priority():
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    kernel.GetThreadPriority.argtypes = [wintypes.HANDLE]
    kernel.GetThreadPriority.restype = ctypes.c_int
    return kernel.GetThreadPriority(kernel.GetCurrentThread())


@pytest.mark.skipif(sys.platform != 'win32',reason='Windows native scheduling policy')
@pytest.mark.parametrize('failure',[False,True])
def test_nested_performance_owner_restores_exact_original_process_policy(failure):
    import psutil
    process = psutil.Process()
    before_affinity = process.cpu_affinity()
    before_thread = thread_affinity()
    before = state()
    before_thread_power = state(True)
    before_priority = thread_priority()
    with performance_scope() as outer:
        assert thread_priority() == 2 and outer['thread_relative_priority'] == 2
        if outer['preferred_cpus'] is not None:
            assert thread_affinity()[0] == sum(1 << cpu for cpu in outer['preferred_cpus'])
        assert process.cpu_affinity() == before_affinity
        assert state()[0] & 1 and not state()[1] & 1
        if outer['thread_qos_applied']:
            assert state(True)[0] & 1 and not state(True)[1] & 1
        try:
            with performance_scope() as inner:
                assert thread_priority() == 2
                assert state()[0] & 1 and not state()[1] & 1
                if failure:
                    raise RuntimeError('failed capture sink')
        except RuntimeError:
            assert failure
        assert inner['scope_released']
        assert state()[0] & 1 and not state()[1] & 1
    assert outer['restored'] and state() == before
    assert process.cpu_affinity() == before_affinity
    assert thread_affinity() == before_thread and outer['affinity_restored']
    assert state(True) == before_thread_power
    assert thread_priority() == before_priority and outer['thread_priority_restored']


def test_system_scheduling_selection_is_portable_and_released():
    with performance_scope(False) as policy:
        assert not policy['applied']
    assert policy['scope_released'] and policy['restored']


def test_cpu_topology_selection_respects_masks_classes_allocation_and_groups():
    import struct
    from signal_processing.runtime._performance import _fast_mask
    def row(cpu,efficiency,flags=0,group=0,core=None):
        return struct.pack('<IIIHBBBBBBIQ',32,0,100+cpu,group,cpu,cpu if core is None else core,0,0,efficiency,flags,0,0)
    data = row(0,1)+row(1,1,flags=1)+row(2,0)+row(3,0)
    assert _fast_mask(data,15) == 3
    assert _fast_mask(data,14) == 2
    assert _fast_mask(data,12) == 0  # Caller explicitly permits only one class.
    assert _fast_mask(row(0,1,flags=2)+row(1,1,flags=6)+row(2,0),7) == 2
    assert _fast_mask(row(0,1)+row(1,0,group=1),3) == 0
    assert _fast_mask(data[:15],15) == 0
    from signal_processing.runtime._performance import _cpu_masks
    eligible,fast = _cpu_masks(row(0,1,flags=2)+row(1,1,flags=6)+row(2,0,flags=2)+row(3,0)+row(4,0,flags=6),31)
    assert eligible == 26 and fast == 2 and eligible & ~fast == 24
    assert _cpu_masks(data[:15],15) == (0,0)
    assert _fast_mask(row(0,1)+row(0,1)+row(1,0),3) == 1
    from signal_processing.runtime._performance import _fast_core_masks
    assert _fast_core_masks(row(0,1,core=0)+row(1,1,core=0)+row(2,1,core=2)+row(3,1,core=2)+row(4,0),31) == [3,12]
    from signal_processing.runtime._performance import _placement_mask
    assert _placement_mask([3,12],31) == 4
    assert _placement_mask([3,12],31,True) == 1
    assert _placement_mask([12],28,True) == 16
    assert _placement_mask([12],12,True) == 0
    assert _placement_mask([],31,True) == 0


@pytest.mark.skipif(sys.platform != 'win32',reason='Windows thread affinity')
def test_concurrent_owners_restore_each_thread_before_last_process_owner_exits():
    import threading
    import psutil
    process = psutil.Process()
    before,caller,power = process.cpu_affinity(),thread_affinity(),state()
    ready,start,entered,release = (threading.Event() for _ in range(4))
    failures = []
    def other():
        try:
            original = thread_affinity()
            original_priority = thread_priority()
            ready.set()
            assert start.wait(5)
            with performance_scope():
                entered.set()
                assert release.wait(5)
            assert thread_affinity() == original
            assert thread_priority() == original_priority
        except BaseException as exc:
            failures.append(exc)
            entered.set()
    thread = threading.Thread(target=other)
    thread.start()
    try:
        assert ready.wait(5)
        with performance_scope():
            start.set()
            assert entered.wait(5)
        assert process.cpu_affinity() == before and thread_affinity() == caller
        assert state()[0] & 1 and not state()[1] & 1
    finally:
        start.set()
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not failures
    assert state() == power and thread_affinity() == caller


@pytest.mark.skipif(sys.platform != 'win32',reason='Windows MMCSS registration')
@pytest.mark.parametrize('failure',[False,True])
def test_streaming_registration_is_nested_and_reverted_on_success_or_failure(failure):
    import ctypes
    from ctypes import wintypes
    from signal_processing.runtime._performance import realtime_scope,_realtime
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    kernel.GetThreadPriority.argtypes = [wintypes.HANDLE]
    kernel.GetThreadPriority.restype = ctypes.c_int
    before = kernel.GetThreadPriority(kernel.GetCurrentThread())
    with realtime_scope() as outer:
        assert outer['applied'] and outer['task'] == 'Pro Audio',outer
        try:
            with realtime_scope() as inner:
                assert inner['applied'] and _realtime.depth == 2
                if failure:
                    raise RuntimeError('streaming sink failed')
        except RuntimeError:
            assert failure
        assert inner['scope_released'] and _realtime.depth == 1
    assert outer['restored'] and _realtime.depth == 0
    assert not hasattr(_realtime,'token')
    assert kernel.GetThreadPriority(kernel.GetCurrentThread()) == before


@pytest.mark.skipif(sys.platform != 'win32',reason='Windows MMCSS affinity preservation')
def test_mmcss_preserves_configured_dsp_thread_placement():
    import ctypes
    from ctypes import wintypes
    from signal_processing.runtime._performance import realtime_scope
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    kernel.SetThreadAffinityMask.argtypes = [wintypes.HANDLE,ctypes.c_size_t]
    kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
    before,group = thread_affinity()
    assert group == 0
    selected = before & -before
    assert kernel.SetThreadAffinityMask(kernel.GetCurrentThread(),selected)
    try:
        with performance_scope():
            configured = thread_affinity()
            with realtime_scope() as registration:
                assert registration['applied'],registration
                assert thread_affinity() == configured
                with realtime_scope():
                    assert thread_affinity() == configured
            assert thread_affinity() == configured
    finally:
        assert kernel.SetThreadAffinityMask(kernel.GetCurrentThread(),before)
    assert thread_affinity() == (before,group)


@pytest.mark.skipif(sys.platform != 'win32',reason='Windows physical core placement')
def test_source_excludes_reserved_dsp_core_and_restores_callers():
    import psutil
    from signal_processing.runtime._performance import realtime_scope
    caller = thread_affinity()
    process = psutil.Process().cpu_affinity()
    with performance_scope(acquisition=True) as source:
        with realtime_scope(priority=2):
            source_mask = thread_affinity()
            if source['preferred_cpus'] is not None:
                assert source_mask[0] == sum(1 << cpu for cpu in source['preferred_cpus'])
    assert thread_affinity() == caller
    with performance_scope() as dsp:
        with realtime_scope():
            dsp_mask = thread_affinity()
            if source['preferred_cpus'] is not None and dsp['preferred_cpus'] is not None:
                assert not source_mask[0] & dsp_mask[0]
    assert thread_affinity() == caller
    assert psutil.Process().cpu_affinity() == process
