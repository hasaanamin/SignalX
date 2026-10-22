"""Scoped Windows performance scheduling, without host power-plan changes."""
from contextlib import contextmanager
import sys
import threading
import struct

_lock = threading.Lock()
_users = 0
_saved = None
_realtime = threading.local()


@contextmanager
def realtime_scope(enabled=True,*,priority=1):
    """Own a Windows MMCSS streaming registration on the acquisition caller."""
    if type(enabled) is not bool:
        raise ValueError('Realtime scheduling selection must be bool')
    if type(priority) is not int or priority not in (0,1,2):
        raise ValueError('Realtime relative priority must be normal, high or critical')
    result = dict(requested=enabled,applied=False,scope_released=False,restored=False,
                  task=None,error=None,priority=priority)
    if not enabled or sys.platform != 'win32':
        try:
            yield result
        finally:
            result.update(scope_released=True,restored=True)
        return
    import ctypes
    from ctypes import wintypes
    if not getattr(_realtime,'depth',0):
        try:
            # MMCSS may replace the caller's affinity with its task profile.
            # Keep explicit/HighQoS placement while retaining MMCSS priority.
            kernel = ctypes.WinDLL('kernel32',use_last_error=True)
            class GroupAffinity(ctypes.Structure):
                _fields_ = [('Mask',ctypes.c_size_t),('Group',wintypes.WORD),('Reserved',wintypes.WORD*3)]
            kernel.GetCurrentThread.restype = wintypes.HANDLE
            kernel.GetThreadGroupAffinity.argtypes = [wintypes.HANDLE,ctypes.POINTER(GroupAffinity)]
            kernel.GetThreadGroupAffinity.restype = wintypes.BOOL
            kernel.SetThreadGroupAffinity.argtypes = [wintypes.HANDLE,ctypes.POINTER(GroupAffinity),ctypes.POINTER(GroupAffinity)]
            kernel.SetThreadGroupAffinity.restype = wintypes.BOOL
            caller_affinity = GroupAffinity()
            thread = kernel.GetCurrentThread()
            if not kernel.GetThreadGroupAffinity(thread,ctypes.byref(caller_affinity)):
                raise ctypes.WinError(ctypes.get_last_error())
            avrt = ctypes.WinDLL('avrt',use_last_error=True)
            avrt.AvSetMmThreadCharacteristicsW.argtypes = [wintypes.LPCWSTR,ctypes.POINTER(wintypes.DWORD)]
            avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
            avrt.AvSetMmThreadPriority.argtypes = [wintypes.HANDLE,ctypes.c_int]
            avrt.AvSetMmThreadPriority.restype = wintypes.BOOL
            avrt.AvRevertMmThreadCharacteristics.argtypes = [wintypes.HANDLE]
            avrt.AvRevertMmThreadCharacteristics.restype = wintypes.BOOL
            index = wintypes.DWORD(0)
            token = avrt.AvSetMmThreadCharacteristicsW('Pro Audio',ctypes.byref(index))
            if not token:
                raise ctypes.WinError(ctypes.get_last_error())
            if not avrt.AvSetMmThreadPriority(token,priority):
                error = ctypes.get_last_error()
                avrt.AvRevertMmThreadCharacteristics(token)
                raise ctypes.WinError(error)
            if not kernel.SetThreadGroupAffinity(thread,ctypes.byref(caller_affinity),None):
                error = ctypes.get_last_error()
                avrt.AvRevertMmThreadCharacteristics(token)
                raise ctypes.WinError(error)
            _realtime.avrt,_realtime.token,_realtime.depth = avrt,token,0
        except OSError as exc:
            result['error'] = str(exc)
            try:
                yield result
            finally:
                result.update(scope_released=True,restored=True)
            return
    _realtime.depth += 1
    result.update(applied=True,task='Pro Audio')
    try:
        yield result
    finally:
        _realtime.depth -= 1
        result['scope_released'] = True
        if not _realtime.depth:
            avrt,token = _realtime.avrt,_realtime.token
            del _realtime.avrt,_realtime.token
            if not avrt.AvRevertMmThreadCharacteristics(token):
                raise ctypes.WinError(ctypes.get_last_error())
            result['restored'] = True


def _cpu_masks(data,allowed):
    """Return eligible/faster-class masks on supported single-group systems."""
    rows = []
    offset = 0
    while offset+8 <= len(data):
        size,kind = struct.unpack_from('<II',data,offset)
        if size < 8 or offset+size > len(data):
            return 0,0
        if kind == 0 and size >= 32:
            _,group,cpu,_,_,_,efficiency,flags = struct.unpack_from('<IHBBBBBB',data,offset+8)
            if group:
                return 0,0  # Do not interpret a group-relative index as a mask bit.
            if allowed & (1 << cpu) and not (flags & 2 and not flags & 4):
                rows.append((cpu,efficiency))
        offset += size
    if offset != len(data):
        return 0,0
    eligible = sum({1 << cpu for cpu,_ in rows})
    if len({row[1] for row in rows}) < 2:
        return eligible,0
    fastest = max(row[1] for row in rows)
    return eligible,sum({1 << cpu for cpu,efficiency in rows if efficiency == fastest})


def _fast_mask(data,allowed):
    return _cpu_masks(data,allowed)[1]


def _fast_core_masks(data,allowed):
    selected = _fast_mask(data,allowed)
    cores = {}
    offset = 0
    while selected and offset+8 <= len(data):
        size,kind = struct.unpack_from('<II',data,offset)
        if kind == 0 and size >= 32:
            _,_,cpu,core,_,_,_,_ = struct.unpack_from('<IHBBBBBB',data,offset+8)
            if selected & (1 << cpu):
                cores[core] = cores.get(core,0) | (1 << cpu)
        offset += size
    return sorted(cores.values(),key=lambda mask:(mask & -mask))


def _placement_mask(cores,eligible,acquisition=False):
    if not cores:
        return 0
    if acquisition:
        # Keep I/O on a separate physical core with stable private caches.
        # Prefer another fast core; otherwise retain one eligible other core.
        candidates = cores[0] if len(cores) > 1 else eligible & ~sum(cores)
    else:
        candidates = cores[-1]
    return candidates & -candidates


def _prefer_fast(kernel,handle,acquisition=False):
    import ctypes
    from ctypes import wintypes
    if not hasattr(kernel,'GetSystemCpuSetInformation'):
        return None,None,[]
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    kernel.GetProcessAffinityMask.restype = wintypes.BOOL
    original,system = ctypes.c_size_t(),ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle,ctypes.byref(original),ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    class GroupAffinity(ctypes.Structure):
        _fields_ = [('Mask',ctypes.c_size_t),('Group',wintypes.WORD),('Reserved',wintypes.WORD*3)]
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    kernel.GetThreadGroupAffinity.argtypes = [wintypes.HANDLE,ctypes.POINTER(GroupAffinity)]
    kernel.GetThreadGroupAffinity.restype = wintypes.BOOL
    current = GroupAffinity()
    if not kernel.GetThreadGroupAffinity(kernel.GetCurrentThread(),ctypes.byref(current)):
        raise ctypes.WinError(ctypes.get_last_error())
    if current.Group:
        return None,None,[]
    query = kernel.GetSystemCpuSetInformation
    query.argtypes = [ctypes.c_void_p,wintypes.ULONG,ctypes.POINTER(wintypes.ULONG),wintypes.HANDLE,wintypes.ULONG]
    query.restype = wintypes.BOOL
    length = wintypes.ULONG()
    query(None,0,ctypes.byref(length),handle,0)
    if not length.value:
        return None,None,[]
    data = ctypes.create_string_buffer(length.value)
    if not query(data,len(data),ctypes.byref(length),handle,0):
        raise ctypes.WinError(ctypes.get_last_error())
    cores = _fast_core_masks(data.raw[:length.value],original.value & current.Mask)
    if not cores:
        return None,None,[]
    # Reserve a hardware thread on the last fast physical core for DSP, keeping
    # it off the first core's interrupt load and avoiding sibling migration.
    # I/O uses another physical core rather than migrating across a CPU class
    # or sharing either DSP SMT sibling.
    eligible = _cpu_masks(data.raw[:length.value],original.value & current.Mask)[0]
    selected = _placement_mask(cores,eligible,acquisition)
    if not selected:
        return None,None,cores
    kernel.SetThreadAffinityMask.argtypes = [wintypes.HANDLE,ctypes.c_size_t]
    kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
    before = kernel.SetThreadAffinityMask(kernel.GetCurrentThread(),selected)
    if not before:
        raise ctypes.WinError(ctypes.get_last_error())
    return before,selected,cores


@contextmanager
def performance_scope(enabled=True,*,prefer_fast_cpus=True,acquisition=False):
    """Request HighQoS for active acquisition owners and restore prior policy.

    Concurrent/nested owners share the request; the last owner restores the
    exact original process power masks. Caller thread power/affinity/priority restore
    independently; process affinity is unchanged. Other platforms retain their scheduler.
    Library DSP receivers do not change process scheduling on construction.
    DSP callers use one hardware thread of the last fast physical core.
    Acquisition callers use a separate physical core. DSP requests the highest
    relative priority within its existing process class, never REALTIME class.
    """
    global _users, _saved
    if type(enabled) is not bool or type(prefer_fast_cpus) is not bool or type(acquisition) is not bool:
        raise ValueError('Performance scheduling selection must be bool')
    result = dict(requested=enabled,applied=False,scope_released=False,restored=False,
                  preferred_cpus=None,preferred_core_cpus=None,affinity_restored=False,thread_qos_applied=False,
                  thread_relative_priority=None,thread_priority_restored=True)
    if not enabled or sys.platform != 'win32':
        try:
            yield result
        finally:
            result.update(scope_released=True,restored=True,affinity_restored=True)
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [('Version',wintypes.ULONG),('ControlMask',wintypes.ULONG),
                    ('StateMask',wintypes.ULONG)]

    with _lock:
        first = not _users
        if first:
            kernel = ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            signature = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
            kernel.GetProcessInformation.argtypes = signature
            kernel.GetProcessInformation.restype = wintypes.BOOL
            kernel.SetProcessInformation.argtypes = signature
            kernel.SetProcessInformation.restype = wintypes.BOOL
            handle = kernel.GetCurrentProcess()
            original = State(1,0,0)
            if not kernel.GetProcessInformation(handle,4,ctypes.byref(original),ctypes.sizeof(original)):
                raise ctypes.WinError(ctypes.get_last_error())
            selected = State(1,original.ControlMask | 1,original.StateMask & ~1)
            if not kernel.SetProcessInformation(handle,4,ctypes.byref(selected),ctypes.sizeof(selected)):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            kernel,handle,original,_ = _saved
        kernel.GetCurrentThread.restype = wintypes.HANDLE
        thread_signature = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        kernel.GetThreadInformation.argtypes = thread_signature
        kernel.GetThreadInformation.restype = wintypes.BOOL
        kernel.SetThreadInformation.argtypes = thread_signature
        kernel.SetThreadInformation.restype = wintypes.BOOL
        thread_original = State(1,0,0)
        priority_original = affinity = None
        try:
            if kernel.GetThreadInformation(kernel.GetCurrentThread(),3,ctypes.byref(thread_original),ctypes.sizeof(thread_original)):
                thread_selected = State(1,thread_original.ControlMask | 1,thread_original.StateMask & ~1)
                if not kernel.SetThreadInformation(kernel.GetCurrentThread(),3,ctypes.byref(thread_selected),ctypes.sizeof(thread_selected)):
                    raise ctypes.WinError(ctypes.get_last_error())
                result['thread_qos_applied'] = True
            elif ctypes.get_last_error() in (50,87):
                thread_original = None  # Older Windows supports process policy only.
            else:
                raise ctypes.WinError(ctypes.get_last_error())
            affinity,selected_mask,cores = _prefer_fast(kernel,handle,acquisition) if prefer_fast_cpus else (None,None,[])
            if not acquisition:
                kernel.GetThreadPriority.argtypes = [wintypes.HANDLE]
                kernel.GetThreadPriority.restype = ctypes.c_int
                kernel.SetThreadPriority.argtypes = [wintypes.HANDLE,ctypes.c_int]
                kernel.SetThreadPriority.restype = wintypes.BOOL
                priority_original = kernel.GetThreadPriority(kernel.GetCurrentThread())
                if priority_original == 0x7fffffff:
                    priority_original = None
                    raise ctypes.WinError(ctypes.get_last_error())
                if not kernel.SetThreadPriority(kernel.GetCurrentThread(),2):
                    raise ctypes.WinError(ctypes.get_last_error())
                result.update(thread_relative_priority=2,thread_priority_restored=False)
        except BaseException:
            if priority_original is not None:
                kernel.SetThreadPriority(kernel.GetCurrentThread(),priority_original)
            if affinity is not None:
                kernel.SetThreadAffinityMask(kernel.GetCurrentThread(),affinity)
            if result['thread_qos_applied']:
                kernel.SetThreadInformation(kernel.GetCurrentThread(),3,ctypes.byref(thread_original),ctypes.sizeof(thread_original))
            if first:
                kernel.SetProcessInformation(handle,4,ctypes.byref(original),ctypes.sizeof(original))
            raise
        if first:
            _saved = kernel,handle,original,cores
        _users += 1
        result['applied'] = True
        if selected_mask is not None:
            result['preferred_cpus'] = [cpu for cpu in range(selected_mask.bit_length()) if selected_mask & (1 << cpu)]
            core = next((mask for mask in cores if mask & selected_mask),selected_mask)
            result['preferred_core_cpus'] = [cpu for cpu in range(core.bit_length()) if core & (1 << cpu)]
    try:
        yield result
    finally:
        error = None
        with _lock:
            _users -= 1
            result['scope_released'] = True
            if not _users:
                if not kernel.SetProcessInformation(handle,4,ctypes.byref(original),ctypes.sizeof(original)):
                    error = ctypes.get_last_error()
                _saved = None
                result['restored'] = error is None
        if thread_original is not None and not kernel.SetThreadInformation(kernel.GetCurrentThread(),3,ctypes.byref(thread_original),ctypes.sizeof(thread_original)):
            error = error or ctypes.get_last_error()
        if priority_original is not None:
            if not kernel.SetThreadPriority(kernel.GetCurrentThread(),priority_original):
                error = error or ctypes.get_last_error()
            else:
                result['thread_priority_restored'] = True
        if affinity is not None and not kernel.SetThreadAffinityMask(kernel.GetCurrentThread(),affinity):
            error = error or ctypes.get_last_error()
        else:
            result['affinity_restored'] = True
        if error is not None:
            raise ctypes.WinError(error)
