"""Explicit owned-process HighQoS experiment; never changes the host power plan.

https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation
The OS may otherwise infer QoS from visibility/background state. This setting
does not alter process priority, guarantee deadlines, or disable thermal limits.
"""
import ctypes
from ctypes import wintypes
import sys


class _State(ctypes.Structure):
    _fields_ = [('Version',wintypes.ULONG),('ControlMask',wintypes.ULONG),('StateMask',wintypes.ULONG)]


class OwnedHighQoS:
    def __init__(self, pids):
        if sys.platform!='win32': raise OSError('HighQoS requires Windows')
        api=self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        api.OpenProcess.restype=wintypes.HANDLE
        for name in ('GetProcessInformation','SetProcessInformation'):
            f=getattr(api,name);f.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD];f.restype=wintypes.BOOL
        api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle.restype=wintypes.BOOL
        api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];api.WaitForSingleObject.restype=wintypes.DWORD
        self.leases=[];self.observations=[]
        try:
            for pid in pids:
                handle=api.OpenProcess(0x100000|0x1000|0x0200,False,pid)
                if not handle: raise ctypes.WinError(ctypes.get_last_error())
                before=_State(1,0,0)
                if not api.GetProcessInformation(handle,4,ctypes.byref(before),ctypes.sizeof(before)):
                    api.CloseHandle(handle);raise ctypes.WinError(ctypes.get_last_error())
                self.leases.append((handle,before))
                target=_State(1,before.ControlMask|1,before.StateMask & ~1)
                if not api.SetProcessInformation(handle,4,ctypes.byref(target),ctypes.sizeof(target)):
                    raise ctypes.WinError(ctypes.get_last_error())
                after=_State(1,0,0)
                if not api.GetProcessInformation(handle,4,ctypes.byref(after),ctypes.sizeof(after)):
                    raise ctypes.WinError(ctypes.get_last_error())
                assert after.ControlMask&1 and not after.StateMask&1
                self.observations.append(dict(pid=pid,before=[before.ControlMask,before.StateMask],
                    after=[after.ControlMask,after.StateMask]))
        except BaseException:
            self.close();raise

    def close(self):
        errors=[]
        for handle,before in reversed(self.leases):
            try:
                # Dead children have no surviving scheduling policy to restore.
                if self.api.WaitForSingleObject(handle,0)==258:
                    if not self.api.SetProcessInformation(handle,4,ctypes.byref(before),ctypes.sizeof(before)):
                        errors.append(ctypes.WinError(ctypes.get_last_error()))
            finally:self.api.CloseHandle(handle)
        self.leases.clear()
        if errors:raise errors[0]
