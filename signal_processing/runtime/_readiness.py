"""Windows coalesced readiness bit, not a job counter or ownership ledger.

Clear BEFORE inspecting persistent queues/pipes, then wait only if no work is
ready. Child death cannot strand a mutex: signaling uses a kernel event only.
Each spawned process owns its duplicated handle; close after its users finish.
"""
from __future__ import annotations

import ctypes
import math
import sys
from ctypes import wintypes


class WindowsReadiness:
    def __init__(self):
        if sys.platform != 'win32':
            raise OSError('Windows readiness requires Windows')
        self._bind()
        self._handle = self._api.CreateEventW(None, True, False, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def _bind(self):
        self._api = ctypes.WinDLL('kernel32', use_last_error=True)
        self._api.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        self._api.CreateEventW.restype = wintypes.HANDLE
        for name in ('SetEvent', 'ResetEvent', 'CloseHandle'):
            method = getattr(self._api, name)
            method.argtypes = [wintypes.HANDLE]
            method.restype = wintypes.BOOL
        self._api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._api.WaitForSingleObject.restype = wintypes.DWORD

    def __getstate__(self):
        from multiprocessing.context import assert_spawning, get_spawning_popen
        assert_spawning(self)
        return get_spawning_popen().duplicate_for_child(self._handle)

    def __setstate__(self, handle):
        self._bind()
        self._handle = handle

    def _call(self, name):
        if self._handle is None:
            raise OSError('Readiness handle closed')
        if not getattr(self._api, name)(self._handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def set(self):
        self._call('SetEvent')

    def clear(self):
        self._call('ResetEvent')

    def wait(self, timeout):
        if self._handle is None:
            raise OSError('Readiness handle closed')
        if not math.isfinite(timeout) or not 0 <= timeout < 4294967:
            raise ValueError('Readiness timeout must be finite and bounded')
        status = self._api.WaitForSingleObject(self._handle, math.ceil(timeout * 1000))
        if status == 0:
            return True
        if status == 258:
            return False
        raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self._handle is not None:
            self._call('CloseHandle')
            self._handle = None
