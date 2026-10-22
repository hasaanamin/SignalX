"""Byte validation must stay bounded independently of recording duration."""

from types import SimpleNamespace

import numpy as np

from benchmarks.qualify_realtime_1920 import CaptureByteOracle


def test_recorded_byte_oracle_remaps_both_directions_and_detects_corruption(tmp_path):
    rng = np.random.default_rng(65381)
    iq = (rng.normal(size=400000) + 1j * rng.normal(size=400000)).astype(np.complex64)
    path = tmp_path / "recording.cf32"
    iq.tofile(path)
    oracle = CaptureByteOracle(path)
    try:
        for start in (0, 131000, 260000, 17, 399000):
            end = min(len(iq), start + 900)
            event = SimpleNamespace(
                event_start_sample=start, event_end_sample=end, iq=iq[start:end]
            )
            assert oracle.matches(event)
            assert len(oracle.mapping) <= 1024 * 1024
            damaged = event.iq.copy()
            damaged[-1] += 0.1
            assert not oracle.matches(
                SimpleNamespace(event_start_sample=start, event_end_sample=end, iq=damaged)
            )
        assert not oracle.matches(
            SimpleNamespace(event_start_sample=400000, event_end_sample=400001, iq=iq[:1])
        )
    finally:
        oracle.close()
    assert oracle.file.closed and oracle.mapping.closed
