"""Separate traced ownership/bounds probe, not an unperturbed timing run."""
import gc
import argparse
import json
from pathlib import Path
import tracemalloc

import numpy as np
import psutil

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
from tests.lte_pbch_fixture import waveform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("compatibility", "low_snr"), default="compatibility")
    parser.add_argument("--output", type=Path, default=Path("output/lte_pbch_20261019/retention.json"))
    args = parser.parse_args()
    output = args.output
    process = psutil.Process()
    before = process.memory_info().rss
    allocation = np.empty(128*1024*1024, np.uint8)
    allocation.fill(0x55)
    after = process.memory_info().rss
    responsive = after-before >= 64*1024*1024
    del allocation
    gc.collect()
    stream = StreamInfo("memory", "synthetic", 1_920_000., 2.14e9)
    config = LTESyncConfig.low_snr(1_920_000.) if args.profile == "low_snr" else LTESyncConfig(input_sample_rate_hz=1_920_000.)
    receiver = LTEBroadcastReceiver(stream, config)
    quiet = np.zeros(4096, np.complex64)
    rows = []
    tracemalloc.start()
    for seq in range(1000):
        result = receiver.process_block(IQBlock(stream, seq, seq*4096, None, quiet))
        assert not result.mibs
        if seq in (99, 499, 999):
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            state = receiver.snapshot()
            assert state["retained_broadcast_samples"] <= (62400 if args.profile == "low_snr" else 24000)
            assert state["retained_bch_frames"] == state["pending_broadcast_frames"] == 0
            rows.append({"received_samples": (seq+1)*4096, "current_traced_bytes": current,
                         "peak_traced_bytes": peak, "rss_bytes": process.memory_info().rss,
                         "snapshot": state})
    tracemalloc.stop()
    # All legitimate caches are already populated by the first checkpoint.
    bounded = rows[-1]["current_traced_bytes"]-rows[0]["current_traced_bytes"] < 1_000_000
    receiver.reset()
    iq = waveform(frames=64, rate=1_920_000.)
    returned = 0
    for seq, start in enumerate(range(0, len(iq), 4096)):
        returned += len(receiver.process_block(IQBlock(stream, seq, start, None, iq[start:start+4096])).mibs)
        assert receiver.snapshot()["retained_bch_frames"] <= 4
    returned += len(receiver.flush().mibs)
    assert returned == 64
    dense = receiver.snapshot()
    receiver.reset()
    reset = receiver.snapshot()
    assert reset["retained_broadcast_samples"] == reset["retained_bch_frames"] == 0
    report = {"profile": args.profile, "rss_control": {"allocation_bytes": 128*1024*1024, "rss_before": before,
                              "rss_after_touch": after, "responsive": responsive},
              "timing_scope": "Traced diagnostic only, not real-time acceptance",
              "quiet_checkpoints": rows, "traced_growth_bound_passed": bounded,
              "event_dense_returned_mibs": returned, "event_dense_snapshot": dense,
              "reset_snapshot": reset}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    return 0 if bounded else 1


if __name__ == "__main__":
    raise SystemExit(main())
