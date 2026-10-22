"""Finite signal-absent exposure; not a rare-event false-lock certificate."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
from tests.lte_fixture import waveform


def main():
    rng = np.random.default_rng(20261021)
    rows = []
    stream = StreamInfo("null", "synthetic", 2e6, 2.14e9)
    for kind, count in (("white_noise", 25), ("ar1_0.8_noise", 25),
                        ("unrelated_ofdm", 10), ("tone", 5), ("silence", 1)):
        samples = synchronized = mibs = 0
        for trial in range(count):
            if kind in ("white_noise", "ar1_0.8_noise"):
                iq = (rng.normal(size=160000)+1j*rng.normal(size=160000))/np.sqrt(2)
                if kind == "ar1_0.8_noise":
                    iq = lfilter([.6], [1, -.8], iq)
            elif kind == "unrelated_ofdm":
                iq = waveform(frames=8, with_sync=False, seed=int(rng.integers(2**31)))
            elif kind == "tone":
                iq = np.exp(2j*np.pi*rng.uniform(-.45, .45)*np.arange(160000))
            else:
                iq = np.zeros(160000)
            iq = np.asarray(iq, np.complex64)
            receiver = LTEBroadcastReceiver(stream, LTESyncConfig.low_snr())
            for seq, start in enumerate(range(0, len(iq), 4093)):
                batch = receiver.process_block(IQBlock(stream, seq, start, None, iq[start:start+4093]))
                synchronized += sum(o.state == "SYNCHRONIZED" for o in batch.synchronization)
                mibs += len(batch.mibs)
            batch = receiver.flush()
            synchronized += sum(o.state == "SYNCHRONIZED" for o in batch.synchronization)
            mibs += len(batch.mibs)
            samples += len(iq)
        rows.append({"input": kind, "independent_records": count, "samples": samples,
                     "exposure_s": samples/2e6, "confirmed_observations": synchronized, "mibs": mibs})
    result = {"seed": 20261021, "profile": "low_snr", "results": rows,
              "scope": "Finite synthetic controls only; no arbitrary-noise or rare-event certificate"}
    output = Path("output/lte_low_snr_20261019/negative_controls.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(json.dumps(result, indent=2))
    return 1 if any(r["confirmed_observations"] or r["mibs"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
