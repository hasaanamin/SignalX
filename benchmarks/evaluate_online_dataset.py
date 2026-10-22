"""Fixed 48-record development characterization; annotations only select/score.

Every record uses the SAME online_ar1 profile; no generator power/rho/SNR or
quiet-prefix label is supplied to the streaming pipeline. Prints JSON lines.
"""
import json
from pathlib import Path

from signal_processing.ui.smoke_runner import SmokeConfig, inspect_smoke_dataset, run_smoke_record


def main():
    root = Path("detector_dataset_complete")
    manifest = [json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
    selected = {row["record_id"] for row in manifest if row["repetition"] == 0 and row["snr_sweep_db"] >= 0}
    totals = dict(records=0, truths=0, matched=0, extras=0, failed_analyses=0, errors=0, exact_iq=True)
    for record in inspect_smoke_dataset(root):
        if record.record_id not in selected:
            continue
        try:
            result = run_smoke_record(record, SmokeConfig(detector_profile="online_ar1"))
            m = result.metrics
            print(json.dumps(dict(record_id=record.record_id, **m)), flush=True)
            totals["records"] += 1
            for key, source in (("truths", "truth_activities"), ("matched", "matched"), ("extras", "extras"), ("failed_analyses", "feature_failures")):
                totals[key] += m[source]
            totals["exact_iq"] &= m["exact_capture_iq"]
        except Exception as exc:
            totals["errors"] += 1
            print(json.dumps(dict(record_id=record.record_id, error=f"{type(exc).__name__}: {exc}")), flush=True)
    print(json.dumps(dict(summary=totals)), flush=True)
    return int(totals["errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
