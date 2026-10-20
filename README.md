# SignalX

Python RF signal-processing foundation for positioned complex IQ: noise
calibration, energy detection, transient segmentation, bounded circular
buffering, exact pre/post-trigger capture and ten feature families producing
56 default scalar features. Includes a desktop dataset testbench with ground
truth comparison and CSV export.

**Status: development candidate, not a frozen or hardware-qualified release.**
Classification, LTE demodulation and SDR hardware acquisition are not completed
components of this package. Scheduled real-time acceptance remains unresolved.

## Get the code and datasets

Install Git LFS before cloning. The binary datasets are tracked through LFS,
not stored as oversized ordinary Git blobs.

```sh
git lfs install
git clone https://github.com/hasaanamin/SignalX.git
cd SignalX
git lfs pull
python -m pip install -e ".[ui,test]"
```

The latest development checks used Windows 11, Python 3.14.7, NumPy 2.5.2 and
SciPy 1.18.0. `pyproject.toml` declares dependency ranges; those ranges do not
constitute qualification of every platform/version. Dataset generation uses
additional TorchSig/PyTorch dependencies documented in the generator script;
they are not necessary to replay the supplied data.

## Dataset UI

On Windows, double-click `run_detector_complete_ui.bat`,
`run_detector_smoke_ui.bat` or `run_hdf5_pipeline_ui.bat`. Alternatively:

```sh
python -m signal_processing.ui.hdf5_app --smoke-dataset detector_dataset_complete
python -m signal_processing.ui.hdf5_app --dataset subset_test.h5
```

The detector tab displays supplied emitter/activity truth, detected intervals,
misses, extras, sample precision/recall, burst detection rate and extracted
features. Select a record or all records; export combined, truth-only or
result-only CSV. Scoring uses one-to-one interval matching, not equal counts.

Select `ar1_spectral` to test the AR1 candidate (N/H=256/8). It fits a fixed
whitener from the authorized quiet prefix, combines 32/64/256-sample windows
with eight spectral bands, and refines event boundaries while retaining exact
raw IQ. The original `compatibility` detector remains available.

Latest characterization on the 240 AR1 records:

| Cases | Interval matches | Recall |
|---|---:|---:|
| All SNRs, all scenarios | 295/360 | 81.94% |
| SNR >=0 dB, all scenarios | 178/180 | 98.89% |
| All SNRs, short-burst scenarios | 94/120 | 78.33% |
| SNR >=0 dB, short-burst scenarios | 58/60 | 96.67% |

These are finite development-dataset observations, not held-out confidence
guarantees. Short bursts at exactly 0 dB achieved >90%.
There were 683 extras overall; fragmentation and weak/DC bursts remain issues.
Only 49/60 nonnegative-SNR short bursts met the tighter endpoint check. AR1
thresholds are model-based, not certified Pfa, and this profile does not
provide online adaptive AR tracking or establish 1 MS/s scheduled operation.

## Layout

```text
signal_processing/          Active production package and dataset UI
tests/                      Unit and regression tests
checks/                     Current independent acceptance/math probes
benchmarks/                 Acquisition and detection evaluation programs
examples/                   Runnable pipeline examples
detector_smoke314/           Eight-record IQ/truth smoke dataset
detector_dataset_complete/  480-record IQ/truth dataset (white and AR1 noise)
subset_test.h5              Supplied HDF5 dataset
```

Each detector dataset includes its own README, manifest, annotations and masks.
Keep each record's absolute sample coordinates and authorized calibration
prefix; never feed evaluation truth into the detector. Overlapping emitters
are scored as activity unions, not individually classified sources.

## Checks and offline evaluation

```sh
python -m pytest -q tests
python -m benchmarks.evaluate_ar1_candidate --output output/ar1_candidate
```

Add `--short-only` for the 30 AR1 short-burst recordings. The evaluator returns
exit 2 when its all-SNR >90% recall gate fails, and exit 1 on execution errors.
Inspect extras, boundaries and feature quality separately even if recall passes.

Current acceptance probes need explicit source routing. In PowerShell:

```powershell
$env:SIGNAL_PROCESSING_SRC = (Get-Location).Path
python -m pytest -q checks/test_signal_processing2_acceptance.py checks/test_new_mechanism_acceptance.py checks/test_prior_repairs_against_signal_processing2.py checks/test_examples2_acceptance.py checks/test_f_public_acceptance.py checks/test_g_public_acceptance.py
```

Before repository housekeeping, the combined implementation passed 1,214
supplied tests and 165 current acceptance checks; counts overlap. Independent
oracles and installed-wheel AR1 checks also passed. Housekeeping preserves
production/test/dataset bytes; it is not a new qualification campaign.

Raw IQ gaps must not splice detector windows or cancel already assembled
feature jobs. Explicit reset/cancel is separate. Processing faults expose
committed/received horizons and recoverable `exception.completed_captures`.
Finish with EOF `flush()` and bounded worker shutdown; handle overload rather
than assuming unlimited buffering. Respect feature validity/reasons:
nonpositive-excess SNR is unavailable, and observed-spectrum bandwidth does
not certify unseen physical emitter tails.

Old ZIPs, installed/build copies, screenshots, generated logs and audit reports
are archived outside the repository rather than presented as active source.
New run outputs and local environments are ignored by Git. See
[HISTORY.md](HISTORY.md) for the explicitly reconstructed development history.
