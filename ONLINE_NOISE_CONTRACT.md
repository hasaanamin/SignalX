# Blind online noise tracking — development candidate

```python
from signal_processing.core import StreamInfo
from signal_processing.detection import OnlineNoiseConfig
from signal_processing.runtime.pipeline import StreamingCapturePipeline

stream = StreamInfo("receiver-1", "sdr", 1_000_000)
pipeline = StreamingCapturePipeline(
    stream, window_size=256, hop_size=64, ring_capacity=16384,
    online_noise=OnlineNoiseConfig(model="ar1"), pre_trigger_samples=256,
)
# Feed positioned IQBlock objects; consume returned capture sidecars.
# At EOF: pipeline.flush(); pipeline.shutdown().
```

No SNR, noise power, rho, truth positions or quiet-prefix argument is used.
`model="gaussian"` assumes iid circular complex Gaussian noise. `model="ar1"`
estimates a complex lag-1 coefficient, including near-zero for white noise.
It does not model arbitrary colored noise. Fixed certificates, configured SOS
bands, legacy spectral-bank/refinement and batching cannot be combined with it.

## Learning and limits

No external calibration is required, but there is an internal learning interval:
eight 1024-sample frames by default (8.192 ms at 1 MS/s). Failed diagnostics extend
learning. No detections or valid reference are fabricated during startup. Received
raw samples still enter the bounded ring. Startup bursts may be missed and remain
counted in evaluation. Retention is not an unlimited recording of startup IQ.

The method assumes a majority of noise-dominated frames in its learning history.
Median frame statistics and mean/correlation/propriety checks reject some obvious
burst, tonal and zero-input contamination. They cannot distinguish a persistent
Gaussian-like signal from Gaussian noise using single-channel IQ alone. This is
not universal calibration-free detection or a certified minimum SNR.

## Adaptation and causality

- Noise level, mean and complex AR1 coefficient are estimated from actual IQ.
- A decision uses the PREVIOUS reference. Only afterward may that hop contribute
  to a later reference. Active/candidate/post-trigger samples are excluded;
  partial training frames are discarded across exclusions. Separate frames never
  create synthetic lag pairs across an excluded interval.
- Defaults refit every four eligible frames (4096 samples); `update_frames=1`
  allows more frequent fitting at greater CPU cost. Default smoothing time is
  20 ms of accepted noise samples, not wall time spent frozen.
- Each window is re-whitened from N+1 raw samples using ONE current coefficient.
  Samples transformed with different coefficients are never mixed in a window.
- ON/OFF thresholds follow physical innovation power using Gamma model quantiles.
  The new profile uses one full-band window, NOT the old eleven-branch bank.
- Reference identity changes with fitted parameters/revision. Profile identity
  includes estimator controls, N/H, Pfa and statistic semantics.
- Changing the statistic invalidates fixed confidence evidence. This mode is
  always `model_only`, `supported_pfa=False`. Requested Pfa is not measured event
  false-alarm performance or a held-out confidence guarantee.

Gradual drift can be followed. An abrupt persistent increase may instead trigger
detection and freeze adaptation: it might be a real transmission. It is not
silently learned away. Default expiry is 1,000,000 received/pending samples without
a successful update. Expiry faults with existing completed-capture recovery;
explicit reset/relearning is required. Gaps restart learning without cancelling
already assembled feature jobs.

`pipeline.noise_estimator.diagnostics()` exposes STARTING/VALID/STALE, reason,
revision, coefficient, innovation power, age, rejected fits and retained counts.
VALID means usable under these assumptions, not certified. `freeze(True)` pauses
learning; unfreeze cannot splice partial frames across a pause. Reset clears
learned state. Capture `noise_provenance` identifies the starting model/assumptions.

## Bounds and interfaces

Default history: 8192 complex128 samples plus one preallocated 1024-sample frame.
Detector residual: at most N+1 raw samples. Temporary fit arrays scale with these
fixed bounds. Ring, capture, worker and result limits remain unchanged. There is
no recording-length IQ/model history. Whitening affects detection only; raw IQ,
pre-trigger retention, ten feature families and the 56-key schema are preserved.
The online detector requires coordinated pipeline hops and rejects standalone
single-measurement/warmup/fixed-certificate APIs.

## UI and compatibility

Detector UI/CLI default: `ar1_spectral` with fixed authorized-prefix calibration.
`online_ar1` and `online_gaussian` remain selectable opt-in modes. Both
start at sample zero, ignore calibration-prefix annotations and score startup
misses. SNR/noise-model/truth metadata are evaluation/display inputs, not supplied
noise power or coefficients. Programmatic `SmokeConfig()` retains its legacy
default; explicitly select an online profile for that API.

The HDF5 tab defaults to its previous seeded adaptive-noise method; its
"Adaptive noise floor" checkbox controls that legacy estimator. To use this
tracker with HDF5, explicitly select an online mode through `HarnessConfig`
or the headless CLI's `--detector-profile online_ar1` / `online_gaussian` option.
That harness still constructs synthetic guards and optionally normalizes frames
using SNR labels. It is NOT raw receiver validation. The online estimator does
not receive those generator values. Programmatic `HarnessConfig()` retains legacy
behavior; select `noise_estimation_mode="online_ar1"` or `"online_gaussian"`.

Legacy compatibility/fitted AR1 profiles remain reproducible. Their old detection
percentages do not qualify the new statistic. The new full-band mode does not yet
offer the spectral bank's short/narrowband sensitivity.

## Checks

```powershell
py -3.14 -m pytest -q tests/test_online_noise.py tests/test_smoke_dataset_ui.py tests/test_hdf5_pipeline_ui.py
py -3.14 -m benchmarks.evaluate_online_noise
py -3.14 -m pytest -q tests
```

Synthetic characterization uses 21 opportunities/cell (one during startup), +10 dB,
512-sample post-startup bursts, fixed/drifting Gaussian/AR1 backgrounds, real
features and exact-IQ checks. It is not a 400-trial detection qualification.

**Lossless 1 MS/s operation is not established.** Initial unpaced measurements
remain below 1 MS/s. Do not reuse old acquisition percentages for this new path.
Hardware and universal robustness remain unverified.

Mathematical primitive: [SciPy Gamma distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html).
This engineering estimator is not claimed to implement or inherit qualification
from a published minimum-statistics algorithm.
