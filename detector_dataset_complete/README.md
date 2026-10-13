# Detector test dataset

Read `dataset.json` and `manifest.jsonl`. Each record has complex64 `iq.npy`, `annotations.json` labeling EVERY emitter, and `truth_masks.npz` containing `any_signal` and `active_emitter_count`. `signals.csv` is the combined emitter table.

All sample/time ends are exclusive. `background` gives the noise interval. The calibration prefix is genuinely noise-only; the evaluator trains on it through the pipeline's causal detection path. Reset all streaming state between records.

Load IQ with `np.load(record_path / 'iq.npy', allow_pickle=False)`. Feed contiguous chunks with start_sample equal to their array index. Keep all emitters in their own ground-truth entries, even during overlap.

For a full-band activity detector use `fullband_activity_intervals`: overlapping emitters produce one aggregate activity region. The built-in evaluator reports one-to-one activity event matches, sample precision/recall, boundary errors, noise-only false events, and temporal coverage of every emitter. Coverage does not establish source identity. For frequency-selective testing, nominal bandwidth is only a generator parameter; account for filter leakage and spectral tails.

SNR means active emitter power / expected full-band noise power. The script also reports realized SNR and full-band SINR. It uses TorchSig's actual builders, with controlled scheduling, frequency shifting and noise addition. No post-mix normalization, clipping, channel-fading or time-warp impairment is silently added.

All timing is offline. Data and event-rate observations do not certify rare-event Pfa, real-noise detection, real-time acquisition or multi-emitter separation.
