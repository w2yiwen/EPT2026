# Training-convergence data contract

The figure reads the formal four-model matrix only:

- `LSTR`
- `GateHUB`
- `TeSTra`
- `EPT-Net (ours)`

Each `results/dual4090_seed42_3way/<experiment>/seed_<seed>/` run directory
must contain:

1. `history.json`, written by the existing trainer; and
2. `validation_epoch_metrics.jsonl`, containing measured validation results.

Each non-empty JSONL line represents one genuinely evaluated epoch. Two forms
are accepted:

```json
{"epoch": 5, "split": "validation", "metric_name": "event.event_ap_iou_0.5", "value": 0.412}
```

or:

```json
{"epoch": 5, "split": "validation", "metrics": {"event": {"event_ap_iou_0.5": 0.412}}}
```

Requirements:

- values must be finite and lie in `[0, 1]`;
- an epoch may appear at most once per run;
- all models and seeds must use the same real evaluation epochs;
- all four models must use the same seed set;
- points must come from the validation split, never the test split;
- omitted epochs stay omitted—there is no interpolation or smoothing.

From the project root, after formal training and validation-trajectory
evaluation have finished:

```bash
.venv/bin/python fig/fig01_training_convergence/plot_fig01_training_convergence.py
```

Explicit roots or run directories can be supplied by repeating `--run`, for
example:

```bash
.venv/bin/python fig/fig01_training_convergence/plot_fig01_training_convergence.py \
  --run 'LSTR=results/dual4090_seed42_3way/lstr_marlin11_eeg_ppg_video_no_text' \
  --run 'GateHUB=results/dual4090_seed42_3way/gatehub_marlin11_eeg_ppg_video_no_text' \
  --run 'TeSTra=results/dual4090_seed42_3way/testra_marlin11_eeg_ppg_video_no_text' \
  --run 'EPT-Net (ours)=results/dual4090_seed42_3way/eptnet_marlin11_eeg_ppg_video_no_text'
```

The script fails when any required source is absent or incompatible. Its
`--demo` option is only for visibly watermarked layout/export QA and must never
be cited as an experiment result.

The completed seed-42 campaign inspected on 2026-10-03 does not contain these
per-epoch validation metric files. Its loss history and final test metrics are
usable for the standalone loss figure and Table 2, but they cannot reconstruct
a validation AP convergence trajectory. Intermediate checkpoints or metrics
must be retained during a future run; final test metrics must never be copied
into this file.
