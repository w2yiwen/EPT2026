# EPT-Net training-loss data contract

The formal figure reads one or more measured EPT-Net run directories named
`seed_<integer>`. Each run must contain a non-empty `history.json` list with
contiguous one-indexed epochs and finite `train.total` and `val.total` values.

The default source is:

```text
results/dual4090_seed42_3way/
  eptnet_marlin11_eeg_ppg_video_no_text/
    seed_*/history.json
```

For several seeds, curves are the epoch-wise arithmetic mean over the common
completed-epoch prefix and bands are the sample standard deviation (`ddof=1`).
For one seed, the measured curve is drawn without an uncertainty band. The
checkpoint marker for each seed is the first epoch attaining that seed's
minimum validation total loss. There is no smoothing or interpolation.

From the project root:

```bash
.venv/bin/python fig/fig01_training_loss/plot_fig01_training_loss.py
```

An alternative experiment root or explicit run directory can be supplied with
`--run`. The script exports synchronized PDF, SVG, and 500-dpi PNG files plus a
QA report beside the plotting script.
