# Paper figures

These scripts read real experiment artifacts from `results/`; they never embed,
estimate, smooth, or repair result values. Each figure is rendered from one
canvas to PDF, SVG, and 500 dpi PNG, then checked against its `figure.yaml`.
The scripts require an exact Arial installation and fail rather than silently
substituting another font.

- `fig03` is the seed-42 main comparison of EPT-Net, LSTR, GateHUB, and TeSTra.
- `fig04` is the paper evidence figure for continuous event
  tracking/localization.
- `fig05` is an exploratory Full-versus-Video comparison. Different seeds are
  allowed, but the real seed metadata is displayed and retained in QA output.

From the project root:

```bash
# Layout and export QA only. Every output is visibly marked DEMO and reads
# deterministic synthetic fixtures under fig/_demo/; never cite these values.
python fig/generate_all.py --suite paper --demo
```

The generated files and `<figure-name>.qa-report.json` stay beside each plotting script and
are ignored by Git because they can be reproduced from the result artifacts.

## Post-training convergence figure

Generate every currently supported post-training artifact with:

```bash
.venv/bin/python fig/generate_training_summary.py \
  --campaign-root results/dual4090_seed42_3way
```

This always generates the measured EPT-Net training/validation total-loss
figure and Table 2 from the protocol-matched test metrics. It generates the
two-panel loss/AP convergence figure only when every included run contains a
real `validation_epoch_metrics.jsonl`; otherwise it reports that panel as
`SKIPPED` and never substitutes test metrics or validation loss. Table 2 is
written as Markdown, LaTeX, CSV, and a provenance manifest under
`fig/table02_overall_performance/`.

The present campaign contains one matched run (seed 42). Its table is therefore
a set of point estimates, not mean ± standard deviation. The project protocol
trains on all 11 sessions, so the table is labelled as a fixed, training-included
test evaluation view rather than a held-out test set.

After the matched `LSTR`, `GateHUB`, `TeSTra`, and `EPT-Net (ours)` runs have
finished, generate the full-width loss/AP convergence figure with:

```bash
.venv/bin/python fig/generate_all.py --suite training
```

The command reads each run's `history.json` plus measured validation Event
AP@IoU=0.5 points from `validation_epoch_metrics.jsonl`. It requires identical
seed sets and evaluation epochs across all four models, and fails rather than
using test metrics, interpolation, or unmatched runs. The complete input
schema is in
[`fig01_training_convergence/DATA_CONTRACT.md`](fig01_training_convergence/DATA_CONTRACT.md).

## Formal paper figures

Select the qualitative sequence before looking at its rendered prediction
trace, then run:

```bash
python fig/generate_all.py --suite paper \
  --main-aggregate 'EPT-Net=results/eptnet_marlin11_eeg_ppg_video_no_text/aggregate.json' \
  --main-aggregate 'LSTR=results/lstr_marlin11_eeg_ppg_video_no_text/aggregate.json' \
  --main-aggregate 'GateHUB=results/gatehub_marlin11_eeg_ppg_video_no_text/aggregate.json' \
  --main-aggregate 'TeSTra=results/testra_marlin11_eeg_ppg_video_no_text/aggregate.json' \
  --predictions results/<full-run>/test_metrics_predictions.jsonl \
  --events results/<full-run>/test_metrics_events.json \
  --metrics results/<full-run>/test_metrics.json \
  --sample-id '<predeclared-sample-id>' \
  --modality-aggregate 'Full=results/eptnet_marlin11_eeg_ppg_video_no_text/aggregate.json' \
  --modality-aggregate 'Video=results/eptnet_marlin11_video_only_no_text/aggregate.json'
```

The full data contracts and selection safeguards are in
[`FIGURE_CONTRACTS.md`](FIGURE_CONTRACTS.md). The formal script rejects
out-of-range probabilities, mismatched event sequence counts, undeclared
qualitative sample IDs, and missing thresholds. Same-run identity for the
dynamic trace must be confirmed upstream.
