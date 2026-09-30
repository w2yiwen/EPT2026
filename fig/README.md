# Paper figures

These scripts read real experiment artifacts from `results/`; they never embed,
estimate, smooth, or repair result values. Each figure is rendered from one
canvas to PDF, SVG, and 500 dpi PNG, then checked against its `figure.yaml`.
The scripts require an exact Arial installation and fail rather than silently
substituting another font.

The figure suite has two layers:

- `fig01`–`fig03` are legacy optimization/comparison diagnostics.
- `fig04` is the short-paper evidence figure for continuous event
  tracking/localization.
- `fig05` is an exploratory Full-versus-Video comparison. Different seeds are
  allowed, but the real seed metadata is displayed and retained in QA output.

From the project root:

```bash
# Existing diagnostics (requires their original result paths)
python fig/generate_all.py --suite legacy

# Layout and export QA only. Every output is visibly marked DEMO and reads
# deterministic synthetic fixtures under fig/_demo/; never cite these values.
python fig/generate_all.py --suite paper --demo
```

The generated files and `<figure-name>.qa-report.json` stay beside each plotting script and
are ignored by Git because they can be reproduced from the result artifacts.

## Formal paper figures

Select the qualitative sequence before looking at its rendered prediction
trace (for example, from a protocol-defined participant/session list), then run:

```bash
python fig/generate_all.py --suite paper \
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
