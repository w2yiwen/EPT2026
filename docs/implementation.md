# Implementation Guide — Frozen aligned11 short-paper workflow

> Updated: 2026-09-30 | Strategy: additive extension of the existing EPT-Net repository
> Status: implementation present; GPU experiments not asserted by this document.
> Extended: immutable-data and safe-resume contracts are project-specific.

## 1. Scope and invariants

The implementation reads the existing
`bci_subjects_ept_v6_marlin4060_aligned11` artifact and adds configurations,
preflight validation, probability metrics, orchestration, aggregation, and
paper figures. It must not create or modify data-preparation code paths as part
of this workflow.

Hard invariants:

1. The exact 11 participant IDs and existing train/validation/test manifests
   remain unchanged; `session_011` remains excluded.
2. Processed tensors, temporal alignment, features, labels, masks, and existing
   results are read-only.
3. `0 = deception`, `1 = truth`, and evaluation uses `positive_class: 0`.
4. Paper inputs are EEG time/spectrum, PPG-derived physiology (`use_hr` in
   legacy config keys), and video; audio and text are disabled.
5. Every new experiment has a distinct `_no_text` output root, and no command
   silently overwrites an existing seed directory.

## 2. Additive project structure

```text
EPT2026/
├── configs/
│   ├── eptnet_marlin11_eeg_ppg_video.yaml
│   ├── gru_marlin11_eeg_ppg_video.yaml
│   ├── transformer_marlin11_eeg_ppg_video.yaml
│   ├── eptnet_marlin11_video_only.yaml
│   ├── eptnet_marlin11_fixed_reader.yaml
│   └── eptnet_marlin11_no_persistent.yaml
├── scripts/experiments/
│   ├── preflight_marlin11_shortpaper.py
│   ├── run_marlin11_shortpaper.sh
│   └── verify_marlin11_shortpaper_results.py
├── src/eptnet/
│   ├── metrics.py
│   ├── evaluate.py
│   └── aggregate.py
├── fig/
│   ├── generate_all.py
│   ├── FIGURE_CONTRACTS.md
│   ├── fig04_dynamic_tracking/
│   └── fig05_modality_evidence/
├── docs/
│   ├── user_requirements.md
│   ├── experiment_design.md
│   ├── implementation.md
│   └── dev_log.md
├── README.md
├── README_SERVER.md
└── REPRODUCIBILITY.md
```

## 3. File responsibilities

| File | Responsibility | Reads | Writes |
|---|---|---|---|
| `configs/eptnet_marlin11_eeg_ppg_video.yaml` | Full paper-facing EPT-Net input selection | Frozen base config | No files |
| `configs/gru_marlin11_eeg_ppg_video.yaml` | Protocol-matched GRU input/output contract | Frozen GRU config | No files |
| `configs/transformer_marlin11_eeg_ppg_video.yaml` | Fusion-Transformer comparison | Full paper config | No files |
| `configs/eptnet_marlin11_video_only.yaml` | Video-only input diagnostic | Full paper config | No files |
| `configs/eptnet_marlin11_{fixed_reader,no_persistent}.yaml` | Targeted mechanism diagnostics | Full paper config | No files |
| `scripts/experiments/preflight_marlin11_shortpaper.py` | Validate and fingerprint the frozen contract | Configs, manifests, metadata | Optional additive preflight JSON |
| `scripts/experiments/run_marlin11_shortpaper.sh` | Sequential preflight/train/evaluate/aggregate orchestration | Frozen data and selected configs | New result/log artifacts only |
| `scripts/experiments/verify_marlin11_shortpaper_results.py` | Validate completed-run and aggregate identity before reuse | Existing short-paper outputs | Read-only pass/fail report |
| `src/eptnet/metrics.py` | Frame probability/discrimination and event metrics | Arrays from evaluation | Metric dictionaries |
| `src/eptnet/evaluate.py` | Validation-only thresholding, held-out evaluation, artifact export | Checkpoint and frozen manifests | New metrics/predictions/events |
| `src/eptnet/aggregate.py` | Protocol-checked aggregation across seeds | Completed metrics JSON | New aggregate JSON/CSV |
| `fig/generate_all.py` | Dispatch legacy, formal-paper, or demo figures | Measured artifacts or labelled demo fixtures | Ignored figure exports/QA |

## 4. Frozen data flow

```text
immutable train/val/test manifests
  -> existing complete-session dataset loader
  -> existing masks and precomputed EEG/PPG/video tensors
  -> selected temporal model
  -> per-step logits, boundary scores, offsets, reader/state diagnostics
  -> validation-only frame threshold selection
  -> held-out complete-session evaluation
  -> new metrics, predictions, decoded events
  -> protocol-checked aggregation and figures
```

No shape, sampling, label, normalization, alignment, or feature change is
introduced at the data boundary. Configuration inheritance keeps the existing
tensor contract; the additive configs only select stored branches and isolated
output identities.

## 5. Preflight implementation

### `validate_configs(paths, *, allow_missing_data, device) -> dict`

Located in `scripts/experiments/preflight_marlin11_shortpaper.py`.

- Resolves and strictly loads each allowed short-paper config.
- Requires the exact dataset name and three frozen manifest paths.
- Requires the exact included/excluded participant lists.
- Requires `num_classes == 2`, `positive_class == 0`, aligned-behavior gates,
  disabled audio/text, and the declared model/modality contract.
- Requires unique isolated result roots ending in `_no_text`.
- When data are present, verifies non-empty manifests, no duplicate/cross-split
  IDs, exact cohort membership, no excluded ID, the success marker, and the
  dataset-summary split contract.
- Recomputes every manifest-referenced session tensor's stored SHA-256 and
  hashes manifest/metadata files without writing below `data/`.
- Resolves the requested device unless `--skip-device-check` is explicit.
- With `--reference`, compares the current dataset/label/cohort/manifest and
  metadata fingerprint fields with an existing preflight JSON and does not
  rewrite that audit record.

The first preflight records the verified artifact as its reference; later
recovery runs must match it before any resume or reuse operation.
Byte-level authenticity is an upstream distribution/read-only-storage
responsibility.

`--allow-missing-data --skip-device-check` is reserved for dry-run config
validation. Its warning-bearing result is not equivalent to a strict preflight.

## 6. Metric and evaluation implementation

### `frame_metrics(probabilities, labels, mask, threshold=0.5, positive_class=0) -> dict`

The function receives the probability assigned to the designated positive
class and a same-shaped label/mask array. With `positive_class=0`, it forms the
binary target as `labels == 0`. It validates shapes, finite values, probability
bounds, and threshold bounds, then returns confusion counts, macro-F1,
balanced accuracy, AUROC, AUPRC, average precision, Brier score, and binary
negative log likelihood. Ranking metrics are `None` for a one-class valid set;
probability scores remain defined when valid targets exist.

### `event_metrics(predictions, targets, thresholds, *, ap_proposals=None) -> dict`

Thresholded decoder events determine operating-point precision, recall, F1,
overlap, and delay. A separate dense proposal list determines event AP/mAP, so
threshold-free ranking evidence is not conflated with the selected operating
point.

### `evaluate_sequences(sequences, config, frame_threshold, efficiency=None)`

This function computes pooled frame, boundary, event, early-detection, and
latency outputs. Its participant-macro block contains frame macro-F1, balanced
accuracy, AUROC, AP, Brier, and NLL; boundary macro-F1; event F1 at IoU 0.5;
and event mAP. AUPRC, early-detection, and latency do not currently have
participant-macro fields. The function records label encoding, threshold
provenance, mask definition, probability-scoring convention, and proposal
protocol in `test_metrics.json`. `_write_predictions` and `_write_events`
export trace artifacts used by formal figures. Probability fields are
`frame.brier_score`, `frame.negative_log_likelihood`,
`subject_macro.metrics.frame_brier_score`, and
`subject_macro.metrics.frame_negative_log_likelihood`.

## 7. Orchestration and recovery contract

`scripts/experiments/run_marlin11_shortpaper.sh` accepts:

```text
--suite all|main|modalities|diagnostics
--seeds "13 42 73"
--analysis-seeds "42"
--device DEVICE
--python PATH
--dry-run
--preflight-only
--skip-existing
--resume-partial
```

Execution order:

1. Run the read-only frozen-data/config preflight.
2. For the main suite, run the exact-config parameter fairness audit.
3. Train and evaluate one model/seed at a time.
4. Aggregate only compatible completed metric files.

Existing-directory behavior is explicit:

- default: refuse any existing seed directory;
- `--skip-existing`: reuse only a directory containing metrics, predictions,
  and decoded events;
- `--resume-partial`: resume only an incomplete directory with no final metrics
  and containing both `last.pt` and `best.pt`; pass `last.pt` to
  `eptnet.train --resume` while retaining `best.pt` for prior best-model
  selection;
- both flags: skip completed seeds and resume eligible interrupted seeds;
- any other partial state: fail without deletion or overwrite.

Completed-run and aggregate reuse is checked by
`scripts/experiments/verify_marlin11_shortpaper_results.py`. The read-only
verifier binds a run to its canonical training-config fingerprint,
experiment/seed, class-0-positive protocol, frozen preflight provenance,
prediction/event schema, and exact aggregate contents; drift fails closed.

The preflight and parameter-fairness files are evidence, not existence
sentinels. Recovery modes revalidate the former through `--reference` and the
latter through the audit script's read-only `--check-reference`. Fairness reuse
requires exact resolved configs, all inherited config-file hashes, the relevant
audit/model implementation-source bundle, and the training-manifest hash;
missing legacy fingerprints or any drift fail closed.

## 8. Result artifact contract

Each new experiment root is
`results/<short-paper-experiment>/`, with one `seed_<N>/` per run.

| Artifact | Format | Meaning |
|---|---|---|
| `resolved_config.yaml` | YAML | Fully inherited configuration used by the run |
| `run_metadata.json` | JSON | Device, source/data provenance, and run identity |
| `history.json` | JSON | Training/validation history |
| `best.pt` | PyTorch checkpoint | Validation-selected model state |
| `last.pt` | PyTorch checkpoint | Trusted interruption-resume state |
| `test_metrics.json` | JSON | Protocol plus pooled and participant-macro metrics |
| `test_predictions.jsonl` | JSONL | Per-step probabilities, masks, timing, boundaries, and reader traces |
| `test_events.json` | JSON | Decoded predictions and reference intervals |
| `aggregate.json` | JSON | Provenance-checked per-seed summary |
| `aggregate.csv` | CSV | Flat aggregate export |

`aggregate_result_files(paths) -> dict` rejects mixed experiment IDs,
configuration fingerprints, prepared-data provenance, duplicate seeds, or
evaluation signatures before calculating summaries.

## 9. Figure implementation

`fig/generate_all.py --suite paper` dispatches:

- Fig. 4: dynamic `P(y_t = 0)`, reference/predicted intervals, boundary
  evidence, and EEG-time/EEG-spectrum/PPG reader traces for a predeclared
  sample.
- Fig. 5: exploratory Full-versus-Video AP, Brier, and Event mAP comparison;
  unmatched seeds are allowed but displayed using true aggregate metadata.

The dynamic renderer checks required fields, probability bounds, sample ID,
threshold presence, and positional sequence alignment. It does not compare a
run/provenance fingerprint across predictions, events, and metrics. Same-run
identity for the three dynamic-trace inputs must therefore be established
upstream. `--demo`
uses labelled synthetic fixtures only. PDF, SVG, and 500-dpi PNG are rendered
from one canvas, and QA records source hashes and dimensions.

## 10. Implementation order and verification state

```text
frozen user requirements
  -> additive configs
  -> read-only preflight
  -> probability/evaluation fields
  -> safe sequential runner
  -> deterministic figure contracts
  -> local/server/reproducibility documentation
  -> documentation checks and command dry-run validation
  -> automated code tests (reported separately by the implementation task)
  -> authorized GPU execution (pending)
```

Documentation-level consistency check:

- ✅ Experiment coverage: main comparison, input-family analysis, and two
  mechanism diagnostics each map to a config and runner suite.
- ✅ Logical consistency: class `0` is used consistently for probability,
  decoding, and reporting; all paper configs inherit the same frozen manifests.
- ✅ Completeness: every additive config, gate, runner, result type, and formal
  figure path has an implementation section above.

These checks assess the implementation plan and file interfaces. They do not
claim that a GPU experiment or new numerical result has been produced.
