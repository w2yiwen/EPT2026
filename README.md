# EPT-Net: multimodal BCI temporal recognition and localization

EPT-Net is an auditable implementation for **multimodal BCI temporal
recognition and localization**. At each valid target-person time step it
estimates the dynamic event probability

```text
P(y_t = 0 | causal EEG, PPG-derived physiology, video history)
```

and predicts event boundaries/intervals. This is not a session-level binary
classification task. The repository label convention is immutable:
`0 = deception` and `1 = truth`; evaluation therefore treats class `0` as the
positive event.

The current executable and reportable evidence is restricted to the frozen
11-participant artifact
`bci_subjects_ept_v6_marlin4060_aligned11`. All quantitative repository-backed
claims must remain within that scope.

## Non-negotiable data boundary

The following are frozen evidence, not build products:

- processed tensors and sample packages;
- train/validation/test manifests and participant assignment;
- participant inclusion/exclusion decisions;
- temporal alignment, feature extraction, and cached features;
- labels and the `0 = deception`, `1 = truth` mapping;
- existing checkpoints, predictions, metrics, and result folders.

The included participant IDs are:

```text
session_002  session_003  session_004  session_006
session_008  session_009  session_010  session_012
session_015  session_017  session_018
```

`session_011` remains excluded. Do not run a preparation, migration,
realignment, relabelling, resplitting, feature-extraction, or overwrite command
for this workflow. New work may only read the artifact and write to new,
isolated locations under `results/` or ignored figure-output paths. A storage-
level read-only mount is recommended.

The paper-facing input families are EEG (time and spectral branches),
PPG-derived physiology (internally retained as `hr` for backward
compatibility), and video. Audio and text may exist in the frozen artifact but
are disabled by every short-paper configuration.

## Repository map

| Path | Role |
|---|---|
| `configs/*marlin11*.yaml` | Additive configurations that inherit the frozen aligned11 contract |
| `scripts/experiments/preflight_marlin11_shortpaper.py` | Read-only cohort/config/manifest gate and metadata fingerprinting |
| `scripts/experiments/run_marlin11_shortpaper.sh` | Fail-fast experiment orchestration with isolated result roots |
| `scripts/experiments/verify_marlin11_shortpaper_results.py` | Read-only completed-run and aggregate identity/provenance verifier |
| `src/eptnet/` | Existing model, training, evaluation, aggregation, and metrics code |
| `fig/` | Deterministic figure rendering and QA; measured and demo paths are separate |
| `docs/experiment_design.md` | Claim-to-experiment plan; contains no results |
| `docs/implementation.md` | Additive implementation and artifact contracts |
| `docs/dev_log.md` | Current implementation status and unexecuted-work record |
| `REPRODUCIBILITY.md` | Frozen-evidence and reporting policy |

## Local setup

Run all commands from the repository root. Python 3.10 or newer is required.
Install a PyTorch build suitable for the local device first, then install the
locked project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

For paper figures, also install:

```bash
python -m pip install -r requirements-figures.txt
```

Figure export requires an exact Arial installation and fails rather than
silently substituting another font.

## Connect the frozen artifact

The code expects this exact path relative to the repository:

```text
data/processed/bci_subjects_ept_v6_marlin4060_aligned11/
├── _SUCCESS.json
├── dataset_summary.json
├── feature_schema.json
├── normalization_stats.npz
├── manifests/
│   ├── sessions_train.jsonl
│   ├── sessions_val.jsonl
│   └── sessions_test.jsonl
├── sessions/<included-session-id>/session_manifest.json
└── samples/
```

If the authorized artifact is stored elsewhere, expose its containing data
root at `data/` with a read-only mount or a carefully inspected symlink. Never
copy it into Git, and never point the short-paper configs at a replacement
dataset.

## Safe local validation

When the private artifact or GPU is absent, validate config inheritance and
print the exact commands without training:

```bash
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --dry-run
```

`--dry-run` intentionally permits missing data and skips the device check. It
is a command/configuration check only; it does not establish data availability
and does not produce experimental evidence.

With the authorized artifact present, run the read-only structural preflight.
CPU is sufficient for this gate:

```bash
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --device cpu \
  --preflight-only
```

The gate validates the declared dataset name, exact participant set, manifest
paths, split disjointness, exclusion of `session_011`, label direction, enabled
input families, and isolated output roots. Its first successful invocation
requires `_SUCCESS.json`, cross-checks `dataset_summary.json` against the three
session manifests, and recomputes every manifest-referenced session tensor's
stored SHA-256. It records those fingerprints as a reference; later recovery
runs compare against that record. The command writes only
`results/marlin11_shortpaper_preflight.json` plus the main-model parameter
fairness audit; it never writes below `data/`.

Because those two audit files are now protected evidence, subsequent runner
commands include `--skip-existing`. The runner revalidates the current frozen
fingerprints against the saved preflight. Before preserving the fairness audit,
it also requires an exact match for the resolved configs, every inherited config
file, the audit/model implementation source bundle, and the training manifest.

## Additive experiment workflow

The primary comparison uses EPT-Net, a protocol-matched early-fusion GRU, and a
protocol-matched fusion Transformer with seeds `13`, `42`, and `73`. Compact modality and mechanism
analyses default to seed `42`.

```bash
# Main comparison only
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite main \
  --device cuda:0 \
  --skip-existing

# EEG+PPG versus video diagnostic
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite modalities \
  --device cuda:0 \
  --skip-existing

# Fixed-reader and no-persistent-state diagnostics
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite diagnostics \
  --device cuda:0 \
  --skip-existing

# Entire declared matrix
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --device cuda:0 \
  --skip-existing
```

Each run writes to `results/<experiment>/seed_<seed>/`; each experiment also
gets `aggregate.json` and `aggregate.csv`. The runner refuses an existing or
partial run directory by default. `--skip-existing` reuses a run only after the
read-only result verifier checks its config fingerprint, protocol, provenance,
metrics, per-step predictions, and decoded events. Aggregate reuse is accepted
only after an in-memory recomputation exactly matches the existing JSON/CSV.
`--resume-partial` continues only a new
short-paper run that is incomplete, has no final metrics, and retains both its
own `last.pt` and `best.pt`; any other partial state fails. Resumption loads
`last.pt` while preserving the prior best-model selection in `best.pt`. Both
flags may be combined to skip completed seeds and explicitly resume eligible
interrupted seeds. Neither mode authorizes changing historical or frozen
result folders. When an earlier preflight record exists, these modes compare
the current frozen-evidence fingerprints with that record and preserve it
rather than rewriting it. The parameter-fairness audit is preserved only after
its read-only `--check-reference` validation succeeds; legacy reports without
input fingerprints and reports affected by config, source, or manifest drift
fail closed.

```bash
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --device cuda:0 \
  --skip-existing \
  --resume-partial
```

The evaluation output distinguishes probability quality, framewise
discrimination, boundary quality, thresholded event localization, threshold-
free event AP/mAP, and detection delay. These outputs describe a dynamic
temporal task, not a single binary decision.

For a Linux GPU server, use the deployment-oriented commands in
[`README_SERVER.md`](README_SERVER.md).

## Additive figure workflow

Layout QA may use deterministic synthetic fixtures, but every generated panel
is visibly marked `DEMO` and its values must never be reported:

```bash
python fig/generate_all.py --suite paper --demo
```

Formal dynamic tracking requires a completed evaluation and a sample ID chosen
before inspecting the rendered trace. The modality panel additionally requires
the same seed policy for Full, EEG+PPG, and Video. Extend the compact modality
runs to the main three-seed policy:

```bash
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite modalities \
  --analysis-seeds "13 42 73" \
  --device cuda:0 \
  --skip-existing
```

`--skip-existing` preserves any existing one-seed diagnostic aggregates. Create
new three-seed figure inputs at previously unused paths; do not replace those
diagnostic aggregates. If `v1` already exists, change the version before
running this block:

```bash
(
set -euo pipefail
figure_input_root=results/marlin11_shortpaper_figure_inputs_v1
if [ -e "$figure_input_root" ]; then
  echo "Figure-input directory already exists; choose a new version" >&2
  exit 1
fi
mkdir -p "$figure_input_root"

python -m eptnet.aggregate \
  results/eptnet_marlin11_physiology_only_no_text/seed_13/test_metrics.json \
  results/eptnet_marlin11_physiology_only_no_text/seed_42/test_metrics.json \
  results/eptnet_marlin11_physiology_only_no_text/seed_73/test_metrics.json \
  --output "$figure_input_root/physiology_only_3seed.json" \
  --csv "$figure_input_root/physiology_only_3seed.csv"

python -m eptnet.aggregate \
  results/eptnet_marlin11_video_only_no_text/seed_13/test_metrics.json \
  results/eptnet_marlin11_video_only_no_text/seed_42/test_metrics.json \
  results/eptnet_marlin11_video_only_no_text/seed_73/test_metrics.json \
  --output "$figure_input_root/video_only_3seed.json" \
  --csv "$figure_input_root/video_only_3seed.csv"
)
```

Then render from matching measured artifacts:

```bash
python fig/generate_all.py --suite paper \
  --predictions results/eptnet_marlin11_eeg_ppg_video_no_text/seed_42/test_predictions.jsonl \
  --events results/eptnet_marlin11_eeg_ppg_video_no_text/seed_42/test_events.json \
  --metrics results/eptnet_marlin11_eeg_ppg_video_no_text/seed_42/test_metrics.json \
  --sample-id '<predeclared-held-out-sample-id>' \
  --modality-aggregate 'Full=results/eptnet_marlin11_eeg_ppg_video_no_text/aggregate.json' \
  --modality-aggregate 'EEG+PPG=results/marlin11_shortpaper_figure_inputs_v1/physiology_only_3seed.json' \
  --modality-aggregate 'Video=results/marlin11_shortpaper_figure_inputs_v1/video_only_3seed.json'
```

Figure scripts read measured artifacts without inventing, smoothing, or
repairing values. Before rendering, independently confirm that the dynamic
files come from the same seed/run. The modality renderer requires matching
protocol, seed, and source/prepared-data provenance across its aggregates; the
dynamic renderer validates schemas but cannot authenticate its three files as
one run.
See [`fig/FIGURE_CONTRACTS.md`](fig/FIGURE_CONTRACTS.md) for input schemas,
sample-selection safeguards, and export QA.

## Reporting discipline

- Report only results generated from the exact frozen aligned11 manifests.
- Keep model-seed variability separate from participant-bootstrap uncertainty.
- Do not claim population generalization, calibrated probabilities, real-time
  deployment, multimodal complementarity, or a mechanism unless the matching
  evidence has actually been run and supports it.
- A successful preflight, smoke check, or demo figure is engineering evidence,
  not a scientific result.
- This documentation records no new experimental performance numbers.
