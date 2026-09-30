# Linux GPU server runbook: frozen aligned11

> **Current training policy:** `sessions_all.jsonl` supplies all 11 aligned
> sessions to training. The frozen validation and test manifests remain fixed
> evaluation views so the existing evaluation and figure pipeline is unchanged,
> but those sessions are not held out from training and must not be described as
> independent generalization evidence.

This runbook executes the short-paper experiment matrix on the immutable
`bci_subjects_ept_v6_marlin4060_aligned11` artifact. It reads EEG,
PPG-derived physiology, and video features to estimate dynamic
`P(y_t = 0)` and localize target-event intervals. Repository semantics are
fixed: `0 = deception`, `1 = truth`, and class `0` is the positive event.

The workflow never prepares data, extracts features, changes alignment,
relabels samples, or rebuilds a split. It fails when the exact frozen artifact
is unavailable or inconsistent.

## 1. Server paths and safety gate

Example layout:

```text
/root/EPT2026/                                      # Git checkout
/root/EPT2026/data                                 # link or read-only mount
/root/autodl-tmp/.autodl/data/                     # authorized private store
└── processed/bci_subjects_ept_v6_marlin4060_aligned11/
    ├── _SUCCESS.json
    ├── dataset_summary.json
    ├── feature_schema.json
    ├── normalization_stats.npz
    ├── manifests/sessions_all.jsonl
    ├── manifests/sessions_train.jsonl
    ├── manifests/sessions_val.jsonl
    ├── manifests/sessions_test.jsonl
    ├── sessions/<included-session-id>/session_manifest.json
    └── samples/
```

Before connecting storage, resolve both paths and inspect any existing `data`
entry. Do not replace a real directory or an unexpected link:

```bash
project_root=/root/EPT2026
private_data_root=/root/autodl-tmp/.autodl/data

cd "$project_root"
test -d "$private_data_root/processed/bci_subjects_ept_v6_marlin4060_aligned11"

if [ -e data ] || [ -L data ]; then
  ls -ld data
  readlink -f data || true
else
  ln -s "$private_data_root" data
fi

test "$(readlink -f data)" = "$(readlink -f "$private_data_root")"
```

A read-only bind mount or storage permission is preferred when available. Do
not use a preparation script as a substitute for a missing artifact. The
frozen IDs are `session_002`, `session_003`, `session_004`, `session_006`,
`session_008`, `session_009`, `session_010`, `session_012`, `session_015`,
`session_017`, and `session_018`; `session_011` must remain excluded.

## 2. Update code without disturbing evidence

Check the worktree before pulling. Stop if it contains changes you do not
understand:

```bash
cd /root/EPT2026
git status --short
git pull --ff-only origin main
```

`git pull` must never include or manage the private `data/`, `results/`, or
`logs/` trees.

## 3. Install the environment

Install a CUDA-enabled PyTorch build compatible with the server driver first.
Then install the locked dependencies and editable package:

```bash
cd /root/EPT2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m pip install -r requirements-figures.txt
python -m pip check
```

Confirm the selected CUDA device:

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print(torch.cuda.get_device_name(0))
print(torch.__version__, torch.version.cuda)
PY
```

For paper figures, install Arial through an authorized server mechanism and
verify exact resolution:

```bash
python - <<'PY'
from matplotlib import font_manager

path = font_manager.findfont("Arial", fallback_to_default=False)
assert font_manager.FontProperties(fname=path).get_name() == "Arial", path
print(path)
PY
```

## 4. Inspect commands without running them

The dry run checks all short-paper configurations, permits missing private
data, skips the GPU gate, and prints the commands it would execute:

```bash
cd /root/EPT2026
source .venv/bin/activate
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --dry-run
```

This is not a data-integrity pass and not an experiment.

## 5. Structural preflight and reference fingerprint

Run the structural gate before allocating a long GPU job:

```bash
cd /root/EPT2026
source .venv/bin/activate
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --device cuda:0 \
  --preflight-only
```

The gate validates the declared dataset name, the all-11 training cohort,
fixed evaluation views, manifest locations, excluded participant, label direction,
EEG/PPG/video selection, disabled audio/text branches, and isolated result
paths. It requires `_SUCCESS.json`, cross-checks `dataset_summary.json`, and
recomputes every manifest-referenced session tensor's stored SHA-256. On its
first pass it records these fingerprints in
`results/marlin11_shortpaper_preflight.json`. The main suite also records the
executed parameter comparison in
`results/marlin11_shortpaper_parameter_fairness.json`.

Later recovery runs compare current evidence fields with the saved preflight.
They also validate the saved fairness audit against the exact current resolved
configs, inherited config files, implementation source, and training manifest.
Any warning, legacy fingerprint omission, drift, or failure is a stop condition.
Do not regenerate the data artifact to make the gate pass.

The standalone gate creates protected preflight and fairness-audit files.
Include `--skip-existing` in subsequent runner commands so they are checked and
preserved rather than treated as overwrite targets.

## 6. Run the experiment matrix

The default full EPT-Net run uses seed `13`. GRU, Transformer, video-only, and
mechanism-diagnostic runs use seed `42`.

Foreground commands:

```bash
# Three-model main comparison
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite main \
  --eptnet-seeds "13" \
  --seeds "42" \
  --device cuda:0 \
  --skip-existing

# Video-only diagnostic
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite modalities \
  --device cuda:0 \
  --skip-existing

# Fixed reader and no persistent state
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite diagnostics \
  --device cuda:0 \
  --skip-existing
```

For a single sequential background job:

```bash
cd /root/EPT2026
source .venv/bin/activate
mkdir -p logs
nohup bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --device cuda:0 \
  --skip-existing \
  > logs/marlin11_shortpaper.log 2>&1 &
job_pid=$!
echo "$job_pid" > logs/marlin11_shortpaper.pid
echo "PID=$job_pid"
```

Monitor without modifying the run:

```bash
tail -f logs/marlin11_shortpaper.log
ps -fp "$(cat logs/marlin11_shortpaper.pid)"
nvidia-smi
```

Each run writes to an isolated
`results/<experiment>/seed_<seed>/` directory. The runner refuses to overwrite
an existing or partial directory. `--skip-existing` accepts only runs that pass
`verify_marlin11_shortpaper_results.py`: config/experiment identity,
protocol/label direction, runtime provenance, metrics, per-step predictions,
and decoded-event structure must agree. Existing aggregates are reused only
when their JSON/CSV exactly match a read-only in-memory recomputation.
`--resume-partial` is explicit permission
to continue only a new short-paper seed directory that lacks final metrics and
contains its own `last.pt` and `best.pt`. Resumption loads `last.pt` while
preserving the prior best-model selection in `best.pt`; any other partial state
remains a manual-review failure. The two flags can be combined after an
interrupted matrix:

```bash
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --device cuda:0 \
  --skip-existing \
  --resume-partial
```

This resumes the trusted checkpoint in place; it never deletes, moves, or
rebuilds the directory and must not be used on historical frozen results. If a
preflight JSON already exists, recovery modes require current frozen-evidence
fingerprints to match it and preserve the original audit file.

## 7. Expected additive outputs

The matrix writes only below new short-paper result roots:

```text
results/eptnet_marlin11_eeg_ppg_video_no_text/
results/gru_marlin11_eeg_ppg_video_no_text/
results/transformer_marlin11_eeg_ppg_video_no_text/
results/eptnet_marlin11_video_only_no_text/
results/eptnet_marlin11_fixed_reader_no_text/
results/eptnet_marlin11_no_persistent_no_text/
```

Each completed seed directory contains its resolved config, provenance,
checkpoints, training history, test metrics, per-step predictions, and decoded
events. Each experiment root receives `aggregate.json` and `aggregate.csv`.
These new artifacts do not authorize editing any earlier result folder.

## 8. Render the formal dynamic-tracking figure

Use the demo path only for layout/export QA; its values are synthetic and
visibly marked:

```bash
python fig/generate_all.py --suite paper --demo
```

Before rendering the formal trace, predeclare a fixed evaluation-view sample
without viewing candidate plots, then use matching predictions, events, and
metrics from the same full-model run:

```bash
python fig/generate_all.py --suite paper \
  --predictions results/eptnet_marlin11_eeg_ppg_video_no_text/seed_13/test_predictions.jsonl \
  --events results/eptnet_marlin11_eeg_ppg_video_no_text/seed_13/test_events.json \
  --metrics results/eptnet_marlin11_eeg_ppg_video_no_text/seed_13/test_metrics.json \
  --sample-id '<predeclared-evaluation-sample-id>' \
  --modality-aggregate 'Full=results/eptnet_marlin11_eeg_ppg_video_no_text/aggregate.json' \
  --modality-aggregate 'Video=results/eptnet_marlin11_video_only_no_text/aggregate.json'
```

The plotting suite exports PDF, SVG, and 500-dpi PNG from one canvas and writes
QA reports with source hashes. Confirm that the trace's three input files come
from the same run. The Full/Video panel permits unmatched seeds but displays
their true aggregate seed lists and is interpreted as exploratory evidence.

## 9. Failure policy

- Missing data, a changed manifest, a cohort mismatch, enabled audio/text, or
  an incompatible device means stop and investigate.
- Never repair a failure by rerunning alignment, preprocessing, feature
  extraction, participant selection, or split generation.
- Never delete, overwrite, or merge an old result directory to make room for a
  new run.
- Keep participant media, tensors, predictions, checkpoints, credentials, and
  machine-specific metadata out of Git.
- Do not report a dry run, preflight, or demo figure as an experiment.
