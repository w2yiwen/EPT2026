# Reproducibility and artifact policy

## Evidence hierarchy

The current-source primary experiment is the 18-subject `bci_subjects_ept_v1` pilot. It uses a frozen 12/2/4 train/validation/test split with no subject overlap. Deterministic exhaustive constrained stratification balances pre-model label and modality summaries; it is performed before training and never revised from model outcomes.

The older `bci_truth_deception_v1` one-participant/one-session experiment is a secondary engineering audit tied to its recorded source-tree and prepared-data fingerprints. Its chronological rows are not statistically independent subjects, and its metrics must never be merged with the primary cohort.

## Trusted temporal and masking protocol

Every experiment unit is a complete, de-duplicated chronological subject/session. Window files are storage shards, not independently shuffled samples. Each source row contributes once per epoch and once during evaluation.

For the primary cohort, `sequence_mask` identifies real timeline bins and `target_mask` identifies bins valid for target-person decisions. Interviewer/background bins remain causal context but are excluded from losses, automatic weights, threshold selection, decoding decisions, and metrics. An invalid target bin breaks event continuity. Missing EEG, physiology, and behavior streams are represented by explicit availability masks.

## Participant-balanced model selection

The continuous-session loader uses batch size one and visits every subject exactly once per epoch. Validation loss is an unweighted mean of the complete-subject losses, and the best checkpoint/early-stopping rule is minimum participant-mean validation total loss. Target-step-weighted losses are diagnostics only.

The frame threshold is selected exclusively on the two validation subjects by maximizing mean subject-level macro-F1 over the declared grid. It is then frozen for the four held-out test subjects. A second evaluation at threshold 0.5 is reported as a sensitivity analysis and is never substituted post hoc for the selected-threshold result.

Event precision/recall/F1, early detection, and latency are operating-point metrics from the causal decoder and therefore use the selected frame threshold. Event AP/mAP is computed independently from a dense ranked proposal set: each target-valid step contributes one proposal scored by its positive-class probability, its interval is formed from predicted left/right offsets and clamped to the contiguous target-valid segment, and neither score filtering nor NMS is applied. The release gate requires event AP at every IoU and mAP to be identical in the selected-threshold and fixed-0.5 evaluations.

Primary reporting uses subject-macro frame, boundary, and event metrics. Each model seed includes a fixed-seed, 10,000-resample percentile bootstrap over held-out subjects; pooled metrics are secondary. Because the test set contains only four subjects, these intervals are descriptive and unstable. Formal comparisons use model seeds `13`, `42`, and `73`; their standard deviation quantifies optimization variability, not population uncertainty.

## Environment

Install a CUDA-enabled PyTorch build compatible with the local driver, followed by the locked CPU-side dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

The local formal runs use an NVIDIA GeForce RTX 4060 Laptop GPU with PyTorch 2.7.1+cu118. Exact package, device, determinism, resolved-configuration, source-tree, and prepared-data fingerprints are written into machine-readable run artifacts.

Behavior extraction additionally requires `requirements-behavior-lock.txt` and NumPy
`>=1.26,<2`. Do not use an environment that mixes NumPy 2 with extensions compiled
against NumPy 1; this workstation's base Anaconda pyarrow/scikit-learn stack fails
that ABI check. MacBERT and WavLM model IDs and immutable revisions are written into
the generated feature schema.

## Verification and execution order

From `code/`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest -q
python -m compileall -q src tests
.\scripts\data\prepare_bci_subjects.ps1 -Source ..\BCI
.\scripts\experiments\run_bci_subjects_formal.ps1 -Python python -Device cuda:0
```

The isolated official-text candidate has a separate preparation and verification
path and never overwrites the original primary data:

```powershell
python -m pip install -r requirements-behavior-lock.txt
.\scripts\data\prepare_bci_subjects_official_text.ps1 -Source ..\BCI
python scripts\audit\verify_behavior_migration.py
```

The matrix runner is fail-fast and sequential; it never launches competing GPU jobs. It runs the de-identified dataset audit and parameter-fairness audit before training, then trains and evaluates every declared seed before aggregation. Existing result directories are protected against silent overwrite. Smoke-test entry points are not part of this workflow.

Source code, configurations, scripts, tests, dependencies, and package metadata are frozen before the first primary run. Any subsequent change to those inputs requires a fresh matrix rather than mixing artifacts from different source fingerprints.

## Data and release boundary

Participant media, annotations, raw/prepared tensors, checkpoints, per-step predictions, decoded event traces, subject-identity mappings, and machine-specific run metadata are excluded from the public release. The release may contain implementation, tests, configurations, aggregate/de-identified summaries, figure-generation code, and rendered figures only after the release gates pass. Reproducing the numerical results requires authorized access to the exact prepared-data fingerprint recorded by the artifacts.

The primary cohort has incomplete modalities: EEG for 5/18 subjects, paired PPG/physiology for 13/18, facial features for 17/18, and no frozen audio features. Transcript characters are uniformly interpolated inside speaker turns, and sensor timelines are duration-normalized because device clocks are inconsistent. Labels are inferred from highlighted spans belonging to the frozen target; residual non-target marks are audit-only and cannot create labels or events. Target identity is frozen before reading those marks by choosing the unique speaker with the largest total non-whitespace character count in the complete transcript; the winning share and runner-up margin are recorded, and a later 95% marked-character consistency audit fails closed without changing the speaker. This is a retrospective, label-independent conversation-structure heuristic, not supplied role metadata or online role discovery, and it still requires confirmation by the data owner. Causality claims cover the model feature stream after offline preprocessing only. These limitations preclude physical-lag claims and make missingness a possible subject-identity cue.

Passing reproducibility, provenance, packaging, and release checks establishes engineering discipline only. The fixed split, two validation subjects, four test subjects, sparse modalities, approximate alignment, and small participant bootstrap do not establish top-conference-level evidence or population generalization.
