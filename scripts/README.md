# Script entry points

> The index includes historical preparation and experiment utilities. For the
> current frozen aligned11 short-paper workflow, do **not** invoke anything in
> `data/`; use only `experiments/preflight_marlin11_shortpaper.py` and
> `experiments/run_marlin11_shortpaper.sh` as documented below.

Run commands from `code/` unless a script explicitly says otherwise. The `scripts/` root is an index only: executable entry points live in one of four responsibility directories, while reusable model, data-contract, training, and metric logic remains under `src/eptnet/`.

## `data/` — preparation, pairing, and staging

- `prepare_data.sh`: prepare the secondary single-session audit dataset.
- `prepare_bci_subjects.ps1` / `.sh`: prepare the primary subject-disjoint pilot.
- `prepare_bci_subjects_official_text.ps1` / `.sh`: build the isolated official-text candidate.
- `prepare_current_staged_sessions.ps1` / `.sh`: checksum-verify the current staged raw
  tree, quarantine unannotated video-only directories, preserve stable session IDs, and
  build the versioned 29-session model-ready dataset without rewriting raw data.
- `prepare_current_no_facial_sessions.ps1` / `.sh`: build the versioned 29-session
  no-facial dataset, exclude all facial CSV records from processed provenance and
  tensors, and write a coarse multimodal alignment report without compatibility windows.
- `prepare_current_absolute_time_no_facial_sessions.ps1` / `.sh`: build the current
  historical v5 no-facial dataset by intersecting transcript bins with OpenBCI row epochs and
  PPG acquisition-start clocks; non-overlap and unanchored streams fail closed.
- `prepare_current_session_registered_no_facial_sessions.ps1` / `.sh`: build the
  current v6 no-facial dataset. It first uses direct clock intersection and, only for
  a confirmed same-session EEG pair with no direct overlap, applies one documented
  session-end constant offset without stretching the signal.
- `prepare_current_session_registered_behavior12.ps1`: retain only the 12 sessions
  with paired video, audio, and annotated text, then extract OpenFace 2.2.0, WavLM
  Base+, and MacBERT features on the registered session timeline.
- `prepare_current_session_registered_marlin12.ps1`: retain the same frozen 12-session
  cohort and extract MARLIN ViT-Small (GPU), WavLM Base+, and MacBERT into an isolated
  dataset with interruption-safe MARLIN caches.
- `prepare_current_session_registered_marlin12_4060.ps1`: RTX 4060-only wrapper with
  batch size 4 and separate processed/cache paths; it cannot write into the 4090 route.
- `watch_behavior12_progress.ps1`: show a live PowerShell progress bar for cached
  OpenFace videos and the later CUDA encoding/dataset-writing stages.
- `watch_marlin12_progress.ps1`: show the MARLIN cache count and the later all-modal
  preparation stages without modifying the running job.
- `watch_marlin12_4060_progress.ps1`: monitor only the isolated RTX 4060 dataset and cache.
- `package_processed_dataset.py`: verify every complete-session tensor by restricted
  deserialization and SHA-256, reject links/raw-source leakage/non-portable references,
  write `FILES.sha256`, and create a deterministic downloadable ZIP beside the dataset.
- `assemble_time_paired_raw_dataset.ps1`, `collect_sensor_texts.ps1`, `extract_bci_docx_text.py`, and `inventory_927_videos.py`: local source inventory and pairing utilities.
- `stage_927_videos.py` and `validate_927_video_staging.py`: stage and validate the 927-video augmentation. These commands operate only on explicitly supplied paths.
- `install_behavior_encoder_assets.py`: install revision-pinned MARLIN-small, WavLM
  Base+, and MacBERT weights plus the official OpenFace 2.2.0 Windows runtime/model files under
  the corresponding `src/eptnet/models/behavior/<modality>/` directories. The command
  verifies SHA-256 values and does not start training.
- `build_ordered_session_dataset.ps1` and `build_three_session_datasets.ps1`:
  deterministic raw-session views retained for data curation; defaults are resolved
  from the repository instead of a machine-specific drive.

Private inputs and generated datasets stay outside public release artifacts.

## `experiments/` — training and evaluation orchestration

- `train.ps1` / `.sh` and `evaluate.ps1` / `.sh`: single-config entry points.
  Training shows nested epoch/train/validation progress bars by default. On Windows,
  `train.ps1` uses `Start-Transcript` so the native progress display is not broken by
  a pipeline; pass `-NoProgress` (or Python `--no-progress`) for CI/plain logs.
  `train.ps1 -Smoke` runs one epoch on one complete real train session plus one
  complete real validation session, writes a hardware-specific runtime estimate, and
  never reads test data. It is an engineering gate, not an experiment result.
- `run_bci_subjects_formal.ps1` / `.sh`: historical fixed-split matrix outside
  the current short-paper evidence scope.
- `run_formal_experiments.ps1` / `.sh`: secondary audit matrix.
- `run_remaining_formal.ps1`: resumable secondary matrix; accepts `-Python` and `-Device` and contains no machine-specific interpreter path.
- `ablation.sh`: legacy ablation launcher; verify the intended protocol before use.
- `preflight_marlin11_shortpaper.py`: read-only gate for the frozen aligned11
  cohort, manifests, label direction, EEG/PPG/video modality contract, isolated
  output roots, and metadata fingerprints.
- `run_marlin11_shortpaper.sh`: current additive short-paper entry point. It
  never calls data preparation; it provides dry-run, strict preflight, main,
  modality, diagnostic, skip-complete, and trusted partial-resume modes.
- `verify_marlin11_shortpaper_results.py`: read-only validator used before any
  completed seed or aggregate is reused; it checks identity, label protocol,
  provenance, predictions, decoded events, and exact aggregate contents.
- `run_marlin11_server.sh`: historical six-input aligned11 launcher retained for
  provenance only; it is not the current EEG+PPG+video short-paper workflow.

## `server/` — deployment gates

- `verify_marlin11_gpu.sh`: historical six-input CUDA/cohort gate retained for
  its matching launcher. Current short-paper runs use
  `experiments/preflight_marlin11_shortpaper.py` instead.
- `watch_training_progress.sh`: live main/GRU epoch progress, latest losses,
  active process, and RTX GPU utilization/memory dashboard.

## `audit/` — read-only gates

- `audit_bci_subjects.py` and `audit_bci_session_compliance.py`: dataset and session contract gates.
- `audit_parameter_fairness.py`: executed-parameter fairness gate. New reports
  include config, implementation-source, and training-manifest fingerprints;
  `--check-reference REPORT.json` validates them read-only before reuse, and
  the direct `--output` path refuses to overwrite an existing report.
- `verify_behavior_migration.py` and `verify_pretrained_extractors.py`: isolated upstream-feature checks.
- `verify_behavior_encoder_baselines.py`: bounded real/synthetic extractor and shared
  causal-TCN verification for OpenFace, WavLM Base+, and MacBERT; it writes
  `results/behavior_encoder_baseline_smoke.json` and never starts a full experiment.

## `reporting/` — deterministic release artifacts

- `build_release_report.py`: validate the frozen primary matrix and write deterministic release outputs under `results/`.
- `gen_fig_experiments.py`: validate the same matrix and write deterministic publication artifacts under `figures/`.
- `gen_fig_training.py`: regenerate vector PDF, 300 DPI PNG, CSV, and a checksum
  manifest from one run's `history.json`. Formal training invokes the same generator
  automatically after the final or early-stopped epoch.
- `export_session_modality_xlsx.py`: export the raw-session modality inventory to
  `data/reports/` using repository-relative paths.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts/reporting/build_release_report.py
python scripts/reporting/gen_fig_experiments.py --check-only
python scripts/reporting/gen_fig_experiments.py --verify-determinism
python scripts/reporting/gen_fig_training.py --run-dir results/<experiment>/seed_42
```

No compatibility wrappers are retained at `scripts/` root. This keeps every command's ownership visible and prevents a second flat launcher layer from drifting out of sync.

Current paper-figure specifications and independently reproducible plotting
scripts live under `fig/` so their PDF/SVG/500-dpi PNG outputs and QA reports
remain colocated, as documented in `README_SERVER.md`. Historical frozen-matrix
release reporting remains in this directory.

Generated data, metrics, figures, and logs must never be used as locations for executable source. See `../../docs/repository_layout.md` for the complete boundary policy.
