# Development Log — Frozen aligned11 short-paper workflow

> Created: 2026-09-30 | Last updated: 2026-09-30
> Experimental status: no new performance number is recorded in this log.

## Project overview

| Item | Value |
|---|---|
| Research task | Multimodal BCI temporal recognition and localization |
| Dynamic target | `P(y_t = 0)`, where `0 = deception` and `1 = truth` |
| Paper inputs | EEG + PPG-derived physiology + video |
| Frozen artifact | `bci_subjects_ept_v6_marlin4060_aligned11` |
| Included scope | 11 existing participants; `session_011` excluded |
| Implementation strategy | Additive configs, metrics, orchestration, figures, and docs around immutable data |
| Prohibited work | Rebuild/modify tensors, manifests, IDs, splits, alignment, features, labels, or existing results |

## Implementation progress

Status meanings: `present` means the implementation exists in the worktree;
`pending evidence` means no experimental outcome is asserted here.

| Area | Files | Status | Notes |
|---|---|---|---|
| Frozen requirements | `docs/user_requirements.md` | present | Canonical scope and safety boundary |
| Main/diagnostic configs | `configs/*marlin11*.yaml` | present | New output roots; audio/text disabled |
| Read-only preflight | `scripts/experiments/preflight_marlin11_shortpaper.py` | present | Exact cohort/config/manifest gate |
| Safe orchestration | `scripts/experiments/run_marlin11_shortpaper.sh` | present | Refuse, skip-complete, and trusted-resume modes |
| Probability metrics | `src/eptnet/metrics.py`, `src/eptnet/evaluate.py` | present | Brier and NLL added beside discrimination/localization |
| Formal figure paths | `fig/fig04_dynamic_tracking/`, `fig/fig05_modality_evidence/` | present | Measured and clearly labelled demo paths separated |
| Runbooks and contracts | `README.md`, `README_SERVER.md`, `REPRODUCIBILITY.md`, `docs/*.md` | present | Local/server workflow aligned to frozen artifact |
| New GPU experiments | `results/` | pending evidence | No result or performance claim added by this documentation task |

## Development entries

### 2026-09-30 — Freeze the executable evidence boundary

- Set `bci_subjects_ept_v6_marlin4060_aligned11` as the only current
  executable/evaluated artifact.
- Recorded the exact included IDs and the continued exclusion of
  `session_011`.
- Fixed the paper-facing modalities to EEG, PPG-derived physiology, and video.
- Preserved repository label semantics and class-0-positive evaluation.

### 2026-09-30 — Add a short-paper experiment surface

- Added distinct EPT-Net, GRU, and Transformer configs for the main comparison.
- Added video-only, fixed-reader, and no-persistent-state diagnostic configs.
- Added a read-only preflight and a sequential runner that never invokes data
  preparation.
- Added explicit handling for new, completed, and checkpointed partial result
  directories without silent overwrite.

### 2026-09-30 — Extend evaluation and figure contracts

- Added probability-quality fields alongside frame discrimination, boundary,
  event, and latency outputs.
- Kept thresholded event F1 separate from threshold-free proposal AP/mAP.
- Added a formal dynamic-tracking figure path plus a visibly labelled
  synthetic demo fixture for layout QA.

### 2026-09-30 — Rewrite documentation

- Replaced stale primary-route guidance with local and server runbooks for the
  frozen aligned11 workflow.
- Documented additive experiment, aggregation, safe-resume, and figure steps.
- Added experiment design and implementation contracts without inserting
  unrun numerical results.

### 2026-09-30 — Documentation-side validation

- `git diff --check` completed without whitespace errors.
- `bash -n scripts/experiments/run_marlin11_shortpaper.sh` completed without a
  shell syntax error.
- The all-suite `--dry-run` validated all six additive configs and printed
  the declared commands. It returned `passed_with_warnings` because the private
  frozen artifact is absent locally; no training, evaluation, aggregation, or
  strict data verification was performed.

### 2026-09-30 — Local implementation verification

- Python 3.12 test run: `288 passed, 9 skipped`; the skips are optional/external
  integration checks, and no GPU experiment was executed.
- `python fig/generate_all.py --suite paper --demo` passed export and contract
  QA for the dynamic figure (PDF, SVG, 500-dpi PNG, and QA report).
- Manual preview confirmed that every demo output is visibly marked synthetic.

## Decisions

- A missing local artifact is not repaired or reconstructed; dry-run may check
  commands, while strict preflight must fail until authorized data are present.
- A complete seed may be reused only when metrics, predictions, and decoded
  events all exist.
- A partial seed may resume only when it has no final metrics and retains both
  its own `last.pt` and `best.pt`; otherwise execution stops for manual review.
- Video-only and mechanism runs are diagnostics.
- Demo figure values are never scientific evidence.

## Pending evidence

- Authorized-artifact preflight and GPU execution remain pending because the
  private frozen artifact is not present in this local checkout.
- Quantitative tables and captions remain pending authorized runs and
  provenance review.
- Paper claims remain tied to the frozen participant-disjoint protocol.
- Reader traces are reported as model-behavior diagnostics.
- The current workflow evaluates causal inference from offline features;
  end-to-end deployment language awaits feature-extraction latency evidence.
