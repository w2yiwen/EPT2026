# Experiment Design — Frozen aligned11 short paper

> Updated: 2026-09-30 | Status: design only; no new experiment has been run or reported here.
> Extended: frozen-evidence and claim-gating sections are project-specific.

## 1. Research objective

This study evaluates multimodal BCI **temporal recognition and localization**.
Given causal EEG, PPG-derived physiology, and video evidence, the system
estimates a time-varying `P(y_t = 0)` and predicts the start and end of target-event
intervals. Repository labels are immutable: `0 = deception`, `1 = truth`, and
class `0` is the positive event. The unit of inference is a target-valid time
step within a complete chronological session, not one binary session label.

The executable evidence uses the frozen 11-session artifact
`bci_subjects_ept_v6_aligned11`. Audio and text are excluded from
the paper-facing experiments. EPT-Net is the reference realization; the task,
dataset protocol, and multimodal temporal evidence are the paper's broader
focus.

## 2. Frozen data protocol

| Item | Contract |
|---|---|
| Dataset | `bci_subjects_ept_v6_aligned11` |
| Included IDs | `session_002`, `session_003`, `session_004`, `session_006`, `session_008`, `session_009`, `session_010`, `session_012`, `session_015`, `session_017`, `session_018` |
| Excluded ID | `session_011` |
| Cohort policy | `sessions_all.jsonl` trains on all 11 aligned sessions; existing validation/test manifests remain fixed in-training evaluation views |
| Inputs | EEG time, EEG spectrum, PPG-derived physiology, video |
| Disabled inputs | Audio and text |
| Labels | `0 = deception`, `1 = truth`; positive event is class `0` |
| Permitted writes | New isolated experiment results and ignored figure outputs only |

Processed tensors, manifests, IDs, splits, temporal alignment, features,
labels, and existing results must never be rebuilt, modified, moved, merged, or
overwritten. All experiments below read the same frozen manifests.

## 3. Research questions and evidence

### RQ1 — Dynamic recognition and localization

Can a causal multimodal model rank target-event time steps, assign useful
probabilities, and localize target-event intervals on the fixed
training-included evaluation sessions?

> Evidence: session-macro frame AP, Brier score, negative log likelihood,
> boundary F1, event F1 at IoU 0.5, and event mAP, plus pooled frame AUPRC,
> early-detection recall, and detection delay. No single metric is treated as
> sufficient.

### RQ2 — Model comparison

Does EPT-Net improve temporal probability and localization outcomes relative
to LSTR, causal GateHUB, and TeSTra under the same inputs, manifests, losses,
decoder, and evaluation code?

> Evidence: the four-model main matrix with seed 42 for every run. The
> parameter audit is a prerequisite, not a performance result.

### RQ3 — Read/update mechanism

Are adaptive temporal reading and persistent state update useful components of
the EPT-Net realization?

> Evidence: targeted fixed-reader and no-persistent-state diagnostics. These
> diagnose model behavior and do not establish a physiological mechanism.

## 4. Experiment matrix

### 4.1 Main comparison

| Model | Configuration | Seeds | Difference and purpose |
|---|---|---|---|
| EPT-Net | `configs/eptnet_marlin11_eeg_ppg_video.yaml` | 42 | Adaptive branch-specific temporal reading plus persistent multimodal state |
| LSTR | `configs/lstr_marlin11_eeg_ppg_video.yaml` | 42 | Learned-query long-memory compression with causal short-memory decoding |
| GateHUB | `configs/gatehub_marlin11_eeg_ppg_video.yaml` | 42 | Gated history compression and causal present decoding; FaH is disabled |
| TeSTra | `configs/testra_marlin11_eeg_ppg_video.yaml` | 42 | LSTR-style long/short memory with exponential temporal smoothing |

**Why this design:** it holds the artifact, all-session training policy, modalities,
training/evaluation interfaces, and event decoder constant while comparing
four temporal fusion strategies. `audit_parameter_fairness.py` records the
executed parameter counts for the exact configs before training.

### 4.2 Video-only input diagnostic

| Variant | Configuration | Default seed | Question |
|---|---|---|---|
| Video | `configs/eptnet_marlin11_video_only.yaml` | 42 | What temporal evidence is available without physiology? |

**Why this design:** this run is a single-seed diagnostic of model behavior
with behavioral video alone.

### 4.3 Targeted mechanism diagnostics

| Variant | Configuration | Default seed | Question |
|---|---|---|---|
| Fixed reader | `configs/eptnet_marlin11_fixed_reader.yaml` | 42 | Is adaptive selection of temporal context useful? |
| No persistent state | `configs/eptnet_marlin11_no_persistent.yaml` | 42 | Is state persistence useful beyond current fused evidence? |

**Why this design:** each variant removes one named mechanism while preserving
the frozen EEG+PPG+video input contract. Single-seed outcomes are diagnostic
and must be labelled as such.

## 5. Evaluation protocol

### 5.1 Frame and probability outcomes

- Average precision for ranking class-0 target steps, available both pooled and
  session-macro (stored under the evaluator's `subject_macro` compatibility
  key); AUPRC is currently a pooled supporting field.
- Brier score for squared error of `P(y_t = 0)`.
- Binary negative log likelihood with the clipping convention recorded in the
  evaluation artifact.
- Macro-F1, balanced accuracy, and AUROC as supporting discrimination metrics.

Probability quality is reported alongside discrimination. Brier score or NLL
alone does not justify the phrase "calibrated probability"; any such claim
would require an explicit calibration analysis.

### 5.2 Boundary, event, and latency outcomes

- Boundary macro-F1 under the configured boundary tolerance.
- Event F1 at IoU 0.5 from the validation-selected operating point.
- Event AP at IoU 0.3, 0.5, and 0.7 and their mean, computed from ranked dense
  proposals independently of the frame threshold.
- Mean detection delay and early-detection recall as causal decoder outcomes.

Threshold selection uses the fixed validation view and is frozen before the
fixed test view. Fixed-threshold evaluation is sensitivity analysis, not an
alternative chosen after seeing test results.

### 5.3 Aggregation and uncertainty

- Use available session-macro AP, Brier, NLL, boundary F1, event F1, and
  event mAP as primary summaries.
- Keep pooled time-step metrics secondary because session length differs.
  AUPRC, early-detection recall, and detection delay currently have no
  session-macro field and must be labelled pooled.
- Report the seed attached to every run and never present unmatched default
  seeds as a seed-matched uncertainty analysis.
- Do not convert within-cohort variation into a population uncertainty statement.

## 6. Execution order and fail-closed gates

```text
read-only config/data preflight
  -> exact cohort, manifests, label direction, modalities, output roots
  -> main-model parameter fairness audit
  -> sequential train/evaluate per declared model and seed
  -> aggregate measured seed artifacts
  -> render figures from matching measured artifacts
  -> manual visual QA and claim audit
```

The executable entry point is
`scripts/experiments/run_marlin11_shortpaper.sh`. It refuses existing or
partial run directories; `--skip-existing` recognizes only completed runs with
metrics, predictions, and decoded events, while `--resume-partial` requires both
`last.pt` and
`best.pt` in an incomplete run with no final metrics. It never invokes a
data-preparation script. An existing parameter-fairness report is reusable only
after its read-only reference check matches the current config, implementation,
and training-manifest fingerprints.

## 7. Planned paper evidence

| Artifact | Source | Allowed interpretation |
|---|---|---|
| Main result table | Three main configs with declared seeds | Protocol-matched model comparison on frozen aligned11 |
| Compact diagnostic rows | Fixed reader, no persistent state | Component-level evidence, explicitly diagnostic |
| Dynamic trace figure | Predeclared evaluation-view sample; matching predictions/events/metrics | Example of temporal probability, decoded intervals, and reader behavior |

## 8. Evidence-linked language

All quantitative statements remain tied to the frozen protocol. Use `causal`
for the current inference path, `probability quality` for Brier/NLL evidence,
and `model-behavior diagnostic` for reader traces. Use stronger language about
deployment, calibration, physiological mechanism, or external generalization
only when the corresponding measured evidence exists.

## 9. Result status

No performance value, ranking, improvement, confidence interval, or runtime is
reported in this document. Tables and figures must remain empty of measured
claims until the declared runs finish and their provenance and protocol fields
pass review.
