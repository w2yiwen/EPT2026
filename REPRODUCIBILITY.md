# Reproducibility and frozen-evidence policy

> Current user-selected cohort policy: all 11 aligned sessions in
> `sessions_all.jsonl` participate in training. The frozen validation/test
> manifests remain fixed evaluation views, but are not held out from training.
> Results from this route must not be described as independent generalization.

## Current executable scope

The sole current executable/evaluated data artifact is
`bci_subjects_ept_v6_marlin4060_aligned11`. It contains 11 included
participants: `session_002`, `session_003`, `session_004`, `session_006`,
`session_008`, `session_009`, `session_010`, `session_012`, `session_015`,
`session_017`, and `session_018`. `session_011` is excluded and must not appear
in any split.

Historical or broader data routes are not evidence for the current paper
workflow. Their checkpoints, predictions, and metrics must not be pooled,
compared as though they share a protocol, or used to expand the scope of a
claim.

## Scientific estimand

The task is causal multimodal BCI temporal recognition and localization. For
each target-valid time step, a model produces a probability for the target
event and predicts event boundaries/intervals:

```text
inputs:  EEG + PPG-derived physiology + video
output:  P(y_t = 0) + start/end evidence + decoded temporal intervals
labels:  0 = deception, 1 = truth
event:   class 0
```

The task is not reduced to one binary label per participant or session. Audio
and text features are excluded from the short-paper configurations even if
they are present in the stored artifact. The legacy internal key `use_hr`
selects the PPG-derived physiology branch; it does not redefine that modality
as an independently measured heart-rate-only input.

## Immutable evidence boundary

The following are immutable for all current runs:

- processed tensors, sample packages, and availability masks;
- participant IDs, inclusion/exclusion, and train/validation/test assignment;
- manifests and complete-session ordering;
- temporal alignment and all extracted features;
- labels, boundary targets, and label direction;
- existing results, checkpoints, predictions, and aggregate reports.

No current reproduction step may prepare, migrate, align, relabel, resplit,
extract, regenerate, repair, or overwrite any item above. Put the private
artifact on storage with read-only enforcement when possible. Repository
scripts provide fail-closed checks, but filesystem permissions remain the
strongest protection against accidental writes.

## Trusted temporal and masking protocol

Every evaluation unit is a unique complete chronological session. Causal
windows are a training protocol, not independent participants. `sequence_mask`
identifies real timeline bins; `target_mask` identifies bins eligible for
target-person losses, threshold selection, decoding decisions, and metrics.
Invalid target intervals break event continuity. Availability masks represent
missing EEG, physiology, or video observations and must never be replaced with
fabricated measurements.

Threshold selection uses the fixed validation view only. The selected threshold is then
frozen for the fixed test view. Thresholded event F1 and delay belong to that
operating point. Event AP/mAP is derived from ranked dense proposals and remains
separate from thresholded decoding.

Frame evaluation includes discrimination and probability quality. The score is
the softmax probability assigned to class `0`, and the binary target is
`1[label == 0]` on target-valid steps. Average precision/AUPRC measures ranking;
Brier score and negative log-likelihood measure probability quality. These
metrics do not by themselves establish that probabilities are calibrated.

Available participant-macro AP, Brier, NLL, boundary F1, event F1, and event
mAP are the primary summaries. AUPRC, early-detection recall, and latency are
currently emitted only as pooled fields and must be labelled accordingly.
Participant bootstrap intervals and variation across model seeds quantify
different uncertainties and must be reported separately. The current modality
figure reads pooled frame AP, pooled Brier, and pooled event mAP, so it is a
secondary diagnostic rather than the primary participant-macro table.

## Configuration contract

The short-paper configurations inherit the frozen aligned11 paths and change
only model/input selection and isolated result identities:

| Purpose | Configuration | Default seeds |
|---|---|---|
| EPT-Net main | `configs/eptnet_marlin11_eeg_ppg_video.yaml` | 13, 42, 73 |
| Protocol-matched GRU | `configs/gru_marlin11_eeg_ppg_video.yaml` | 13, 42, 73 |
| Fusion Transformer | `configs/transformer_marlin11_eeg_ppg_video.yaml` | 13, 42, 73 |
| Video diagnostic | `configs/eptnet_marlin11_video_only.yaml` | 42 |
| Fixed-reader diagnostic | `configs/eptnet_marlin11_fixed_reader.yaml` | 42 |
| No-persistent-state diagnostic | `configs/eptnet_marlin11_no_persistent.yaml` | 42 |

The preflight rejects a changed dataset identity, manifest path, participant
set, exclusion, label direction, input family, model family, or output-root
contract. It also requires the success marker, checks the dataset summary, and
verifies every manifest-referenced session tensor against its stored SHA-256.
The orchestration script never calls a data-preparation command.

## Environment

Use Python 3.10 or newer and install a device-compatible PyTorch build before
the repository dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m pip check
```

Record the Git commit, Python/PyTorch/CUDA versions, device identity, resolved
configuration, and preflight report with every run. Do not commit private or
machine-specific artifacts.

## Verification and execution order

From the repository root:

```bash
# No private data or GPU required; validates configs and prints commands only.
bash scripts/experiments/run_marlin11_shortpaper.sh --suite all --dry-run

# Requires the authorized artifact at the declared path; records a reference.
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --device cuda:0 \
  --preflight-only

# Executes the declared matrix sequentially after the same preflight.
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --device cuda:0 \
  --skip-existing
```

The structural preflight writes
`results/marlin11_shortpaper_preflight.json`; the main suite also writes
`results/marlin11_shortpaper_parameter_fairness.json`. Each model/seed has a
new `results/<experiment>/seed_<seed>/` directory. The runner refuses existing
or partial directories. `--skip-existing` recognizes only a completed run that
passes the read-only identity, protocol, provenance, prediction, and event
checks in `verify_marlin11_shortpaper_results.py`; compatible aggregates must
also exactly match an in-memory recomputation. Accepted evidence is never
overwritten. `--resume-partial`
continues only
an incomplete new short-paper run with no final metrics and with both its own
`last.pt` and `best.pt`; it resumes the former while preserving best-model
selection in the latter. The flags may be combined to skip complete seeds and
explicitly resume eligible interrupted ones. All other pre-existing states
fail closed, and historical frozen results remain outside this recovery
mechanism. If the preflight record already exists, recovery modes validate the
current frozen-evidence fingerprints against it through `--reference` and do
not rewrite the record. They preserve an existing parameter-fairness audit only
after `--check-reference` confirms exact resolved-config, inherited-config-file,
implementation-source, and training-manifest fingerprints. A legacy or stale
audit fails closed and is never silently reused.

The first preflight requires `_SUCCESS.json`, cross-checks the split contract
recorded by `dataset_summary.json` and the three manifests, and recomputes the
stored SHA-256 for every tensor referenced by those manifests. It records that
verified structure as the local reference, and subsequent recovery runs compare
the current evidence fields with it. The gate does not independently prove the
artifact's authorized origin against an external canonical digest; distribution
checksums and read-only storage remain upstream responsibilities.

If the frozen data are unavailable locally, only `--dry-run` is meaningful.
Its `passed_with_warnings` status must not be represented as a successful
artifact verification.

## Figure provenance

Formal figures may read only completed measured artifacts from the same cohort,
split, decoder, threshold policy, and seed policy. Dynamic-trace sample IDs must
be declared before viewing candidate renderings. Figure code records input
hashes and exports PDF, SVG, and PNG from the same canvas. Extending compact
single-seed modality diagnostics must preserve their original aggregates and
write matched-seed figure aggregates to a new versioned path.

The `--demo` route uses deterministic synthetic fixtures solely for layout and
export QA. Demo outputs are visibly labelled and must never appear in a result
table, claim, abstract, or quantitative caption. Full figure contracts are in
`fig/FIGURE_CONTRACTS.md`.

## Release boundary

Participant media, annotations, processed tensors, manifests containing private
references, checkpoints, per-step predictions, decoded traces, participant
identity mappings, credentials, and machine-specific metadata remain private.
The public release may contain implementation, tests, configurations,
de-identified aggregate summaries, figure code, and rendered figures only after
their release gates pass.

Reproducing numerical results requires authorized access to the exact frozen
artifact. Code tests and preflight checks establish the execution contract;
quantitative paper claims are added only from completed, provenance-matched
experiments. This document introduces no new experimental performance numbers.
