# Data Audit

- Audit date: 2026-09-26
- Dataset: `bci_truth_deception_v1`
- Label handling: **read-only; no smoothing, merging, or relabeling**
- Event-safe split boundaries: **yes**

## Split statistics

| Split | Unique steps | Label 0 / 1 | Event runs | Event length steps (min/median/mean/max) | Boundaries start/end | Time span (s) | Windows | Duplicate observation rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 709 | 420 / 289 | 172 | 1/2.00/2.44/17 | 172 / 172 | 281.96 | 22 | 49.64% |
| val | 150 | 75 / 75 | 36 | 1/1.00/2.08/7 | 36 / 36 | 57.70 | 4 | 41.41% |
| test | 156 | 97 / 59 | 30 | 1/2.00/3.23/16 | 30 / 30 | 62.92 | 4 | 39.06% |

## Overall

- 1015 unique steps over 402.58 seconds.
- Class counts: label 0 = 592, label 1 = 423.
- 238 event runs; length min/median/mean/max = 1/2.00/2.49/17 steps.
- Derived boundary counts: 238 starts and 238 ends.

## Leakage and temporal-dependence screen

| Modality | Max |r| train | Max |r| test (post-hoc) | Train-selected 1-feature BA train/test | Train-selected 1-feature test AUROC | Exact test rows in train | Test nearest-train cosine q95 | Test feature lag-1 median |
|---|---:|---:|---:|---:|---:|---:|---:|
| eeg_time | 0.070 | 0.045 | 0.557/0.500 | 0.500 | 0 | 1.0000 | 0.000 |
| eeg_spectral | 0.081 | 0.151 | 0.553/0.503 | 0.505 | 53 | 1.0000 | 0.263 |
| physiology | 0.063 | 0.155 | 0.577/0.551 | 0.541 | 0 | 0.9997 | 0.382 |
| video | 0.196 | 0.210 | 0.590/0.498 | 0.450 | 0 | 0.8949 | 0.321 |
| audio | 0.280 | 0.268 | 0.636/0.624 | 0.658 | 0 | 0.8650 | 0.140 |
| text | 0.424 | 0.410 | 0.711/0.681 | 0.708 | 105 | 1.0000 | -0.049 |

## Seed-42 result interpretation

The observed EPT-Net test AUROC is **0.943** on 156 unique steps from one session.

The score is a within-session, one-seed pilot and may be inflated by participant/session leakage, temporal dependence, high-dimensional feature selection, and unverified upstream feature extraction. The audit does not prove that a label column was copied into the inputs.

Across all model inputs jointly, 0 test rows exactly match a training row. For text alone, 105/156 test rows exactly match a training embedding; the training-majority label agrees on 100.0% of those matches. An exact-text lookup with the training-majority fallback reaches test balanced accuracy 0.737.

Test-label lag-1 correlation is 0.206; a previous-label predictor reaches balanced accuracy 0.603. Thus label persistence is present but does not by itself account for AUROC 0.943; same-session feature and lexical reuse remain important confounds.

The strongest train-selected single feature on the test split came from `text` with AUROC 0.708 and balanced accuracy 0.681. Low single-feature separability or zero exact row matches cannot exclude multivariate leakage or identity/session memorization. Post-hoc test maxima are diagnostics only.

## Provenance and definitions

Raw audio/video and word timestamps are present, but the staged bundle contains no extractor code. The 768-dimensional text embedding, video landmark/action-unit features, and acoustic features therefore cannot be reproduced or checked for causal context and fit scope from this bundle alone.

Boundary counts were deterministically recomputed from contiguous label-0 runs. Window duplication is reported as `(window observations - unique steps) / window observations`; it describes storage/sampling overlap, not additional independent data.

The observational unit remains one session. Split rows and event runs are temporally correlated and must not be interpreted as independent participants.
