# BCI session compliance report

Selected profile: `model_contract`.

Result: **29/29 sessions compliant**.

Missing raw signals are not imputed or synthesized. A zero feature row is valid only when its availability mask is false.

## Profile coverage

| Profile | Compliant sessions | Coverage |
|---|---:|---:|
| `model_contract` | 29/29 | 100.0% |
| `legacy_face_text` | 20/29 | 69.0% |
| `strong_behavior_sources` | 12/29 | 41.4% |
| `strong_behavior_features` | 0/29 | 0.0% |
| `complete_multimodal_sources` | 7/29 | 24.1% |

## Selected-profile failures

| Session | Status | Missing requirements |
|---|---|---|
| `session_001` | pass | — |
| `session_002` | pass | — |
| `session_003` | pass | — |
| `session_004` | pass | — |
| `session_005` | pass | — |
| `session_006` | pass | — |
| `session_007` | pass | — |
| `session_008` | pass | — |
| `session_009` | pass | — |
| `session_010` | pass | — |
| `session_011` | pass | — |
| `session_012` | pass | — |
| `session_013` | pass | — |
| `session_014` | pass | — |
| `session_015` | pass | — |
| `session_016` | pass | — |
| `session_017` | pass | — |
| `session_018` | pass | — |
| `session_019` | pass | — |
| `session_020` | pass | — |
| `session_021` | pass | — |
| `session_022` | pass | — |
| `session_023` | pass | — |
| `session_024` | pass | — |
| `session_025` | pass | — |
| `session_026` | pass | — |
| `session_027` | pass | — |
| `session_028` | pass | — |
| `session_029` | pass | — |

## Interpretation

- `model_contract` means the session can safely enter the masked EPT pipeline.
- `legacy_face_text` additionally requires observed face and transcript features.
- `strong_behavior_sources` requires real video and audio for frozen published extractors; placeholders do not pass.
- `strong_behavior_features` requires observed OpenFace, WavLM Base+, and MacBERT features after extraction.
- `complete_multimodal_sources` further requires usable EEG and paired PPG.
