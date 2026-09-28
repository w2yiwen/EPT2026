# BCI session compliance report

Selected profile: `strong_behavior_sources`.

Result: **1/18 sessions compliant**.

Missing raw signals are not imputed or synthesized. A zero feature row is valid only when its availability mask is false.

## Profile coverage

| Profile | Compliant sessions | Coverage |
|---|---:|---:|
| `model_contract` | 18/18 | 100.0% |
| `legacy_face_text` | 17/18 | 94.4% |
| `strong_behavior_sources` | 1/18 | 5.6% |
| `strong_behavior_features` | 0/18 | 0.0% |
| `complete_multimodal_sources` | 0/18 | 0.0% |

## Selected-profile failures

| Session | Status | Missing requirements |
|---|---|---|
| `session_001` | fail | `raw_video`, `raw_audio` |
| `session_002` | fail | `raw_video`, `raw_audio` |
| `session_003` | fail | `raw_video`, `raw_audio` |
| `session_004` | fail | `raw_video`, `raw_audio` |
| `session_005` | fail | `raw_video`, `raw_audio` |
| `session_006` | fail | `raw_video`, `raw_audio` |
| `session_007` | fail | `raw_video`, `raw_audio` |
| `session_008` | fail | `raw_video`, `raw_audio` |
| `session_009` | fail | `raw_video`, `raw_audio` |
| `session_010` | fail | `raw_video`, `raw_audio` |
| `session_011` | fail | `raw_video`, `raw_audio` |
| `session_012` | fail | `raw_video`, `raw_audio` |
| `session_013` | fail | `raw_video`, `raw_audio` |
| `session_014` | fail | `raw_video`, `raw_audio` |
| `session_015` | fail | `raw_video`, `raw_audio` |
| `session_016` | fail | `raw_video`, `raw_audio` |
| `session_017` | pass | — |
| `session_018` | fail | `raw_video`, `raw_audio` |

## Interpretation

- `model_contract` means the session can safely enter the masked EPT pipeline.
- `legacy_face_text` additionally requires observed face and transcript features.
- `strong_behavior_sources` requires real video and audio for frozen published extractors; placeholders do not pass.
- `strong_behavior_features` requires observed OpenFace, WavLM Base+, and MacBERT features after extraction.
- `complete_multimodal_sources` further requires usable EEG and paired PPG.
