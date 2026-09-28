# De-identified BCI-subject dataset audit

This audit reports aggregate dataset properties only. It contains no subject identifiers, source filenames, or row-level signals.

The 20/4/5 split was frozen before training by model-independent label/modality-aware stratification (`fixed_seed_constrained_monte_carlo_stratification_v2`). It is a fixed pilot split, not cross-validation.

## Subject-disjoint split

| Split | Subjects | Context steps | Target-valid steps | Deception / valid | Prevalence | Events | Median event length |
|---|---:|---:|---:|---:|---:|---:|
| train | 20 | 8222 | 6428 | 1613 / 6428 | 0.251 | 264 | 3.0 |
| val | 4 | 1693 | 1303 | 311 / 1303 | 0.239 | 52 | 3.5 |
| test | 5 | 1951 | 1541 | 415 / 1541 | 0.269 | 65 | 3.0 |

## Modality allocation by split

| Modality | Train subjects | Validation subjects | Test subjects |
|---|---:|---:|---:|
| eeg_time | 5 | 1 | 1 |
| physiology | 8 | 2 | 2 |
| video | 0 | 0 | 0 |

## Modality coverage

| Modality | Subjects with any data | Available steps |
|---|---:|---:|
| eeg_time | 7 / 29 | 2377 / 11866 |
| eeg_spectral | 7 / 29 | 2377 / 11866 |
| physiology | 12 / 29 | 4637 / 11866 |
| video | 0 / 29 | 0 / 11866 |
| audio | 0 / 29 | 0 / 11866 |
| text | 29 / 29 | 11411 / 11866 |

## Clock alignment

- Scope: `session_registered_timestamp_intersection`
- Shared hardware clock: `False`
- EEG valid overlap: 7 / 8 source-present sessions (2377 one-second steps)
- PPG valid overlap: 12 / 14 source-present sessions (4637 one-second steps)
- Facial CSV source records: 0
- EEG sessions using a constant clock offset: 4
- Confirmed clock-mismatch control registered: `True`

## Limitations

- The fixed split uses labels and modality availability for pre-training stratification.
- One fixed 20/4/5 split is a pilot protocol, not cross-validation or population-level validation.
- Speaker-turn timestamps require within-turn character interpolation.
- Confirmed same-session EEG first uses direct absolute intersection; when its device clock does not overlap, one documented session-end constant offset is applied. This is registration, not shared hardware synchronization, so sub-second lag claims remain unsupported.
- Only 7 of 29 sessions contain valid EEG overlap after the declared clock policy and 12 contain valid PPG overlap.
- EEG split coverage must be interpreted together with the reported masks.
- EEG and physiology are missing for a subset of subjects and are handled by explicit masks.
- The selected prepared dataset has no audio representation; audio is disabled and unavailable for every subject.
- Target identity is inferred retrospectively from complete-transcript speaking volume; it is not supplied role metadata or an online role-discovery method.
- The target-speaker heuristic and highlighted-label semantics still require independent confirmation by the data owner.
- This audit does not establish annotation validity or population-level generalization.

Prepared-data fingerprint: `696b60fdb0f571203b2e72da494e192ea996d30ee65823e6c0d4475d9cd30b4b`
