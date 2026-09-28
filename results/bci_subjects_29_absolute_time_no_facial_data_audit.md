# De-identified BCI-subject dataset audit

This audit reports aggregate dataset properties only. It contains no subject identifiers, source filenames, or row-level signals.

The 20/4/5 split was frozen before training by model-independent label/modality-aware stratification (`fixed_seed_constrained_monte_carlo_stratification_v2`). It is a fixed pilot split, not cross-validation.

## Subject-disjoint split

| Split | Subjects | Context steps | Target-valid steps | Deception / valid | Prevalence | Events | Median event length |
|---|---:|---:|---:|---:|---:|---:|
| train | 20 | 8380 | 6454 | 1617 / 6454 | 0.251 | 266 | 3.0 |
| val | 4 | 1622 | 1294 | 314 / 1294 | 0.243 | 51 | 3.0 |
| test | 5 | 1864 | 1524 | 408 / 1524 | 0.268 | 64 | 2.0 |

## Modality allocation by split

| Modality | Train subjects | Validation subjects | Test subjects |
|---|---:|---:|---:|
| eeg_time | 2 | 0 | 0 |
| physiology | 8 | 2 | 2 |
| video | 0 | 0 | 0 |

## Modality coverage

| Modality | Subjects with any data | Available steps |
|---|---:|---:|
| eeg_time | 2 / 29 | 883 / 11866 |
| eeg_spectral | 2 / 29 | 883 / 11866 |
| physiology | 12 / 29 | 4473 / 11866 |
| video | 0 / 29 | 0 / 11866 |
| audio | 0 / 29 | 0 / 11866 |
| text | 29 / 29 | 11411 / 11866 |

## Absolute-time alignment

- Scope: `absolute_timestamp_intersection`
- Shared hardware clock: `False`
- EEG valid overlap: 2 / 8 source-present sessions (883 one-second steps)
- PPG valid overlap: 12 / 14 source-present sessions (4473 one-second steps)
- Facial CSV source records: 0
- Known non-overlap control rejected: `True`

## Limitations

- The fixed split uses labels and modality availability for pre-training stratification.
- One fixed 20/4/5 split is a pilot protocol, not cross-validation or population-level validation.
- Speaker-turn timestamps require within-turn character interpolation.
- Sensor samples are retained only where source and transcript absolute-time intervals intersect; device clocks did not share a hardware trigger, so sub-second lag claims remain unsupported.
- Only 2 of 29 sessions contain valid absolute-time EEG overlap and 12 contain valid PPG overlap.
- The frozen validation and test splits contain no valid EEG sessions; this dataset split cannot support an EEG generalization claim.
- EEG and physiology are missing for a subset of subjects and are handled by explicit masks.
- The selected prepared dataset has no audio representation; audio is disabled and unavailable for every subject.
- Target identity is inferred retrospectively from complete-transcript speaking volume; it is not supplied role metadata or an online role-discovery method.
- The target-speaker heuristic and highlighted-label semantics still require independent confirmation by the data owner.
- This audit does not establish annotation validity or population-level generalization.

Prepared-data fingerprint: `b06a12e5f5120497916d4b41a976e9a29cbbe7e4c53cf723892189588f9f5bb3`
