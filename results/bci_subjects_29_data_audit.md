# De-identified BCI-subject dataset audit

This audit reports aggregate dataset properties only. It contains no subject identifiers, source filenames, or row-level signals.

The 20/4/5 split was frozen before training by model-independent label/modality-aware stratification (`fixed_seed_constrained_monte_carlo_stratification_v2`). It is a fixed pilot split, not cross-validation.

## Subject-disjoint split

| Split | Subjects | Context steps | Target-valid steps | Deception / valid | Prevalence | Events | Median event length |
|---|---:|---:|---:|---:|---:|---:|
| train | 20 | 8195 | 6281 | 1641 / 6281 | 0.261 | 267 | 3.0 |
| val | 4 | 1501 | 1234 | 312 / 1234 | 0.253 | 51 | 3.0 |
| test | 5 | 2170 | 1757 | 386 / 1757 | 0.220 | 63 | 3.0 |

## Modality allocation by split

| Modality | Train subjects | Validation subjects | Test subjects |
|---|---:|---:|---:|
| eeg_time | 6 | 1 | 1 |
| physiology | 10 | 2 | 2 |
| video | 14 | 3 | 3 |

## Modality coverage

| Modality | Subjects with any data | Available steps |
|---|---:|---:|
| eeg_time | 8 / 29 | 2741 / 11866 |
| eeg_spectral | 8 / 29 | 2741 / 11866 |
| physiology | 14 / 29 | 5731 / 11866 |
| video | 20 / 29 | 8013 / 11866 |
| audio | 0 / 29 | 0 / 11866 |
| text | 29 / 29 | 11411 / 11866 |

## Limitations

- The fixed split uses labels and modality availability for pre-training stratification.
- One fixed 20/4/5 split is a pilot protocol, not cross-validation or population-level validation.
- Speaker-turn timestamps require within-turn character interpolation.
- Sensor streams are duration-normalized because acquisition clocks are inconsistent.
- EEG and physiology are missing for a subset of subjects and are handled by explicit masks.
- The selected prepared dataset has no audio representation; audio is disabled and unavailable for every subject.
- Target identity is inferred retrospectively from complete-transcript speaking volume; it is not supplied role metadata or an online role-discovery method.
- The target-speaker heuristic and highlighted-label semantics still require independent confirmation by the data owner.
- This audit does not establish annotation validity or population-level generalization.

Prepared-data fingerprint: `f988292741f0daf86299634a74e9855ad5f3ed1f00d6389b346709e2a680a781`
