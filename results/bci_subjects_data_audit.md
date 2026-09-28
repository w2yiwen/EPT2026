# De-identified BCI-subject dataset audit

This audit reports aggregate dataset properties only. It contains no subject identifiers, source filenames, or row-level signals.

The 12/2/4 split was frozen before training by exhaustive, model-independent label/modality-aware stratification. It is a pilot split, not cross-validation.

## Subject-disjoint split

| Split | Subjects | Context steps | Target-valid steps | Deception / valid | Prevalence | Events | Median event length |
|---|---:|---:|---:|---:|---:|---:|
| train | 12 | 4982 | 3617 | 817 / 3617 | 0.226 | 168 | 2.0 |
| val | 2 | 918 | 672 | 141 / 672 | 0.210 | 25 | 2.0 |
| test | 4 | 1505 | 1201 | 287 / 1201 | 0.239 | 55 | 4.0 |

## Modality allocation by split

| Modality | Train subjects | Validation subjects | Test subjects |
|---|---:|---:|---:|
| eeg_time | 3 | 1 | 1 |
| physiology | 9 | 1 | 3 |
| video | 11 | 2 | 4 |

## Modality coverage

| Modality | Subjects with any data | Available steps |
|---|---:|---:|
| eeg_time | 5 / 18 | 1931 / 7405 |
| eeg_spectral | 5 / 18 | 1931 / 7405 |
| physiology | 13 / 18 | 5493 / 7405 |
| video | 17 / 18 | 6999 / 7405 |
| audio | 0 / 18 | 0 / 7405 |
| text | 18 / 18 | 6954 / 7405 |

## Limitations

- The fixed split uses labels and modality availability for pre-training stratification.
- One fixed 12/2/4 split is a pilot protocol, not cross-validation or population-level validation.
- Speaker-turn timestamps require within-turn character interpolation.
- Sensor streams are duration-normalized because acquisition clocks are inconsistent.
- EEG and physiology are missing for a subset of subjects and are handled by explicit masks.
- The selected prepared dataset has no audio representation; audio is disabled and unavailable for every subject.
- Target identity is inferred retrospectively from complete-transcript speaking volume; it is not supplied role metadata or an online role-discovery method.
- The target-speaker heuristic and highlighted-label semantics still require independent confirmation by the data owner.
- This audit does not establish annotation validity or population-level generalization.

Prepared-data fingerprint: `d2a1329c34ab96b06ecb288abf6e1a6afb4d83b89fd6f01655f85ad56a847e9f`
