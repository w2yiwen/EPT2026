# User Requirements and Frozen Evidence Boundary

## Immutable current dataset

- The current executable paper pipeline uses only the existing
  `bci_subjects_ept_v6_marlin4060_aligned11` artifact.
- Its processed tensors, manifests, participant selection, temporal alignment,
  feature extraction, labels, splits, and existing result folders are frozen.
- The short-paper workflow must never prepare, migrate, relabel, overwrite, or
  implicitly regenerate this dataset.
- The frozen IDs are `session_002`, `session_003`, `session_004`, `session_006`,
  `session_008`, `session_009`, `session_010`, `session_012`, `session_015`,
  `session_017`, and `session_018`; `session_011` remains excluded.

## Scientific focus

- Frame the work as multimodal temporal recognition and localization in a BCI
  setting, with EEG, PPG-derived physiology, and video as the paper-facing input
  families.
- The model produces a time-varying target-event probability and temporal
  boundaries, not one session-level binary decision.
- Repository semantics are immutable: `0 = deception`, `1 = truth`; probability
  metrics and event decoding therefore treat label `0` as positive.
- New experiment configs may select existing feature branches, but must read the
  same frozen manifests and write only to new result directories.
- The dataset/task and multimodal BCI framing lead the 4--6 page paper; EPT-Net
  is the reference realization rather than the entire contribution.

## Evaluation and claim discipline

- All repository-backed model claims are scoped to the frozen 11-subject aligned
  artifact unless a separate dataset integration is explicitly authorized later.
- A broader collection effort may be described as context, but it is not part of
  the current executable or evaluated code path.
- Report probability quality and temporal/event localization alongside framewise
  discrimination, with only a compact modality analysis and targeted mechanism
  diagnostic.
- Generated figures must distinguish measured outputs from demo or schematic
  content and must never imply an unrun GPU experiment.

## Delivery

- Additive code may provide evaluation metrics, experiment orchestration,
  paper-figure rendering, and documentation around the frozen artifact.
- No participant data, sensor arrays, private predictions, credentials,
  checkpoints, or machine-specific paths may be committed.
- The local checkout may omit large files; server commands must fail safely when
  the frozen data are absent.
