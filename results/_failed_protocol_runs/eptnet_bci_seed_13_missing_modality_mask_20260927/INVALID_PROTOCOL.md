# Invalid protocol run: missing-modality mask propagation

This interrupted run must not be used as experimental evidence.

- Experiment: `eptnet_bci_subjects_no_text`, seed 13.
- Interrupted after epoch 18 on 2026-09-27, before test evaluation.
- Reason: physiology encoder outputs were zeroed for unavailable streams, but the
  EPT reader cache mask and multimodal-update token mask still treated those
  branches as available. Learned positional/bias terms could therefore turn a
  zero placeholder into non-zero pseudo-evidence.
- Required remediation: propagate per-step physiology availability through the
  reader and fusion update, prove missing-value invariance, regenerate source
  provenance, and restart the complete formal run from epoch 1.

The directory was moved intact rather than deleted so the failed protocol and
its learning curve remain auditable.
