# Reproducibility contract

Every formal run must record:

1. the resolved YAML configuration;
2. the Git commit and source-tree fingerprint;
3. manifest fingerprints and the declared cohort policy;
4. Python, NumPy, PyTorch, platform, and resolved-device metadata;
5. deterministic settings, seed, checkpoint provenance, metrics, and figure inputs.

Use the public entry point for environment checks and config validation:

```bash
ept doctor --config configs/experiments/aligned11_eptnet.yaml --data-root /root/data
ept resolve-config --config configs/experiments/aligned11_eptnet.yaml \
  --output results/aligned11_eptnet/resolved.yaml
```

The repository intentionally does not run a complete formal experiment as part
of CI. CI validates imports, configuration contracts, deterministic writers,
training/evaluation semantics, and dry-run paths; a full run remains an explicit
server operation using the pinned data manifest.
