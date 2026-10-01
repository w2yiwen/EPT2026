# EPT-Net repository architecture

The repository is organized by responsibility so that a paper result can be
traced from a frozen configuration to source data, model code, metrics, and
figures.

```text
src/eptnet/
  config.py                 strict YAML loading and validation
  data/                     frozen-manifest readers and tensor schemas
  models/                   EPT-Net and paper baselines
  training/                 training loop, loss, and trainer CLI
  evaluation/              decoder, metrics, evaluator, aggregation
  provenance.py             source/config/data fingerprints
  cli/                      public `ept` command
configs/                    experiment specifications
scripts/                    experiment, audit, and reporting entry points
fig/                        paper figure contracts and generators
tests/                      unit, contract, integration, and reproducibility tests
```

`eptnet.train`, `eptnet.evaluate`, `eptnet.aggregate`, and the top-level
metric/decoder modules are small stable command facades. New code should use
the domain paths under `training/` and `evaluation/`; the facades are retained
so recorded commands and checkpoints remain executable.

The canonical aligned11 entry point is
`configs/experiments/aligned11_eptnet.yaml`. It preserves the declared
`all_sessions_training` policy: all 11 aligned sessions are used by the
training manifest, while validation and test remain fixed evaluation views.
