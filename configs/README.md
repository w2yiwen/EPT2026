# Configuration layout

Configuration filenames use the compact form `<model>_<protocol>_<seed>.yaml`.
The `experiments/` directory contains canonical entry points; the root files
remain compatibility targets for older launchers and are not new experiment
definitions.

`experiments/aligned11_eptnet.yaml` is the canonical all-11 training route.
It inherits the validated aligned11 defaults and keeps the explicit
`data.cohort_policy: all_sessions_training` contract.
