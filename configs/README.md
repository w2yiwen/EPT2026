# Configuration layout

Configuration filenames use the compact form `<model>_<protocol>_<seed>.yaml`.
The root files define the six configurations in the final experiment matrix;
the `experiments/` directory contains the canonical minimal entry point.

Every paper configuration inherits the same frozen aligned11 data contract:
all 11 sessions train together, while validation and test manifests are fixed
training-included evaluation views rather than held-out cohorts.
