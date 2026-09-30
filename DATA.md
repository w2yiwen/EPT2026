# Data and model assets

The repository does not redistribute the BCI data or private pretrained model
assets. Put the dataset under `/root/data` on the server (or set the equivalent
paths in a copied configuration), then run the manifest and provenance audits
before training. Generated caches, checkpoints, and result figures belong under
`data/processed/`, `results/`, and `paper/figures/` and are intentionally not
source inputs.
