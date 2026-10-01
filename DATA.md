# Data and model assets

The repository does not redistribute the BCI recordings or the frozen aligned
feature artifact. Put `bci_subjects_ept_v6_aligned11` under `data/processed/`
(or update paths in a copied configuration), then run the preflight before
training. The public code reads the immutable tensors and manifests; historical
raw-data conversion, cache migration, and feature-extractor installers are not
part of the final experiment release. Checkpoints and metrics belong under
`results/`; generated paper figures stay under `fig/` and are ignored by Git.
