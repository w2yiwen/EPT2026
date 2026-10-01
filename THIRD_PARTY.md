# Third-party components

Runtime dependencies are declared in `pyproject.toml` and pinned in the lock
files. The final repository contains no vendored pretrained model source or
weights. Frozen feature tensors remain external data; their provenance is
checked by the experiment preflight.
