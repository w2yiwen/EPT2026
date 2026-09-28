from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncodedSequence:
    """One aligned feature sequence with explicit per-step availability."""

    features: np.ndarray
    mask: np.ndarray

    def validate(self, *, output_dim: int) -> EncodedSequence:
        features = np.asarray(self.features)
        mask = np.asarray(self.mask)
        if features.ndim != 2 or features.shape[1] != output_dim:
            raise ValueError(f"features must have shape [T, {output_dim}]")
        if mask.shape != (features.shape[0],):
            raise ValueError("mask must have shape [T]")
        if not np.isfinite(features).all():
            raise ValueError("encoded features must be finite")
        if not np.all(features[~mask.astype(bool)] == 0):
            raise ValueError("unavailable feature rows must be exactly zero")
        return self
