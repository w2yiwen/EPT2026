from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from ..eptnet import EPTNetConfig
from .common import _BaselineBase


class EarlyFusionGRU(_BaselineBase):
    """Encode modalities first, then model the fused causal sequence with a GRU."""

    def __init__(self, config: EPTNetConfig) -> None:
        super().__init__(config)
        self.temporal = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        fused = self.encode(batch)
        states, _ = self.temporal(fused)
        return self.make_output(states, batch["sequence_mask"])
