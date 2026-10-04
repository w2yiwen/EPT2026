"""Causal LSTR baseline adapted from the official Amazon Science implementation.

The original project separates a long history from an eight-frame work memory,
compresses the history with learned queries, and decodes the work memory against
that compressed representation.  This adaptation keeps that temporal core while
using EPT-Net's shared modality encoders and prediction heads.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..eptnet import EPTNetConfig
from .common import _BaselineBase
from .position import SinusoidalPositionEncoding


def _decoder_layer(config: EPTNetConfig) -> nn.TransformerDecoderLayer:
    return nn.TransformerDecoderLayer(
        d_model=config.hidden_dim,
        nhead=config.num_heads,
        dim_feedforward=config.hidden_dim * 4,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class LSTR(_BaselineBase):
    """Long Short-Term Transformer with strictly causal blockwise prediction."""

    def __init__(self, config: EPTNetConfig) -> None:
        super().__init__(config)
        d = config.hidden_dim
        q1, q2 = config.lstr_long_queries
        l1, l2 = config.lstr_encoder_layers
        self.work_steps = config.lstr_work_steps
        self.position_encoding = SinusoidalPositionEncoding()
        self.long_query_1 = nn.Parameter(torch.empty(1, q1, d))
        self.long_query_2 = nn.Parameter(torch.empty(1, q2, d))
        self.null_history = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.long_query_1, std=0.02)
        nn.init.normal_(self.long_query_2, std=0.02)
        self.long_encoder_1 = nn.TransformerDecoder(
            _decoder_layer(config), l1, norm=nn.LayerNorm(d)
        )
        self.long_encoder_2 = nn.TransformerDecoder(
            _decoder_layer(config), l2, norm=nn.LayerNorm(d)
        )
        self.work_decoder = nn.TransformerDecoder(
            _decoder_layer(config), config.lstr_decoder_layers, norm=nn.LayerNorm(d)
        )

    def _compress_history(self, history: Tensor) -> Tensor:
        batch_size = history.shape[0]
        if history.shape[1] == 0:
            history = self.null_history.expand(batch_size, -1, -1)
        first = self.long_encoder_1(self.long_query_1.expand(batch_size, -1, -1), history)
        return self.long_encoder_2(self.long_query_2.expand(batch_size, -1, -1), first)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        fused = self.position_encoding(self.encode(batch))
        blocks: list[Tensor] = []
        for start in range(0, fused.shape[1], self.work_steps):
            end = min(start + self.work_steps, fused.shape[1])
            work = fused[:, start:end]
            long_memory = self._compress_history(fused[:, :start])
            length = end - start
            causal_mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=fused.device), diagonal=1
            )
            blocks.append(self.work_decoder(work, long_memory, tgt_mask=causal_mask))
        states = torch.cat(blocks, dim=1)
        return self.make_output(states, batch["sequence_mask"])
