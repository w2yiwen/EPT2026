"""TeSTra baseline adapted from the official ECCV 2022 implementation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..eptnet import EPTNetConfig
from .common import _BaselineBase
from .lstr import _decoder_layer
from .position import SinusoidalPositionEncoding


class DecayedCrossAttention(nn.Module):
    """Multi-head attention with TeSTra's exponential recency prior."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, decay: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.decay = decay
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def _heads(self, values: Tensor) -> Tensor:
        batch, steps, _ = values.shape
        return values.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(memory))
        v = self._heads(self.v_proj(memory))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        age = torch.arange(
            memory.shape[1] - 1, -1, -1, device=memory.device, dtype=scores.dtype
        )
        scores = scores + age.view(1, 1, 1, -1) * math.log(self.decay)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous()
        attended = attended.view(query.shape[0], query.shape[1], self.hidden_dim)
        return self.out_proj(attended)


class DecayedQueryCompressor(nn.Module):
    """Transformer decoder layer whose history cross-attention is smoothed in time."""

    def __init__(self, config: EPTNetConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        self.self_attention = nn.MultiheadAttention(
            d, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.cross_attention = DecayedCrossAttention(
            d, config.num_heads, config.dropout, config.testra_decay
        )
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d * 4, d)
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        residual = self.norm1(query)
        query = query + self.dropout(self.self_attention(residual, residual, residual)[0])
        query = query + self.dropout(self.cross_attention(self.norm2(query), memory))
        return query + self.dropout(self.ffn(self.norm3(query)))


class TeSTra(_BaselineBase):
    """Temporal Smoothing Transformer with causal long/short memory blocks."""

    def __init__(self, config: EPTNetConfig) -> None:
        super().__init__(config)
        d = config.hidden_dim
        q1, q2 = config.lstr_long_queries
        _, l2 = config.lstr_encoder_layers
        self.work_steps = config.lstr_work_steps
        self.position_encoding = SinusoidalPositionEncoding()
        self.long_query_1 = nn.Parameter(torch.empty(1, q1, d))
        self.long_query_2 = nn.Parameter(torch.empty(1, q2, d))
        self.null_history = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.long_query_1, std=0.02)
        nn.init.normal_(self.long_query_2, std=0.02)
        self.smoothing_encoder = DecayedQueryCompressor(config)
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
        first = self.smoothing_encoder(
            self.long_query_1.expand(batch_size, -1, -1), history
        )
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
