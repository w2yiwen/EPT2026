"""Strictly causal GateHUB baseline adapted from the official CVPR 2022 code.

Only the online Gated History Unit is used.  The optional Future-augmented
History path is intentionally excluded because it is incompatible with the
shared past-and-present-only evaluation protocol.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..eptnet import EPTNetConfig
from .common import _BaselineBase
from .position import SinusoidalPositionEncoding


class ProjectedAttention(nn.Module):
    """GateHUB-style multi-head attention with optional history gates."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def _heads(self, values: Tensor) -> Tensor:
        batch, steps, _ = values.shape
        return values.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        causal: bool = False,
        gate_scores: Tensor | None = None,
    ) -> Tensor:
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(key))
        v = self._heads(self.v_proj(value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if gate_scores is not None:
            # Official GateHUB Eq. 3: G = log(z) + z, z in (0, 1).
            z = gate_scores.clamp_min(torch.finfo(scores.dtype).tiny)
            scores = scores + (torch.log(z) + z).unsqueeze(1).unsqueeze(2)
        if causal:
            mask = torch.triu(
                torch.ones(
                    query.shape[1], key.shape[1], dtype=torch.bool, device=query.device
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(mask.view(1, 1, *mask.shape), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous()
        attended = attended.view(query.shape[0], query.shape[1], self.hidden_dim)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.norm(inputs + self.dropout(self.network(inputs)))


class HistoryLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = ProjectedAttention(hidden_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.norm(
            inputs + self.dropout(self.attention(inputs, inputs, inputs))
        )
        return self.ffn(inputs)


class PresentLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = ProjectedAttention(hidden_dim, num_heads, dropout)
        self.cross_attention = ProjectedAttention(hidden_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout)

    def forward(self, present: Tensor, history: Tensor) -> Tensor:
        present = self.norm1(
            present
            + self.dropout(
                self.self_attention(present, present, present, causal=True)
            )
        )
        present = self.norm2(
            present
            + self.dropout(self.cross_attention(present, history, history))
        )
        return self.ffn(present)


class GateHUB(_BaselineBase):
    """Gated history compression followed by causal present decoding."""

    def __init__(self, config: EPTNetConfig) -> None:
        super().__init__(config)
        d = config.hidden_dim
        self.present_steps = config.gatehub_present_steps
        self.position_encoding = SinusoidalPositionEncoding()
        self.gate_layer = nn.Linear(d, 1)
        self.latent_query = nn.Parameter(torch.empty(1, config.gatehub_latent_size, d))
        self.null_history = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.latent_query, std=0.02)
        self.gated_history_attention = ProjectedAttention(
            d, config.num_heads, config.dropout
        )
        self.history_dropout = nn.Dropout(config.dropout)
        self.history_norm = nn.LayerNorm(d)
        self.history_ffn = FeedForward(d, config.dropout)
        self.history_layers = nn.ModuleList(
            HistoryLayer(d, config.num_heads, config.dropout)
            for _ in range(config.gatehub_history_layers)
        )
        self.present_layers = nn.ModuleList(
            PresentLayer(d, config.num_heads, config.dropout)
            for _ in range(config.gatehub_decoder_layers)
        )
        self.output_norm = nn.LayerNorm(d)

    def _encode_history(self, history: Tensor) -> Tensor:
        batch_size = history.shape[0]
        if history.shape[1] == 0:
            history = self.null_history.expand(batch_size, -1, -1)
        history = self.position_encoding(history)
        gate_scores = torch.sigmoid(self.gate_layer(history)).squeeze(-1) + 1e-8
        latent = self.latent_query.expand(batch_size, -1, -1)
        latent = self.history_norm(
            latent
            + self.history_dropout(
                self.gated_history_attention(
                    latent, history, history, gate_scores=gate_scores
                )
            )
        )
        latent = self.history_ffn(latent)
        for layer in self.history_layers:
            latent = layer(latent)
        return latent

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        fused = self.encode(batch)
        blocks: list[Tensor] = []
        for start in range(0, fused.shape[1], self.present_steps):
            end = min(start + self.present_steps, fused.shape[1])
            history = self._encode_history(fused[:, :start])
            present = self.position_encoding(fused[:, start:end])
            for layer in self.present_layers:
                present = layer(present, history)
            blocks.append(self.output_norm(present))
        states = torch.cat(blocks, dim=1)
        return self.make_output(states, batch["sequence_mask"])
