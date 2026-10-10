"""PMSU: seven-token attention and boundary-modulated GRU increments."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PersistentStateUpdate(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.token_embeddings = nn.Parameter(torch.randn(1, 7, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                                batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(4 * hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.write_gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, state: Tensor, tokens: Tensor, mask: Tensor,
                anchors: Tensor) -> tuple[Tensor, Tensor]:
        valid = mask.any(dim=-1)
        tokens = torch.where(mask[..., None], tokens + self.token_embeddings,
                             torch.zeros_like(tokens))
        safe = mask.clone()
        safe[~valid, 0] = True
        attended, _ = self.attention(state[:, None], tokens, tokens,
                                     key_padding_mask=~safe, need_weights=False)
        update = self.attention_norm(state + attended[:, 0])
        update = self.norm(update + self.feedforward(update))
        candidate = self.gru(update, state)
        transition = 1 - (1 - anchors[:, 1]) * (1 - anchors[:, 2])
        gate = torch.sigmoid(self.write_gate(torch.cat((update, state), dim=-1))
                             + (2 * transition - 1)[:, None])
        gate = torch.where(valid[:, None], gate, torch.zeros_like(gate))
        return state + gate * (candidate - state), gate
