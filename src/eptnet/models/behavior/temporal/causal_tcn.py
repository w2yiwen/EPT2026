from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn


class _CausalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, *, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        values = inputs.transpose(1, 2)
        values = self.conv(F.pad(values, (self.left_padding, 0))).transpose(1, 2)
        values = self.dropout(F.gelu(values))
        return self.norm(residual + values)


class CausalTCNDecoder(nn.Module):
    """Capacity-controlled causal decoder shared by all behavior encoders."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("input_dim, hidden_dim, and num_layers must be positive")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                _CausalResidualBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
                for layer in range(num_layers)
            ]
        )
        self.classification_head = nn.Linear(hidden_dim, num_classes)
        self.boundary_head = nn.Linear(hidden_dim, 2)
        self.offset_head = nn.Linear(hidden_dim, 2)

    def forward(self, features: Tensor, sequence_mask: Tensor) -> dict[str, Any]:
        if features.ndim != 3:
            raise ValueError("features must have shape [B, T, D]")
        if sequence_mask.shape != features.shape[:2]:
            raise ValueError("sequence_mask must have shape [B, T]")
        mask = sequence_mask.bool()
        states = self.input_projection(features * mask.unsqueeze(-1).to(features.dtype))
        states = states * mask.unsqueeze(-1).to(states.dtype)
        for block in self.blocks:
            states = block(states)
            states = states * mask.unsqueeze(-1).to(states.dtype)
        return {
            "class_logits": self.classification_head(states),
            "boundary_logits": self.boundary_head(states),
            "offsets": F.softplus(self.offset_head(states)),
            "states": states,
            "sequence_mask": mask,
        }
