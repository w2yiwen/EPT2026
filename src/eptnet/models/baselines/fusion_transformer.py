from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from ..eptnet import EPTNetConfig
from .common import _BaselineBase


class SinusoidalPositionEncoding(nn.Module):
    """Add deterministic absolute positions without a fixed maximum length.

    The encoding for a position depends only on its absolute index and the
    hidden dimension. In particular, it is independent of the total sequence
    length, which preserves prefix equivalence for causal inference. Trig
    functions are evaluated in float32 for low-precision inputs and the result
    is cast back to the input dtype before addition.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("SinusoidalPositionEncoding expects [batch, steps, hidden] input")
        if not inputs.is_floating_point():
            raise TypeError("SinusoidalPositionEncoding requires floating-point input")

        _, steps, hidden_dim = inputs.shape
        if hidden_dim <= 0:
            raise ValueError("The hidden dimension must be positive")
        compute_dtype = (
            torch.float32 if inputs.dtype in (torch.float16, torch.bfloat16) else inputs.dtype
        )
        positions = torch.arange(steps, device=inputs.device, dtype=compute_dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_dim, 2, device=inputs.device, dtype=compute_dtype)
            * (-math.log(10_000.0) / hidden_dim)
        )
        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.empty((steps, hidden_dim), device=inputs.device, dtype=compute_dtype)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : hidden_dim // 2])
        return inputs + encoding.to(dtype=inputs.dtype).unsqueeze(0)


class FusionTransformer(_BaselineBase):
    """Encode modalities first, then apply a position-aware causal Transformer."""

    def __init__(self, config: EPTNetConfig) -> None:
        super().__init__(config)
        self.position_encoding = SinusoidalPositionEncoding()
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        fused = self.position_encoding(self.encode(batch))
        steps = fused.shape[1]
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=fused.device), diagonal=1
        )
        states = self.temporal(
            fused,
            mask=causal_mask,
            src_key_padding_mask=~batch["sequence_mask"].bool(),
        )
        return self.make_output(states, batch["sequence_mask"])
