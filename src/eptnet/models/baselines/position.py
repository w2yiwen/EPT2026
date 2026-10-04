from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionEncoding(nn.Module):
    """Add length-independent sinusoidal positions to ``[B, T, D]`` inputs."""

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
