# SPDX-License-Identifier: MIT
# Copyright (c) 2020 Yao-Hung Hubert Tsai and Shaojie Bai

"""Causal MulT baseline adapted from the official ACL 2019 implementation.

Upstream: https://github.com/yaohungt/Multimodal-Transformer (MIT)

The original MulT models every ordered pair among language, audio, and vision.
This baseline applies the same construction to EPT-Net's four shared evidence
streams: temporal EEG, spectral EEG, heart-rate/physiology, and behavioural
context.  With three inputs the construction reduces to the six directional
cross-modal Transformers in the official model.
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


def _linear(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class CrossModalTransformerLayer(nn.Module):
    """Official-style pre-norm attention and feed-forward residual block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.zeros_(self.attention.in_proj_bias)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        self.fc1 = _linear(hidden_dim, hidden_dim * 4)
        self.fc2 = _linear(hidden_dim * 4, hidden_dim)
        self.layer_norms = nn.ModuleList(
            (nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim))
        )
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        source: Tensor | None,
        *,
        attention_mask: Tensor,
        source_padding_mask: Tensor,
    ) -> Tensor:
        residual = query
        normalized_query = self.layer_norms[0](query)
        if source is None:
            normalized_source = normalized_query
        else:
            # The official implementation applies the same first LayerNorm to
            # query, key, and value.
            normalized_source = self.layer_norms[0](source)
        attended, _ = self.attention(
            normalized_query,
            normalized_source,
            normalized_source,
            attn_mask=attention_mask,
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        query = residual + F.dropout(attended, p=self.dropout, training=self.training)

        residual = query
        query = self.layer_norms[1](query)
        query = F.relu(self.fc1(query))
        query = F.dropout(query, p=self.dropout, training=self.training)
        query = self.fc2(query)
        query = F.dropout(query, p=self.dropout, training=self.training)
        return residual + query


class CrossModalTransformer(nn.Module):
    """Stack with fixed cross-modal keys/values, as in the official MulT."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        layers: int,
        dropout: float,
        position_encoding: SinusoidalPositionEncoding,
    ) -> None:
        super().__init__()
        self.scale = math.sqrt(hidden_dim)
        self.dropout = dropout
        self.position_encoding = position_encoding
        self.layers = nn.ModuleList(
            CrossModalTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _embed(self, inputs: Tensor) -> Tensor:
        inputs = self.position_encoding(inputs * self.scale)
        return F.dropout(inputs, p=self.dropout, training=self.training)

    def forward(
        self,
        query: Tensor,
        source: Tensor | None,
        sequence_mask: Tensor,
    ) -> Tensor:
        query = self._embed(query)
        if source is not None:
            source = self._embed(source)
        steps = query.shape[1]
        source_steps = query.shape[1] if source is None else source.shape[1]
        # All EPT streams are time-aligned.  A standard upper-triangular mask
        # therefore implements past-and-present-only MulT attention.
        attention_mask = torch.triu(
            torch.ones(
                steps,
                source_steps,
                dtype=torch.bool,
                device=query.device,
            ),
            diagonal=1,
        )
        source_padding_mask = ~sequence_mask.bool()
        for layer in self.layers:
            query = layer(
                query,
                source,
                attention_mask=attention_mask,
                source_padding_mask=source_padding_mask,
            )
        return self.final_norm(query)


class MulT(_BaselineBase):
    """All-pairs causal Multimodal Transformer with EPT-Net output heads."""

    MODALITIES = ("time", "spec", "hr", "context")

    def __init__(self, config: EPTNetConfig) -> None:
        super().__init__(config)
        d = config.hidden_dim
        self.position_encoding = SinusoidalPositionEncoding()
        self.cross_modal = nn.ModuleDict()
        for target in self.MODALITIES:
            for source in self.MODALITIES:
                if source != target:
                    self.cross_modal[f"{target}_with_{source}"] = CrossModalTransformer(
                        d,
                        config.num_heads,
                        config.mult_cross_layers,
                        config.dropout,
                        self.position_encoding,
                    )

        target_dim = d * (len(self.MODALITIES) - 1)
        self.memory = nn.ModuleDict(
            {
                target: CrossModalTransformer(
                    target_dim,
                    config.num_heads,
                    config.mult_memory_layers,
                    config.dropout,
                    self.position_encoding,
                )
                for target in self.MODALITIES
            }
        )
        fused_dim = target_dim * len(self.MODALITIES)
        self.residual_projection_1 = nn.Linear(fused_dim, fused_dim)
        self.residual_projection_2 = nn.Linear(fused_dim, fused_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(fused_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout),
        )

    @staticmethod
    def _carry_last_valid(states: Tensor, sequence_mask: Tensor) -> Tensor:
        """Keep padded outputs finite and equal to the final valid state."""
        valid = sequence_mask.bool()
        lengths = valid.sum(dim=1).clamp_min(1)
        gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, states.shape[-1])
        last_valid = states.gather(1, gather_index)
        return torch.where(valid.unsqueeze(-1), states, last_valid)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        sequence_mask = batch["sequence_mask"].bool()
        encoded = dict(zip(self.MODALITIES, self.encode_modalities(batch), strict=True))
        target_states: list[Tensor] = []
        for target in self.MODALITIES:
            cross_modal_states = [
                self.cross_modal[f"{target}_with_{source}"](
                    encoded[target],
                    encoded[source],
                    sequence_mask,
                )
                for source in self.MODALITIES
                if source != target
            ]
            reinforced = torch.cat(cross_modal_states, dim=-1)
            target_states.append(self.memory[target](reinforced, None, sequence_mask))

        fused = torch.cat(target_states, dim=-1)
        projected = self.residual_projection_1(fused)
        projected = F.relu(projected)
        projected = F.dropout(projected, p=self.config.dropout, training=self.training)
        projected = self.residual_projection_2(projected)
        states = self.state_projection(fused + projected)
        states = self._carry_last_valid(states, sequence_mask)
        return self.make_output(states, sequence_mask)


# Alias used by the upstream repository.
MULTModel = MulT


__all__ = ["MulT", "MULTModel"]
