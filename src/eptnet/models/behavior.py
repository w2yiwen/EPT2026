"""Independent audiovisual anchors, followed by text-conditioned context."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MaskedAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, tokens: int, dropout: float) -> None:
        super().__init__()
        self.modality_embeddings = nn.Parameter(torch.randn(1, tokens, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        valid = mask.any(dim=-1)
        tokens = torch.where(mask[..., None], tokens + self.modality_embeddings,
                             torch.zeros_like(tokens))
        safe = mask.clone()
        safe[~valid, 0] = True
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=~safe,
                                     need_weights=False)
        encoded = self.norm(tokens + attended)
        pooled = (encoded * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return torch.where(valid[:, None], pooled, torch.zeros_like(pooled)), valid


class EventGuidanceEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.2,
                 num_classes: int = 2, positive_class: int = 0) -> None:
        super().__init__()
        self.positive_class = positive_class
        self.projections = nn.ModuleDict({name: nn.Linear(width, hidden_dim)
            for name, width in (("video", 384), ("audio", 768), ("text", 768))})
        self.av_fusion = MaskedAttentionFusion(hidden_dim, num_heads, 2, dropout)
        self.context_fusion = MaskedAttentionFusion(hidden_dim, num_heads, 2, dropout)
        self.classification_head = nn.Linear(hidden_dim, num_classes)
        self.boundary_head = nn.Linear(hidden_dim, 2)

    def forward(self, behavior: dict[str, Tensor], mask: Tensor) -> dict[str, Tensor]:
        projected = []
        for index, name in enumerate(("video", "audio", "text")):
            value = torch.where(mask[:, index, None], behavior[name],
                                torch.zeros_like(behavior[name]))
            projected.append(self.projections[name](value))
        av, av_valid = self.av_fusion(torch.stack(projected[:2], dim=1), mask[:, :2])
        av_logits = self.classification_head(av)
        boundary_logits = self.boundary_head(av)
        boundary_logits = torch.where(av_valid[:, None], boundary_logits,
                                      torch.full_like(boundary_logits, -20.0))
        anchors = torch.cat((torch.softmax(av_logits, dim=-1)[:, self.positive_class, None],
                             torch.sigmoid(boundary_logits)), dim=-1)
        anchors = torch.where(av_valid[:, None], anchors, torch.zeros_like(anchors))
        context_mask = torch.stack((av_valid, mask[:, 2]), dim=-1)
        context, context_valid = self.context_fusion(
            torch.stack((av, projected[2]), dim=1), context_mask)
        return {"context": context, "context_mask": context_valid, "anchors": anchors,
                "av_class_logits": av_logits, "av_mask": av_valid,
                "boundary_logits": boundary_logits}
