"""ATER: event-conditioned lag/span reading on native observation timestamps."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .memory import BRANCHES, RepresentationMemory


class TemporalEvidenceEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.position = nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, summaries: Tensor) -> Tensor:
        return self.norm(self.encoder(summaries + self.position).mean(dim=1))


class AdaptiveTemporalReader(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4,
                 dropout: float = 0.2, adaptive: bool = True,
                 shared_policy: bool = False, event_guidance: bool = True,
                 state_feedback: bool = True, fixed_width: float = 0.25) -> None:
        super().__init__()
        self.adaptive = adaptive
        self.shared_policy = shared_policy
        self.event_guidance = event_guidance
        self.state_feedback = state_feedback
        self.fixed_width = fixed_width
        self.condition = nn.Sequential(
            nn.Linear(5 * hidden_dim + 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        policy_names = BRANCHES[:1] if shared_policy else BRANCHES
        self.policy = nn.ModuleDict({name: nn.Linear(hidden_dim, 2) for name in policy_names})
        self.local = nn.ModuleDict({name: TemporalEvidenceEncoder(hidden_dim, num_heads, dropout)
                                   for name in BRANCHES})

    def forward(self, state: Tensor, context: Tensor, latest: dict[str, Tensor],
                anchors: Tensor, memory: RepresentationMemory, now: float,
                enabled: dict[str, bool]) -> tuple[dict[str, Tensor], dict]:
        anchors = anchors if self.event_guidance else torch.zeros_like(anchors)
        feedback = state if self.state_feedback else torch.zeros_like(state)
        condition = self.condition(torch.cat((feedback, context,
            *(latest[name] for name in BRANCHES), anchors), dim=-1))
        transition = 1.0 - (1.0 - anchors[:, 1]) * (1.0 - anchors[:, 2])
        evidence, details = {}, {}
        for name in BRANCHES:
            history = memory.history(name, now)
            if not enabled[name] or history is None or history.values.shape[0] == 0:
                evidence[name] = torch.zeros_like(state)
                details[name] = {
                    "valid": False, "lag": state.new_zeros(1),
                    "span": state.new_zeros(1), "centers": state.new_zeros(1, 4),
                    "weights": state.new_zeros(1, 4, 0), "support": state.new_zeros(0, 2),
                    "available": state.new_zeros(0),
                    "indices": torch.empty(0, dtype=torch.long, device=state.device),
                    "interval_seconds": state.new_zeros(1, 2),
                    "lag_seconds": state.new_zeros(1), "span_seconds": state.new_zeros(1)
                }
                continue
            horizon = memory.horizons[name]
            ages = ((now - history.support[:, 1]) / horizon).to(state.dtype)
            minimum, maximum = ages.min(), ages.max()
            available_range = (maximum - minimum).clamp_min(0.0)
            lower = available_range.clamp_max(0.08)
            upper = available_range.clamp_max(1.0)
            policy_name = BRANCHES[0] if self.shared_policy else name
            lag_score, span_score = self.policy[policy_name](condition).unbind(dim=-1)
            if self.adaptive:
                span = lower + (upper - lower) * torch.sigmoid(span_score + 2 * transition - 1)
                lag = minimum + (available_range - span) * torch.sigmoid(lag_score)
            else:
                span = torch.minimum(available_range, state.new_tensor(self.fixed_width)).reshape(1)
                lag = minimum.reshape(1)
            midpoints = (torch.arange(4, device=state.device, dtype=state.dtype) + 0.5) / 4
            centers = lag[:, None] + span[:, None] * midpoints[None, :]
            scale = (span / 4).clamp_min(1e-3)
            # Gaussian tails are intentionally normalized over the whole valid
            # history, rather than cut off at the nominal interval endpoints.
            logits = -0.5 * ((ages[None, None, :] - centers[:, :, None])
                             / scale[:, None, None]).square()
            weights = torch.softmax(logits, dim=-1)
            selected = weights @ history.values.unsqueeze(0)
            evidence[name] = self.local[name](selected)
            details[name] = {
                "valid": True, "lag": lag, "span": span, "centers": centers,
                "weights": weights, "support": history.support,
                "available": history.available, "indices": history.indices,
                "interval_seconds": torch.stack((now - (lag + span) * horizon,
                                                  now - lag * horizon), dim=-1),
                "lag_seconds": lag * horizon, "span_seconds": span * horizon
            }
        return evidence, details
