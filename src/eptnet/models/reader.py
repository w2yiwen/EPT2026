from __future__ import annotations

import torch
from torch import Tensor, nn


class DifferentiableIntervalSampler(nn.Module):
    def __init__(self, num_glimpses: int, min_scale: float = 1e-3) -> None:
        super().__init__()
        self.num_glimpses = num_glimpses
        self.min_scale = min_scale

    def forward(
        self,
        cache: Tensor,
        cache_mask: Tensor,
        center: Tensor,
        width: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, length, _ = cache.shape
        positions = torch.linspace(0.0, 1.0, length, device=cache.device, dtype=cache.dtype)
        offsets = torch.linspace(
            -0.5, 0.5, self.num_glimpses, device=cache.device, dtype=cache.dtype
        )
        glimpse_centers = center[:, None] + offsets[None, :] * width[:, None]
        glimpse_centers = glimpse_centers.clamp(0.0, 1.0)
        scale = (width[:, None, None] / max(self.num_glimpses, 1)).clamp_min(self.min_scale)
        logits = -0.5 * ((positions[None, None, :] - glimpse_centers[:, :, None]) / scale).pow(2)
        valid = cache_mask[:, None, :].bool()
        has_evidence = valid.any(dim=-1, keepdim=True)
        masked_logits = logits.masked_fill(~valid, -torch.inf)
        # Softmax over an entirely masked row is undefined.  Give such rows a
        # finite temporary support, then remove it exactly after softmax.  This
        # keeps both the sampled evidence and the diagnostic read weights at
        # zero when a branch has no causal observations in its cache.
        safe_logits = torch.where(has_evidence, masked_logits, torch.zeros_like(masked_logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        selected = torch.einsum("bkl,bld->bkd", weights, cache)
        return selected, weights


class LocalEvidenceEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_glimpses: int,
        dropout: float,
        latest_state_bypass: bool,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.position = nn.Parameter(torch.randn(1, num_glimpses, hidden_dim) * 0.02)
        # Keep the bypass exactly inert when the current branch is unavailable.
        # A bias term would turn a masked all-zero latest feature into a learned
        # constant and violate the missing-value invariance contract.
        self.latest = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.latest_state_bypass = latest_state_bypass
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, selected: Tensor, latest: Tensor) -> Tensor:
        evidence = self.encoder(selected + self.position).mean(dim=1)
        if self.latest_state_bypass:
            evidence = evidence + self.latest(latest)
        return self.norm(evidence)


class EventGuidedReader(nn.Module):
    BRANCHES = ("time", "spec", "hr")

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_glimpses: int,
        min_width: float,
        max_width: float,
        fixed_width: float,
        adaptive: bool,
        shared_policy: bool,
        latest_state_bypass: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.adaptive = adaptive
        self.shared_policy = shared_policy
        self.min_width = min_width
        self.max_width = max_width
        self.fixed_width = fixed_width
        outputs = 2 if shared_policy else 2 * len(self.BRANCHES)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, outputs),
        )
        self.sampler = DifferentiableIntervalSampler(num_glimpses)
        self.local_encoders = nn.ModuleDict(
            {
                name: LocalEvidenceEncoder(
                    hidden_dim,
                    num_heads,
                    num_glimpses,
                    dropout,
                    latest_state_bypass,
                )
                for name in self.BRANCHES
            }
        )

    def _policy_parameters(self, context: Tensor) -> tuple[Tensor, Tensor]:
        batch = context.shape[0]
        if not self.adaptive:
            width = torch.full(
                (batch, len(self.BRANCHES)),
                self.fixed_width,
                dtype=context.dtype,
                device=context.device,
            )
            center = 1.0 - width / 2.0
            return center, width
        raw = self.policy(context)
        if self.shared_policy:
            raw = raw.repeat(1, len(self.BRANCHES))
        raw = raw.reshape(batch, len(self.BRANCHES), 2)
        center = torch.sigmoid(raw[..., 0])
        width = self.min_width + (self.max_width - self.min_width) * torch.sigmoid(raw[..., 1])
        return center, width

    def forward(
        self,
        previous_state: Tensor,
        behavior_context: Tensor,
        latest: dict[str, Tensor],
        caches: dict[str, Tensor],
        cache_masks: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        context = torch.cat(
            [previous_state, behavior_context] + [latest[name] for name in self.BRANCHES],
            dim=-1,
        )
        centers, widths = self._policy_parameters(context)
        evidence: dict[str, Tensor] = {}
        weights: dict[str, Tensor] = {}
        for branch_index, name in enumerate(self.BRANCHES):
            selected, branch_weights = self.sampler(
                caches[name],
                cache_masks[name],
                centers[:, branch_index],
                widths[:, branch_index],
            )
            branch_available = cache_masks[name].bool().any(dim=1)
            encoded = self.local_encoders[name](selected, latest[name])
            evidence[name] = torch.where(
                branch_available.unsqueeze(-1), encoded, torch.zeros_like(encoded)
            )
            weights[name] = branch_weights
        metadata = {"centers": centers, "widths": widths, "weights": weights}
        return evidence, metadata


class PersistentMultimodalUpdate(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        persistent: bool,
        state_update: str = "gated",
        write_gate_bias: float = -1.0,
        retention_gate_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.persistent = persistent
        self.state_update = state_update
        if state_update not in {"vanilla", "gated"}:
            raise ValueError("state_update must be 'vanilla' or 'gated'")
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.write_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.retention_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.write_gate.bias, write_gate_bias)
        nn.init.constant_(self.retention_gate.bias, retention_gate_bias)
        self.current_only = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        previous_state: Tensor,
        tokens: Tensor,
        token_mask: Tensor | None = None,
        *,
        return_metadata: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if token_mask is None:
            token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            token_mask = token_mask.bool()
        has_token = token_mask.any(dim=1)
        safe_token_mask = token_mask
        if not torch.all(has_token):
            safe_token_mask = token_mask.clone()
            safe_token_mask[~has_token, 0] = True
        tokens = torch.where(token_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        if not self.persistent:
            candidate = self.current_only(tokens.flatten(start_dim=1))
            state = torch.where(has_token.unsqueeze(-1), candidate, previous_state)
            metadata = {
                "write_gate": torch.where(
                    has_token.unsqueeze(-1),
                    torch.ones_like(state),
                    torch.zeros_like(state),
                ),
                "retention_gate": torch.zeros_like(state),
            }
            return (state, metadata) if return_metadata else state
        attended, _ = self.attention(
            previous_state.unsqueeze(1),
            tokens,
            tokens,
            key_padding_mask=~safe_token_mask,
            need_weights=False,
        )
        attended = self.attention_norm(attended.squeeze(1))
        updated = self.ffn_norm(attended + self.ffn(attended))
        candidate = self.gru(updated, previous_state)
        if self.state_update == "gated":
            gate_input = torch.cat([updated, previous_state], dim=-1)
            write_gate = torch.sigmoid(self.write_gate(gate_input))
            retention_gate = torch.sigmoid(self.retention_gate(gate_input))
            candidate = write_gate * candidate + (1.0 - write_gate) * (
                retention_gate * previous_state
            )
        else:
            write_gate = torch.ones_like(candidate)
            retention_gate = torch.ones_like(candidate)
        state = torch.where(has_token.unsqueeze(-1), candidate, previous_state)
        metadata = {
            "write_gate": torch.where(
                has_token.unsqueeze(-1), write_gate, torch.zeros_like(write_gate)
            ),
            "retention_gate": torch.where(
                has_token.unsqueeze(-1), retention_gate, torch.zeros_like(retention_gate)
            ),
        }
        return (state, metadata) if return_metadata else state
