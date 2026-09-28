from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FeatureProjectionEncoder(nn.Module):
    """Project supplied word-level features to the shared evidence space."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class CausalConv1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class EEGTimeEncoder(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        mid = max(hidden_dim // 2, 16)
        self.network = nn.Sequential(
            CausalConv1d(channels, mid, kernel_size=5, dilation=1),
            nn.GELU(),
            nn.GroupNorm(1, mid),
            CausalConv1d(mid, hidden_dim, kernel_size=3, dilation=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, eeg: Tensor) -> Tensor:
        batch, steps, channels, samples = eeg.shape
        features = self.network(eeg.reshape(batch * steps, channels, samples))
        weights = torch.softmax(self.gate(features), dim=-1)
        pooled = (features * weights).sum(dim=-1)
        return pooled.reshape(batch, steps, -1)


class EEGSpectralEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        n_ffts: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        del channels  # Channel aggregation is intentionally permutation-safe at this stage.
        self.n_ffts = tuple(int(value) for value in n_ffts)
        branch_dim = max(hidden_dim // len(self.n_ffts), 8)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, branch_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                for _ in self.n_ffts
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(branch_dim * len(self.n_ffts), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, eeg: Tensor) -> Tensor:
        batch, steps, channels, samples = eeg.shape
        flattened = eeg.reshape(batch * steps * channels, samples)
        branch_outputs = []
        for n_fft, branch in zip(self.n_ffts, self.branches, strict=False):
            signal = flattened
            if signal.shape[-1] < n_fft:
                signal = F.pad(signal, (0, n_fft - signal.shape[-1]))
            window = torch.hann_window(n_fft, device=signal.device, dtype=signal.dtype)
            spectrum = torch.stft(
                signal,
                n_fft=n_fft,
                hop_length=max(n_fft // 4, 1),
                win_length=n_fft,
                window=window,
                center=False,
                return_complex=True,
            )
            magnitude = torch.log1p(spectrum.abs()).unsqueeze(1)
            encoded = branch(magnitude).flatten(1)
            encoded = encoded.reshape(batch * steps, channels, -1).mean(dim=1)
            branch_outputs.append(encoded)
        fused = self.projection(torch.cat(branch_outputs, dim=-1))
        return fused.reshape(batch, steps, -1)


class HREncoder(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.sample_projection = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, hr: Tensor) -> Tensor:
        batch, steps, channels, samples = hr.shape
        sequence = hr.permute(0, 1, 3, 2).reshape(batch * steps, samples, channels)
        sequence = self.sample_projection(sequence)
        _, hidden = self.gru(sequence)
        return hidden[-1].reshape(batch, steps, -1)


class BehaviorContextEncoder(nn.Module):
    MODALITIES = ("video", "audio", "text")

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Linear(video_dim, hidden_dim),
                nn.Linear(audio_dim, hidden_dim),
                nn.Linear(text_dim, hidden_dim),
            ]
        )
        self.modality_embeddings = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        text: Tensor,
        modality_mask: Tensor,
        enabled_modalities: Sequence[bool] = (True, True, True),
    ) -> Tensor:
        if len(enabled_modalities) != len(self.MODALITIES):
            raise ValueError(
                f"Expected {len(self.MODALITIES)} behavior modality switches, "
                f"got {len(enabled_modalities)}"
            )
        enabled = tuple(bool(value) for value in enabled_modalities)
        batch, steps = video.shape[:2]
        projected = [
            (
                layer(value).reshape(batch * steps, -1)
                if is_enabled
                else value.new_zeros(batch * steps, self.modality_embeddings.shape[-1])
            )
            for layer, value, is_enabled in zip(
                self.projections, (video, audio, text), enabled, strict=False
            )
        ]
        tokens = torch.stack(projected, dim=1) + self.modality_embeddings.unsqueeze(0)
        enabled_mask = torch.tensor(enabled, dtype=torch.bool, device=modality_mask.device)
        availability = modality_mask.reshape(batch * steps, 3).bool() & enabled_mask
        no_available = ~availability.any(dim=1)
        safe_availability = availability
        if no_available.any():
            safe_availability = availability.clone()
            safe_availability[no_available, 0] = True
        tokens = tokens * availability.unsqueeze(-1)
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~safe_availability,
            need_weights=False,
        )
        attended = self.norm(tokens + attended)
        denominator = availability.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (attended * availability.unsqueeze(-1)).sum(dim=1) / denominator
        pooled = pooled.masked_fill(no_available.unsqueeze(-1), 0.0)
        return pooled.reshape(batch, steps, -1)
