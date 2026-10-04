from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..encoders import (
    BehaviorContextEncoder,
    EEGSpectralEncoder,
    EEGTimeEncoder,
    FeatureProjectionEncoder,
    HREncoder,
)
from ..eptnet import EPTNetConfig


class _BaselineBase(nn.Module):
    """Shared encoders, modality gates, fusion, and prediction heads."""

    def __init__(self, config: EPTNetConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        self.config = config
        if config.input_mode == "raw_windows":
            self.time_encoder = EEGTimeEncoder(config.eeg_channels, d, config.dropout)
            self.spec_encoder = EEGSpectralEncoder(
                config.eeg_channels, d, config.spectral_n_ffts, config.dropout
            )
            self.hr_encoder = HREncoder(config.hr_channels, d, config.dropout)
        elif config.input_mode == "precomputed_features":
            self.time_encoder = FeatureProjectionEncoder(config.eeg_time_dim, d, config.dropout)
            self.spec_encoder = FeatureProjectionEncoder(config.eeg_spectral_dim, d, config.dropout)
            self.hr_encoder = FeatureProjectionEncoder(config.physiology_dim, d, config.dropout)
        else:
            raise ValueError(f"Unknown input mode: {config.input_mode}")
        self.context_encoder = BehaviorContextEncoder(
            config.video_dim,
            config.audio_dim,
            config.text_dim,
            d,
            config.num_heads,
            config.dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(d * 4, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(config.dropout)
        )
        self.classification_head = nn.Linear(d, config.num_classes)
        self.boundary_head = nn.Linear(d, 2)
        self.offset_head = nn.Linear(d, 2)

    def encode_modalities(self, batch: dict[str, Tensor]) -> tuple[Tensor, ...]:
        """Encode the four shared evidence streams without fusing them."""
        if self.config.input_mode == "raw_windows":
            time_input = batch["eeg"]
            spec_input = batch["eeg"]
            hr_input = batch["hr"]
        else:
            time_input = batch["eeg_time"]
            spec_input = batch["eeg_spectral"]
            hr_input = batch["physiology"]
        modalities = [
            self.time_encoder(time_input),
            self.spec_encoder(spec_input),
            self.hr_encoder(hr_input),
            self.context_encoder(
                batch["video"],
                batch["audio"],
                batch["text"],
                batch["modality_mask"],
                enabled_modalities=(
                    self.config.use_behavior_context and self.config.use_video,
                    self.config.use_behavior_context and self.config.use_audio,
                    self.config.use_behavior_context and self.config.use_text,
                ),
            ),
        ]
        physiology_mask = batch.get("physiology_mask")
        if physiology_mask is not None:
            for index in range(3):
                modalities[index] = modalities[index] * physiology_mask[..., index].unsqueeze(
                    -1
                ).to(modalities[index].dtype)
        switches = (
            self.config.use_eeg_time,
            self.config.use_eeg_spec,
            self.config.use_hr,
            self.config.use_behavior_context,
        )
        modalities = [
            value if enabled else torch.zeros_like(value)
            for value, enabled in zip(modalities, switches, strict=False)
        ]
        return tuple(modalities)

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        return self.fusion(torch.cat(self.encode_modalities(batch), dim=-1))

    def make_output(self, states: Tensor, mask: Tensor) -> dict[str, Any]:
        return {
            "class_logits": self.classification_head(states),
            "boundary_logits": self.boundary_head(states),
            "offsets": F.softplus(self.offset_head(states)),
            "states": states,
            "sequence_mask": mask.bool(),
        }
