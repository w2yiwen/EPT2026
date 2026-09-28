from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eptnet.data.schema import validate_batch

from .encoders import (
    BehaviorContextEncoder,
    EEGSpectralEncoder,
    EEGTimeEncoder,
    FeatureProjectionEncoder,
    HREncoder,
)
from .reader import EventGuidedReader, PersistentMultimodalUpdate


@dataclass
class EPTNetConfig:
    input_mode: str = "raw_windows"
    eeg_channels: int = 8
    hr_channels: int = 1
    eeg_time_dim: int = 8
    eeg_spectral_dim: int = 40
    physiology_dim: int = 40
    video_dim: int = 64
    audio_dim: int = 48
    text_dim: int = 32
    num_classes: int = 2
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    cache_time: int = 32
    cache_spec: int = 64
    cache_hr: int = 64
    spectral_n_ffts: tuple[int, ...] = (32, 64, 128)
    num_glimpses: int = 4
    min_read_width: float = 0.08
    max_read_width: float = 1.0
    fixed_read_width: float = 0.25
    adaptive_reader: bool = True
    shared_read_policy: bool = False
    persistent_state: bool = True
    state_update: str = "gated"
    write_gate_bias: float = -1.0
    retention_gate_bias: float = 2.0
    latest_state_bypass: bool = True
    use_behavior_context: bool = True
    use_video: bool = True
    use_audio: bool = True
    use_text: bool = False
    use_eeg_time: bool = True
    use_eeg_spec: bool = True
    use_hr: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> EPTNetConfig:
        data = config.get("data", config)
        model = config.get("model", config)
        cache = model.get("cache_lengths", {})
        return cls(
            input_mode=str(data.get("input_mode", "raw_windows")),
            eeg_channels=int(data.get("eeg_channels", 8)),
            hr_channels=int(data.get("hr_channels", 1)),
            eeg_time_dim=int(data.get("eeg_time_dim", 8)),
            eeg_spectral_dim=int(data.get("eeg_spectral_dim", 40)),
            physiology_dim=int(data.get("physiology_dim", 40)),
            video_dim=int(data.get("video_dim", 64)),
            audio_dim=int(data.get("audio_dim", 48)),
            text_dim=int(data.get("text_dim", 32)),
            num_classes=int(data.get("num_classes", 2)),
            hidden_dim=int(model.get("hidden_dim", 64)),
            num_heads=int(model.get("num_heads", 4)),
            dropout=float(model.get("dropout", 0.1)),
            cache_time=int(cache.get("time", 32)),
            cache_spec=int(cache.get("spec", 64)),
            cache_hr=int(cache.get("hr", 64)),
            spectral_n_ffts=tuple(model.get("spectral_n_ffts", (32, 64, 128))),
            num_glimpses=int(model.get("num_glimpses", 4)),
            min_read_width=float(model.get("min_read_width", 0.08)),
            max_read_width=float(model.get("max_read_width", 1.0)),
            fixed_read_width=float(model.get("fixed_read_width", 0.25)),
            adaptive_reader=bool(model.get("adaptive_reader", True)),
            shared_read_policy=bool(model.get("shared_read_policy", False)),
            persistent_state=bool(model.get("persistent_state", True)),
            state_update=str(model.get("state_update", "gated")),
            write_gate_bias=float(model.get("write_gate_bias", -1.0)),
            retention_gate_bias=float(model.get("retention_gate_bias", 2.0)),
            latest_state_bypass=bool(model.get("latest_state_bypass", True)),
            use_behavior_context=bool(model.get("use_behavior_context", True)),
            use_video=bool(model.get("use_video", True)),
            use_audio=bool(model.get("use_audio", True)),
            use_text=bool(model.get("use_text", False)),
            use_eeg_time=bool(model.get("use_eeg_time", True)),
            use_eeg_spec=bool(model.get("use_eeg_spec", True)),
            use_hr=bool(model.get("use_hr", True)),
        )


class EPTNet(nn.Module):
    """Causal EPT-Net with differentiable cache reading and persistent state."""

    def __init__(self, config: EPTNetConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
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
        self.reader = EventGuidedReader(
            d,
            config.num_heads,
            config.num_glimpses,
            config.min_read_width,
            config.max_read_width,
            config.fixed_read_width,
            config.adaptive_reader,
            config.shared_read_policy,
            config.latest_state_bypass,
            config.dropout,
        )
        self.update = PersistentMultimodalUpdate(
            d,
            config.num_heads,
            config.dropout,
            config.persistent_state,
            config.state_update,
            config.write_gate_bias,
            config.retention_gate_bias,
        )
        self.classification_head = nn.Linear(d, config.num_classes)
        self.boundary_head = nn.Linear(d, 2)
        self.offset_head = nn.Linear(d, 2)

    @staticmethod
    def _masked_input(values: Tensor, availability: Tensor) -> Tensor:
        mask = availability
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, values, torch.zeros_like(values))

    def _availability(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        sequence_mask = batch["sequence_mask"].bool()
        physiology_mask = batch.get("physiology_mask")
        if physiology_mask is None:
            physiology_availability = sequence_mask.unsqueeze(-1).expand(-1, -1, 3)
        else:
            physiology_availability = physiology_mask.bool() & sequence_mask.unsqueeze(-1)
        physiology_enabled = torch.tensor(
            (self.config.use_eeg_time, self.config.use_eeg_spec, self.config.use_hr),
            dtype=torch.bool,
            device=sequence_mask.device,
        )
        physiology_availability = physiology_availability & physiology_enabled

        behavior_enabled = torch.tensor(
            (
                self.config.use_behavior_context and self.config.use_video,
                self.config.use_behavior_context and self.config.use_audio,
                self.config.use_behavior_context and self.config.use_text,
            ),
            dtype=torch.bool,
            device=sequence_mask.device,
        )
        behavior_availability = (
            batch["modality_mask"].bool() & sequence_mask.unsqueeze(-1) & behavior_enabled
        )
        context_availability = behavior_availability.any(dim=-1)
        return physiology_availability, behavior_availability, context_availability

    def _encode_inputs(
        self,
        batch: dict[str, Tensor],
        physiology_availability: Tensor,
        behavior_availability: Tensor,
    ) -> dict[str, Tensor]:
        if self.config.input_mode == "raw_windows":
            time_input = self._masked_input(batch["eeg"], physiology_availability[..., 0])
            spec_input = self._masked_input(batch["eeg"], physiology_availability[..., 1])
            hr_input = self._masked_input(batch["hr"], physiology_availability[..., 2])
        else:
            time_input = self._masked_input(batch["eeg_time"], physiology_availability[..., 0])
            spec_input = self._masked_input(batch["eeg_spectral"], physiology_availability[..., 1])
            hr_input = self._masked_input(batch["physiology"], physiology_availability[..., 2])
        behavior_inputs = [
            self._masked_input(batch[name], behavior_availability[..., index])
            for index, name in enumerate(("video", "audio", "text"))
        ]
        features = {
            "time": self.time_encoder(time_input),
            "spec": self.spec_encoder(spec_input),
            "hr": self.hr_encoder(hr_input),
            "context": self.context_encoder(
                *behavior_inputs,
                behavior_availability,
                enabled_modalities=(
                    self.config.use_behavior_context and self.config.use_video,
                    self.config.use_behavior_context and self.config.use_audio,
                    self.config.use_behavior_context and self.config.use_text,
                ),
            ),
        }
        for index, name in enumerate(("time", "spec", "hr")):
            features[name] = torch.where(
                physiology_availability[..., index].unsqueeze(-1),
                features[name],
                torch.zeros_like(features[name]),
            )
        return features

    def _cache_at(
        self,
        sequence: Tensor,
        sequence_mask: Tensor,
        step: int,
        maximum_length: int,
    ) -> tuple[Tensor, Tensor]:
        start = max(0, step - maximum_length + 1)
        return sequence[:, start : step + 1], sequence_mask[:, start : step + 1]

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        validate_batch(batch, require_targets=False, input_mode=self.config.input_mode)
        sequence_mask = batch["sequence_mask"].bool()
        physiology_availability, behavior_availability, context_availability = self._availability(
            batch
        )
        features = self._encode_inputs(batch, physiology_availability, behavior_availability)
        batch_size, steps = sequence_mask.shape
        state = features["context"].new_zeros(batch_size, self.config.hidden_dim)

        states = []
        centers = []
        widths = []
        read_weights: dict[str, list[Tensor]] = {"time": [], "spec": [], "hr": []}
        write_gates = []
        retention_gates = []
        cache_lengths = {
            "time": self.config.cache_time,
            "spec": self.config.cache_spec,
            "hr": self.config.cache_hr,
        }

        for step in range(steps):
            latest = {name: features[name][:, step] for name in ("time", "spec", "hr")}
            caches: dict[str, Tensor] = {}
            cache_masks: dict[str, Tensor] = {}
            for branch_index, name in enumerate(("time", "spec", "hr")):
                caches[name], cache_masks[name] = self._cache_at(
                    features[name],
                    physiology_availability[..., branch_index],
                    step,
                    cache_lengths[name],
                )
            evidence, read_metadata = self.reader(
                state,
                features["context"][:, step],
                latest,
                caches,
                cache_masks,
            )
            tokens = torch.stack(
                [evidence["time"], evidence["spec"], evidence["hr"], features["context"][:, step]],
                dim=1,
            )
            token_mask = torch.stack(
                [
                    cache_masks["time"].any(dim=1),
                    cache_masks["spec"].any(dim=1),
                    cache_masks["hr"].any(dim=1),
                    context_availability[:, step],
                ],
                dim=1,
            )
            candidate, update_metadata = self.update(
                state,
                tokens,
                token_mask=token_mask,
                return_metadata=True,
            )
            valid = sequence_mask[:, step].unsqueeze(-1)
            state = torch.where(valid, candidate, state)
            states.append(state)
            centers.append(read_metadata["centers"])
            widths.append(read_metadata["widths"])
            write_gates.append(update_metadata["write_gate"])
            retention_gates.append(update_metadata["retention_gate"])
            for name in read_weights:
                read_weights[name].append(read_metadata["weights"][name])

        state_sequence = torch.stack(states, dim=1)
        output: dict[str, Any] = {
            "class_logits": self.classification_head(state_sequence),
            "boundary_logits": self.boundary_head(state_sequence),
            "offsets": F.softplus(self.offset_head(state_sequence)),
            "states": state_sequence,
            "read_centers": torch.stack(centers, dim=1),
            "read_widths": torch.stack(widths, dim=1),
            "read_weights": read_weights,
            "write_gates": torch.stack(write_gates, dim=1),
            "retention_gates": torch.stack(retention_gates, dim=1),
            "sequence_mask": sequence_mask,
        }
        return output
