"""EPT-Net implementation of the manuscript's native-time streaming protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ater import AdaptiveTemporalReader
from .behavior import EventGuidanceEncoder
from .memory import BRANCHES, RepresentationMemory
from .pmsu import PersistentStateUpdate


class PhysiologicalProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim))

    def forward(self, values: Tensor) -> Tensor:
        return self.network((values - self.mean) / self.scale.clamp_min(1e-6))


@dataclass
class EPTNetState:
    state: Tensor
    memory: RepresentationMemory
    last_time: float = -float("inf")

    def detach(self) -> EPTNetState:
        """Preserve recording history while truncating the previous chunk's graph."""
        self.state = self.state.detach()
        self.memory.detach()
        return self


class EPTNet(nn.Module):
    """One explicit runtime per recording; step() and forward() share one path.

    The public model name, output head names and physiological keys remain
    ``eptnet``, ``class_logits/boundary_logits/offsets`` and ``time/spec/hr``.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        model = config["model"]
        d = int(model.get("hidden_dim", 64))
        heads = int(model.get("num_heads", 4))
        dropout = float(model.get("dropout", 0.2))
        self.hidden_dim = d
        self.horizons = {name: float(model["retention_seconds"][name]) for name in BRANCHES}
        self.enabled = dict(zip(BRANCHES, (bool(model.get("use_eeg_time", True)),
            bool(model.get("use_eeg_spec", True)), bool(model.get("use_hr", True))), strict=True))
        self.behavior_enabled = (bool(model.get("use_video", True)),
                                 bool(model.get("use_audio", True)), bool(model.get("use_text", True)))
        self.persistent = bool(model.get("persistent_state", True))
        self.event_guidance = bool(model.get("event_guidance", True))
        self.projections = nn.ModuleDict({name: PhysiologicalProjection(width, d)
                                         for name, width in (("time", 8), ("spec", 40), ("hr", 1))})
        self.context_encoder = EventGuidanceEncoder(d, heads, dropout,
            int(config["data"].get("num_classes", 2)), int(config["evaluation"].get("positive_class", 0)))
        self.reader = AdaptiveTemporalReader(d, heads, dropout,
            adaptive=bool(model.get("adaptive_reader", True)),
            shared_policy=bool(model.get("shared_read_policy", False)),
            event_guidance=self.event_guidance,
            state_feedback=bool(model.get("state_feedback", True)),
            fixed_width=float(model.get("fixed_read_width", 0.25)))
        self.update = PersistentStateUpdate(d, heads, dropout)
        self.classification_head = nn.Linear(d, int(config["data"].get("num_classes", 2)))
        self.offset_head = nn.Linear(d, 2)

    def set_normalization(self, statistics: Mapping[str, Any]) -> None:
        for name, projection in self.projections.items():
            projection.mean.copy_(torch.as_tensor(statistics[name]["mean"],
                device=projection.mean.device, dtype=projection.mean.dtype))
            projection.scale.copy_(torch.as_tensor(statistics[name]["scale"],
                device=projection.scale.device, dtype=projection.scale.dtype).clamp_min(1e-6))

    def initial_state(self) -> EPTNetState:
        state = self.classification_head.weight.new_zeros(1, self.hidden_dim)
        return EPTNetState(state, RepresentationMemory(self.horizons, state))

    def step(self, decision: Mapping[str, Any], runtime: EPTNetState) -> dict[str, Any]:
        now = float(decision["time"])
        if now <= runtime.last_time:
            raise ValueError("Decisions must advance in session time; initialize a new state for a new recording")
        memory = runtime.memory
        memory.prune(now)
        previous = runtime.state if self.persistent else torch.zeros_like(runtime.state)
        latest = dict(memory.latest)
        latest_valid = dict(memory.latest_valid)
        pending, projected, new_observations = {}, {}, {}
        for name, observations in decision.get("streams", {}).items():
            if name not in BRANCHES or not self.enabled[name]:
                continue
            observations = observations.to(previous.device)
            new_observations[name] = observations
            selected = memory.pending(name, observations, now)
            valid = selected & observations.mask.bool() & torch.isfinite(observations.values).all(dim=-1)
            values = torch.where(valid[:, None], observations.values, torch.zeros_like(observations.values))
            projected[name] = self.projections[name](values)
            pending[name] = selected
            if bool(valid.any()):
                latest[name] = projected[name][valid][-1:]
                latest_valid[name] = True
        physiology_mask = decision.get("physiology_mask", (True, True, True))
        active = {name: self.enabled[name] and bool(physiology_mask[index])
                  for index, name in enumerate(BRANCHES)}
        for name in BRANCHES:
            latest_valid[name] = latest_valid[name] and active[name]
            if not latest_valid[name]:
                latest[name] = torch.zeros_like(previous)
        enabled_behavior = torch.tensor(self.behavior_enabled, device=previous.device)
        mask = decision["modality_mask"].to(previous.device).bool().reshape(1, 3) & enabled_behavior
        behavior = {name: decision[name].to(previous.device).reshape(1, -1)
                    for name in ("video", "audio", "text")}
        mask &= torch.stack([torch.isfinite(behavior[name]).all(dim=-1)
                             for name in ("video", "audio", "text")], dim=-1)
        guidance = self.context_encoder(behavior, mask)
        evidence, details = self.reader(previous, guidance["context"], latest,
                                        guidance["anchors"], memory, now, active)
        tokens = torch.stack((*(evidence[name] for name in BRANCHES),
                              *(latest[name] for name in BRANCHES), guidance["context"]), dim=1)
        token_mask = torch.tensor([[*(details[name]["valid"] for name in BRANCHES),
                                    *(latest_valid[name] for name in BRANCHES),
                                    bool(guidance["context_mask"].item())]], device=previous.device)
        anchors = guidance["anchors"] if self.event_guidance else torch.zeros_like(guidance["anchors"])
        state, gate = self.update(previous, tokens, token_mask, anchors)
        if not bool(decision.get("sequence_mask", True)):
            state, gate = runtime.state, torch.zeros_like(gate)
        runtime.state = state
        runtime.last_time = now
        # HPRM has not seen the new entries during the read or state update.
        if bool(decision.get("sequence_mask", True)):
            for name in pending:
                memory.commit(name, projected[name], new_observations[name], pending[name], now)
        return {"class_logits": self.classification_head(state),
                "av_class_logits": guidance["av_class_logits"], "av_mask": guidance["av_mask"],
                "boundary_logits": guidance["boundary_logits"],
                "offsets": F.softplus(self.offset_head(state)), "states": state,
                "anchors": guidance["anchors"], "read_details": details,
                "read_lags": torch.stack([details[name]["lag"] for name in BRANCHES], dim=-1),
                "read_centers": torch.stack([details[name]["lag"] + details[name]["span"] / 2
                                             for name in BRANCHES], dim=-1),
                "read_widths": torch.stack([details[name]["span"] for name in BRANCHES], dim=-1),
                "write_gates": gate, "retention_gates": 1 - gate, "token_mask": token_mask}

    def forward(self, session, runtime: EPTNetState | None = None,
                start: int = 0, stop: int | None = None, trace: bool = False) -> dict[str, Any]:
        runtime = self.initial_state() if runtime is None else runtime
        stop = len(session) if stop is None else stop
        outputs = []
        device = runtime.state.device
        for index in range(start, stop):
            outputs.append(self.step(session.decision(index, device), runtime))
        if not outputs:
            raise ValueError("A chronological chunk must contain at least one decision")
        keys = ("class_logits", "av_class_logits", "av_mask", "boundary_logits", "offsets",
                "states", "anchors", "read_lags", "read_centers", "read_widths", "write_gates",
                "retention_gates", "token_mask")
        result = {key: torch.stack([output[key] for output in outputs], dim=1) for key in keys}
        result["sequence_mask"] = session.sequence_mask[start:stop].to(device)[None]
        result["target_mask"] = session.target_mask[start:stop].to(device)[None]
        result["runtime"] = runtime
        if trace:
            result["read_details"] = [output["read_details"] for output in outputs]
        return result
