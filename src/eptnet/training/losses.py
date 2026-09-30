from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EPTNetLoss(nn.Module):
    def __init__(self, config: Mapping) -> None:
        super().__init__()
        loss = config.get("loss", config)
        self.cls_weight = float(loss.get("cls_weight", 1.0))
        self.boundary_weight = float(loss.get("boundary_weight", 1.0))
        self.offset_weight = float(loss.get("offset_weight", 0.5))
        self.smooth_weight = float(loss.get("smooth_weight", 0.05))
        self.label_smoothing = float(loss.get("label_smoothing", 0.0))
        weights = loss.get("class_weights")
        if weights is None or weights == "auto":
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        boundary_pos_weight = loss.get("boundary_pos_weight")
        if boundary_pos_weight is None or boundary_pos_weight == "auto":
            self.register_buffer("boundary_pos_weight", None)
        else:
            self.register_buffer(
                "boundary_pos_weight",
                torch.tensor(boundary_pos_weight, dtype=torch.float32),
            )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)

    def forward(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        valid = batch["sequence_mask"].bool()
        if "target_mask" in batch:
            valid = valid & batch["target_mask"].bool()
        if not bool(valid.any().item()):
            raise ValueError("A supervised batch must contain at least one target-valid step")
        class_loss = F.cross_entropy(
            outputs["class_logits"].transpose(1, 2),
            batch["labels"].long(),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        class_loss = self._masked_mean(class_loss, valid)

        boundary_loss = F.binary_cross_entropy_with_logits(
            outputs["boundary_logits"],
            batch["boundaries"].float(),
            pos_weight=self.boundary_pos_weight,
            reduction="none",
        )
        boundary_loss = self._masked_mean(boundary_loss, valid)

        positive = valid & batch["positive_mask"].bool()
        offset_loss = F.smooth_l1_loss(
            outputs["offsets"], batch["offsets"].float(), reduction="none"
        )
        offset_loss = self._masked_mean(offset_loss, positive)

        probabilities = torch.softmax(outputs["class_logits"], dim=-1)
        if probabilities.shape[1] > 1:
            # Only suppress unsupported flicker. Penalizing across a supervised class
            # transition would directly fight the boundary and classification targets.
            same_target = batch["labels"][:, 1:] == batch["labels"][:, :-1]
            pair_mask = valid[:, 1:] & valid[:, :-1] & same_target
            smooth = (probabilities[:, 1:] - probabilities[:, :-1]).abs().mean(dim=-1)
            smooth_loss = self._masked_mean(smooth, pair_mask)
        else:
            smooth_loss = class_loss.new_zeros(())

        total = (
            self.cls_weight * class_loss
            + self.boundary_weight * boundary_loss
            + self.offset_weight * offset_loss
            + self.smooth_weight * smooth_loss
        )
        return {
            "total": total,
            "classification": class_loss.detach(),
            "boundary": boundary_loss.detach(),
            "offset": offset_loss.detach(),
            "smooth": smooth_loss.detach(),
        }
