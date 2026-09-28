from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


class BatchContractError(ValueError):
    """Raised when a dataset adapter violates the EPT-Net tensor contract."""


COMMON_INPUTS = {
    "video": 3,  # [B, T, D_video]
    "audio": 3,  # [B, T, D_audio]
    "text": 3,  # [B, T, D_text]
    "sequence_mask": 2,  # [B, T]
    "modality_mask": 3,  # [B, T, 3] ordered as video/audio/text
}

RAW_WINDOW_INPUTS = {
    "eeg": 4,  # [B, T, C_eeg, S_eeg]
    "hr": 4,  # [B, T, C_hr, S_hr]
}

PRECOMPUTED_INPUTS = {
    "eeg_time": 3,  # [B, T, D_eeg_time]
    "eeg_spectral": 3,  # [B, T, D_eeg_spectral]
    "physiology": 3,  # [B, T, D_physiology]
}

OPTIONAL_TARGETS = {
    "labels": 2,  # [B, T]
    "boundaries": 3,  # [B, T, 2] start/end
    "offsets": 3,  # [B, T, 2] left/right distance
    "positive_mask": 2,  # [B, T]
}

# Optional supervision/evaluation support.  ``target_mask`` marks timeline
# positions that belong to the evaluated person.  The model may observe the
# full interaction, while losses and reported metrics exclude interviewer,
# background, and unannotated time.  Legacy datasets without this mask retain
# their previous all-valid behavior.
OPTIONAL_DECISION_MASKS = {
    "target_mask": 2,  # [B, T]
}

OPTIONAL_AVAILABILITY = {
    # Per-step availability for [eeg_time, eeg_spectral, physiology].  Older
    # fully observed datasets may omit this key.
    "physiology_mask": 3,
}


def _require_keys(batch: dict[str, Tensor], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in batch]
    if missing:
        raise BatchContractError(f"Missing batch keys: {missing}")


def infer_input_mode(batch: dict[str, Tensor]) -> str:
    if "eeg_time" in batch:
        return "precomputed_features"
    if "eeg" in batch:
        return "raw_windows"
    raise BatchContractError("Cannot infer input mode: expected eeg_time or eeg")


def validate_batch(
    batch: dict[str, Tensor], require_targets: bool = False, input_mode: str | None = None
) -> None:
    """Validate keys, ranks, aligned batch/time axes, masks, and target shapes."""
    input_mode = input_mode or infer_input_mode(batch)
    if input_mode == "raw_windows":
        mode_inputs = RAW_WINDOW_INPUTS
    elif input_mode == "precomputed_features":
        mode_inputs = PRECOMPUTED_INPUTS
    else:
        raise BatchContractError(f"Unknown input mode: {input_mode}")
    required_inputs = {**COMMON_INPUTS, **mode_inputs}
    _require_keys(batch, required_inputs)
    if require_targets:
        _require_keys(batch, OPTIONAL_TARGETS)

    for key, rank in required_inputs.items():
        if not isinstance(batch[key], Tensor) or batch[key].ndim != rank:
            raise BatchContractError(
                f"{key} must be a rank-{rank} Tensor, got {type(batch[key])} "
                f"with shape {getattr(batch[key], 'shape', None)}"
            )

    batch_size, steps = batch["sequence_mask"].shape
    for key in tuple(mode_inputs) + ("video", "audio", "text", "modality_mask"):
        if batch[key].shape[:2] != (batch_size, steps):
            raise BatchContractError(
                f"{key} has leading shape {tuple(batch[key].shape[:2])}; "
                f"expected {(batch_size, steps)}"
            )

    if batch["modality_mask"].shape[-1] != 3:
        raise BatchContractError("modality_mask must use order [video, audio, text]")
    valid_steps = batch["sequence_mask"].bool()
    if not torch.all(valid_steps.any(dim=1)):
        raise BatchContractError("Every sequence must contain at least one valid step")
    if valid_steps.shape[1] > 1 and torch.any(valid_steps[:, 1:] & ~valid_steps[:, :-1]):
        raise BatchContractError(
            "sequence_mask must contain one contiguous valid prefix followed by padding"
        )
    # A real timeline step may contain silence and no observed sensor sample.  Such a
    # step remains valid temporal context but carries all-false availability masks.
    # Encoders must contribute no fabricated *current* evidence; a persistent model
    # may still update by reading genuinely observed evidence from its causal cache.

    for key, rank in OPTIONAL_TARGETS.items():
        if key not in batch:
            continue
        if batch[key].ndim != rank or batch[key].shape[:2] != (batch_size, steps):
            raise BatchContractError(f"Invalid target shape for {key}: {batch[key].shape}")
    for key, rank in OPTIONAL_DECISION_MASKS.items():
        if key not in batch:
            continue
        if batch[key].ndim != rank or batch[key].shape[:2] != (batch_size, steps):
            raise BatchContractError(f"Invalid decision-mask shape for {key}: {batch[key].shape}")
        decision_mask = batch[key].bool()
        if torch.any(decision_mask & ~valid_steps):
            raise BatchContractError(f"{key} cannot mark padded steps as valid")
        if not torch.all(decision_mask.any(dim=1)):
            raise BatchContractError(f"Every sequence must contain at least one {key} step")
    for key in ("boundaries", "offsets"):
        if key in batch and batch[key].shape[-1] != 2:
            raise BatchContractError(f"{key} must end with [start/end] or [left/right]")

    for key, rank in OPTIONAL_AVAILABILITY.items():
        if key not in batch:
            continue
        if batch[key].ndim != rank or batch[key].shape[:2] != (batch_size, steps):
            raise BatchContractError(f"Invalid availability shape for {key}: {batch[key].shape}")
        if batch[key].shape[-1] != 3:
            raise BatchContractError(
                "physiology_mask must use order [eeg_time, eeg_spectral, physiology]"
            )
