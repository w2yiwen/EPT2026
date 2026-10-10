from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from eptnet.config import load_config
from eptnet.data import StitchedManifestDataset, collate_multimodal
from eptnet.models import build_model
from eptnet.provenance import portable_path, training_config_fingerprint
from eptnet.training.loop import move_to_device
from eptnet.training.trainer import build_runtime_provenance, resolve_device
from .decoding import decode_events
from .metrics import (
    PROBABILITY_EPSILON,
    boundary_metrics,
    early_detection_recall,
    event_metrics,
    frame_metrics,
    interval_iou,
)

EVENT_F1_PROPOSAL_SOURCE = "thresholded causal event decoder"
EVENT_AP_PROPOSAL_SOURCE = "one dense proposal per target-valid step"
EVENT_AP_SCORE = "positive-class probability at the proposal step"
EVENT_AP_INTERVAL = (
    "[step - predicted left offset, step + predicted right offset], clamped to "
    "the proposal step's contiguous target-valid segment"
)
EVENT_AP_RANKING = (
    "global descending score; deterministic sequence and chronological-step tie-break"
)


def _training_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Backward-compatible alias for the canonical provenance helper."""

    return training_config_fingerprint(config)


def _events_from_labels(
    labels: torch.Tensor,
    positive_class: int,
    target_mask: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    valid = torch.ones_like(labels, dtype=torch.bool) if target_mask is None else target_mask.bool()
    if valid.shape != labels.shape:
        raise ValueError("target_mask must have the same shape as labels")
    indices = torch.nonzero(labels.eq(positive_class) & valid, as_tuple=False).flatten().tolist()
    events: list[dict[str, float]] = []
    if not indices:
        return events
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            events.append({"start": float(start), "end": float(previous)})
            start = index
        previous = index
    events.append({"start": float(start), "end": float(previous)})
    return events


def targets_from_batch(
    batch: Mapping[str, Any], positive_class: int = 0
) -> list[list[dict[str, float]]]:
    all_events = []
    target_masks = batch.get("target_mask", batch["sequence_mask"])
    for labels, mask, target_mask in zip(
        batch["labels"], batch["sequence_mask"], target_masks, strict=False
    ):
        valid_length = int(mask.sum().item())
        all_events.append(
            _events_from_labels(labels[:valid_length], positive_class, target_mask[:valid_length])
        )
    return all_events


@torch.no_grad()
def collect_sequences(
    model, loader: Iterable, device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    sequences: list[dict[str, Any]] = []
    inference_seconds = 0.0
    total_steps = 0
    total_target_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        outputs = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started

        for batch_index in range(batch["sequence_mask"].shape[0]):
            length = int(batch["sequence_mask"][batch_index].sum().item())
            total_steps += length
            target_mask = (
                raw_batch["target_mask"][batch_index, :length].bool().cpu()
                if "target_mask" in raw_batch
                else torch.ones(length, dtype=torch.bool)
            )
            total_target_steps += int(target_mask.sum().item())
            sequence: dict[str, Any] = {
                "sample_id": raw_batch["sample_id"][batch_index],
                "metadata": raw_batch["metadata"][batch_index],
                "class_logits": outputs["class_logits"][batch_index, :length].detach().cpu(),
                "boundary_logits": outputs["boundary_logits"][batch_index, :length].detach().cpu(),
                "offsets": outputs["offsets"][batch_index, :length].detach().cpu(),
                "labels": raw_batch["labels"][batch_index, :length].cpu(),
                "boundaries": raw_batch["boundaries"][batch_index, :length].cpu(),
                "target_mask": target_mask,
                "row_indices": (
                    raw_batch["row_indices"][batch_index, :length].cpu()
                    if "row_indices" in raw_batch
                    else torch.arange(length)
                ),
                "timestamps": (
                    raw_batch["timestamps"][batch_index, :length].cpu()
                    if "timestamps" in raw_batch
                    else None
                ),
            }
            for key in ("read_centers", "read_widths"):
                if key in outputs:
                    sequence[key] = outputs[key][batch_index, :length].detach().cpu()
            sequences.append(sequence)

    if not sequences:
        raise RuntimeError("Evaluation DataLoader produced no sequences")
    efficiency = {
        "inference_seconds": inference_seconds,
        "num_steps": total_steps,
        "num_target_steps": total_target_steps,
        "steps_per_second": total_steps / max(inference_seconds, 1e-12),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    return sequences, efficiency


def _sequence_target_mask(sequence: Mapping[str, Any]) -> torch.Tensor:
    labels = sequence["labels"]
    mask = sequence.get("target_mask")
    if mask is None:
        return torch.ones(labels.shape[0], dtype=torch.bool)
    mask = mask.bool()
    if mask.shape != labels.shape:
        raise ValueError("A sequence target_mask must have the same shape as labels")
    return mask


def _frame_arrays(
    sequences: list[dict[str, Any]], positive_class: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = []
    labels = []
    masks = []
    for sequence in sequences:
        probability = torch.softmax(sequence["class_logits"], dim=-1)[:, positive_class]
        probabilities.append(probability.numpy())
        labels.append(sequence["labels"].numpy())
        masks.append(_sequence_target_mask(sequence).numpy())
    return np.concatenate(probabilities), np.concatenate(labels), np.concatenate(masks)


def calibrate_frame_threshold(
    sequences: list[dict[str, Any]], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    positive_class = int(evaluation["positive_class"])
    thresholds = np.linspace(
        float(evaluation.get("calibration_min_threshold", 0.1)),
        float(evaluation.get("calibration_max_threshold", 0.9)),
        int(evaluation.get("calibration_steps", 81)),
    )
    candidates = []
    for threshold in thresholds:
        subject_scores = []
        for sequence in sequences:
            probability = torch.softmax(sequence["class_logits"], dim=-1)[:, positive_class].numpy()
            metrics = frame_metrics(
                probability,
                sequence["labels"].numpy(),
                _sequence_target_mask(sequence).numpy(),
                threshold=float(threshold),
                positive_class=positive_class,
            )
            score = metrics["macro_f1"]
            if score is None:
                raise RuntimeError(
                    "Participant-balanced threshold calibration requires both classes "
                    "for every validation subject"
                )
            subject_scores.append(float(score))
        mean_score = float(np.mean(subject_scores))
        candidates.append(
            (
                mean_score,
                -abs(float(threshold) - 0.5),
                -float(threshold),
                float(threshold),
                subject_scores,
            )
        )
    if not candidates:
        raise RuntimeError("Cannot calibrate a frame threshold without validation subjects")
    score, _, _, threshold, subject_scores = max(candidates)
    return {
        "frame_threshold": threshold,
        "validation_subject_macro_f1": score,
        "validation_subject_macro_f1_sample_std": (
            float(np.std(subject_scores, ddof=1)) if len(subject_scores) > 1 else None
        ),
        "validation_subject_macro_f1_values": subject_scores,
        "validation_num_subjects": len(subject_scores),
        "tie_break": "highest mean, then nearest 0.5, then lower threshold",
    }


def _padded_boundary_arrays(
    sequences: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = max(sequence["boundaries"].shape[0] for sequence in sequences)
    scores = np.zeros((len(sequences), maximum, 2), dtype=np.float64)
    targets = np.zeros((len(sequences), maximum, 2), dtype=np.float64)
    mask = np.zeros((len(sequences), maximum), dtype=bool)
    for index, sequence in enumerate(sequences):
        length = sequence["boundaries"].shape[0]
        scores[index, :length] = torch.sigmoid(sequence["boundary_logits"]).numpy()
        targets[index, :length] = sequence["boundaries"].numpy()
        mask[index, :length] = _sequence_target_mask(sequence).numpy()
    return scores, targets, mask


def _decode_sequence(
    sequence: Mapping[str, Any],
    positive_class: int,
    frame_threshold: float,
    boundary_threshold: float,
) -> list[dict[str, float]]:
    length = sequence["class_logits"].shape[0]
    outputs = {
        "class_logits": sequence["class_logits"].unsqueeze(0),
        "boundary_logits": sequence["boundary_logits"].unsqueeze(0),
        "offsets": sequence["offsets"].unsqueeze(0),
        "sequence_mask": torch.ones(1, length, dtype=torch.bool),
        "target_mask": _sequence_target_mask(sequence).unsqueeze(0),
    }
    return decode_events(
        outputs,
        positive_class=positive_class,
        frame_threshold=frame_threshold,
        boundary_threshold=boundary_threshold,
    )[0]


def _dense_event_proposals(
    sequence: Mapping[str, Any], positive_class: int
) -> list[dict[str, float]]:
    """Create one threshold-free offset proposal at every target-valid step.

    Proposals are deliberately not filtered or suppressed.  Each interval is
    clamped to the contiguous target-valid segment containing its anchor, so an
    interviewer/background gap can never be bridged by a large predicted offset.
    """

    logits = sequence["class_logits"]
    offsets = sequence["offsets"]
    if logits.ndim != 2:
        raise ValueError(f"class_logits must have shape [T,C], got {tuple(logits.shape)}")
    if offsets.shape != (logits.shape[0], 2):
        raise ValueError(
            f"offsets must have shape {(logits.shape[0], 2)}, got {tuple(offsets.shape)}"
        )
    if positive_class < 0 or positive_class >= logits.shape[1]:
        raise ValueError("positive_class is outside the class-logit dimension")
    probabilities = torch.softmax(logits, dim=-1)[:, positive_class]
    if not torch.isfinite(probabilities).all() or not torch.isfinite(offsets).all():
        raise ValueError("Dense event proposals require finite probabilities and offsets")
    if torch.any(offsets < 0):
        raise ValueError("Dense event proposals require non-negative left/right offsets")

    valid = _sequence_target_mask(sequence)
    proposals: list[dict[str, float]] = []
    step = 0
    length = int(logits.shape[0])
    while step < length:
        if not bool(valid[step]):
            step += 1
            continue
        segment_start = step
        while step + 1 < length and bool(valid[step + 1]):
            step += 1
        segment_end = step
        for anchor in range(segment_start, segment_end + 1):
            left = float(offsets[anchor, 0])
            right = float(offsets[anchor, 1])
            proposals.append(
                {
                    "start": max(float(segment_start), float(anchor) - left),
                    "end": min(float(segment_end), float(anchor) + right),
                    "score": float(probabilities[anchor]),
                    "anchor_step": float(anchor),
                }
            )
        step += 1
    return proposals


def _mean_delay_seconds(
    predictions: list[list[dict[str, float]]],
    targets: list[list[dict[str, float]]],
    sequences: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> float | None:
    """Mean non-negative onset-to-alert time for matched timestamped events.

    A word-level alert becomes available at the emitting word's end timestamp.
    Latency is therefore that availability time minus target onset, clamped at
    zero.  Step ordering alone is insufficient because adjacent words may have
    overlapping timestamp intervals.
    """
    iou_threshold = float(iou_threshold)
    if not np.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be finite and within (0, 1]")
    ranked = sorted(
        (
            (float(proposal["score"]), sequence_index, proposal)
            for sequence_index, proposals in enumerate(predictions)
            for proposal in proposals
        ),
        key=lambda item: -item[0],
    )
    used = [set() for _ in targets]
    delays = []
    for _, sequence_index, proposal in ranked:
        candidates = [
            (
                interval_iou(
                    (proposal["start"], proposal["end"]),
                    (target["start"], target["end"]),
                ),
                target_index,
            )
            for target_index, target in enumerate(targets[sequence_index])
            if target_index not in used[sequence_index]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou < iou_threshold:
            continue
        used[sequence_index].add(best_index)
        timestamps = sequences[sequence_index].get("timestamps")
        if timestamps is None:
            continue
        raw_emit_step = float(proposal["emit_step"])
        raw_target_start = float(targets[sequence_index][best_index]["start"])
        if not np.isfinite(raw_emit_step) or not np.isfinite(raw_target_start):
            raise ValueError("event emit and target start steps must be finite")
        emit_step = int(raw_emit_step)
        target_start = int(raw_target_start)
        # A word-level decision becomes available at the observed word's end.
        raw_delay = float(timestamps[emit_step, 1] - timestamps[target_start, 0])
        if not np.isfinite(raw_delay):
            raise ValueError("timestamp-based event detection delay must be finite")
        delays.append(max(raw_delay, 0.0))
    return float(np.mean(delays)) if delays else None


def _subject_summary(values: list[float | None]) -> dict[str, float | int | None]:
    observed = np.asarray([value for value in values if value is not None], dtype=np.float64)
    missing = len(values) - int(observed.size)
    if not observed.size:
        return {
            "mean": None,
            "sample_std": None,
            "bootstrap_ci95_lower": None,
            "bootstrap_ci95_upper": None,
            "n": 0,
            "missing": missing,
        }
    if observed.size == 1:
        lower = upper = float(observed[0])
        sample_std = None
    else:
        # Fixed-seed participant-level nonparametric bootstrap.  This quantifies
        # uncertainty across fixed evaluation sessions, not across model initializations.
        generator = np.random.default_rng(0)
        indices = generator.integers(0, observed.size, size=(10_000, observed.size))
        bootstrap_means = observed[indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap_means, (0.025, 0.975)).tolist()
        sample_std = float(observed.std(ddof=1))
    return {
        "mean": float(observed.mean()),
        "sample_std": sample_std,
        "bootstrap_ci95_lower": float(lower),
        "bootstrap_ci95_upper": float(upper),
        "n": int(observed.size),
        "missing": missing,
    }


def _subject_macro_metrics(
    sequences: list[dict[str, Any]],
    predictions: list[list[dict[str, float]]],
    ap_proposals: list[list[dict[str, float]]],
    targets: list[list[dict[str, float]]],
    *,
    positive_class: int,
    frame_threshold: float,
    boundary_threshold: float,
    boundary_tolerance: int,
    event_iou_thresholds: Iterable[float],
) -> dict[str, Any]:
    collected: dict[str, list[float | None]] = {
        "frame_macro_f1": [],
        "frame_balanced_accuracy": [],
        "frame_auroc": [],
        "frame_average_precision": [],
        "frame_brier_score": [],
        "frame_negative_log_likelihood": [],
        "boundary_macro_f1": [],
        "event_f1_iou_0.5": [],
        "event_map": [],
    }
    for index, sequence in enumerate(sequences):
        probability = torch.softmax(sequence["class_logits"], dim=-1)[:, positive_class].numpy()
        frame = frame_metrics(
            probability,
            sequence["labels"].numpy(),
            _sequence_target_mask(sequence).numpy(),
            threshold=frame_threshold,
            positive_class=positive_class,
        )
        boundary_scores, boundary_targets, boundary_mask = _padded_boundary_arrays([sequence])
        boundary = boundary_metrics(
            boundary_scores,
            boundary_targets,
            mask=boundary_mask,
            threshold=boundary_threshold,
            tolerance=boundary_tolerance,
        )
        event = event_metrics(
            [predictions[index]],
            [targets[index]],
            event_iou_thresholds,
            ap_proposals=[ap_proposals[index]],
        )
        collected["frame_macro_f1"].append(frame["macro_f1"])
        collected["frame_balanced_accuracy"].append(frame["balanced_accuracy"])
        collected["frame_auroc"].append(frame["auroc"])
        collected["frame_average_precision"].append(frame["average_precision"])
        collected["frame_brier_score"].append(frame["brier_score"])
        collected["frame_negative_log_likelihood"].append(
            frame["negative_log_likelihood"]
        )
        collected["boundary_macro_f1"].append(boundary["boundary_macro_f1"])
        collected["event_f1_iou_0.5"].append(event["event_f1_iou_0.5"])
        collected["event_map"].append(event["event_map"])
    return {
        "unit": "fixed evaluation session",
        "num_subjects": len(sequences),
        "uncertainty": "fixed-seed 10000-resample percentile bootstrap across subjects",
        "metrics": {name: _subject_summary(values) for name, values in collected.items()},
    }


def evaluate_sequences(
    sequences: list[dict[str, Any]],
    config: Mapping[str, Any],
    frame_threshold: float,
    efficiency: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[list[dict[str, float]]], list[list[dict[str, float]]]]:
    evaluation = config["evaluation"]
    positive_class = int(evaluation["positive_class"])
    boundary_threshold = float(evaluation["boundary_threshold"])
    probabilities, labels, frame_mask = _frame_arrays(sequences, positive_class)
    frame = frame_metrics(
        probabilities,
        labels,
        frame_mask,
        threshold=frame_threshold,
        positive_class=positive_class,
    )

    boundary_scores, boundary_targets, boundary_mask = _padded_boundary_arrays(sequences)
    boundary = boundary_metrics(
        boundary_scores,
        boundary_targets,
        mask=boundary_mask,
        threshold=boundary_threshold,
        tolerance=int(evaluation.get("boundary_tolerance_steps", 1)),
    )
    predictions = [
        _decode_sequence(
            sequence,
            positive_class,
            frame_threshold,
            boundary_threshold,
        )
        for sequence in sequences
    ]
    ap_proposals = [_dense_event_proposals(sequence, positive_class) for sequence in sequences]
    targets = [
        _events_from_labels(sequence["labels"], positive_class, _sequence_target_mask(sequence))
        for sequence in sequences
    ]
    events = event_metrics(
        predictions,
        targets,
        evaluation["event_iou_thresholds"],
        ap_proposals=ap_proposals,
    )
    early = early_detection_recall(
        predictions,
        targets,
        allowed_delays=evaluation.get("early_detection_delays", (0, 1, 2)),
        iou_threshold=0.5,
    )
    latency_seconds = _mean_delay_seconds(predictions, targets, sequences, iou_threshold=0.5)
    subject_macro = _subject_macro_metrics(
        sequences,
        predictions,
        ap_proposals,
        targets,
        positive_class=positive_class,
        frame_threshold=frame_threshold,
        boundary_threshold=boundary_threshold,
        boundary_tolerance=int(evaluation.get("boundary_tolerance_steps", 1)),
        event_iou_thresholds=evaluation["event_iou_thresholds"],
    )
    metrics = {
        "protocol": {
            "evaluation_unit": "unique chronological rows reconstructed per session",
            "positive_class": positive_class,
            "class_encoding": {"0": "deception", "1": "truth"},
            "frame_threshold": frame_threshold,
            "boundary_threshold": boundary_threshold,
            "boundary_tolerance_steps": int(evaluation.get("boundary_tolerance_steps", 1)),
            "event_iou_thresholds": [float(value) for value in evaluation["event_iou_thresholds"]],
            "early_detection_delays_steps": [
                float(value) for value in evaluation.get("early_detection_delays", (0, 1, 2))
            ],
            "latency_definition": (
                "non-negative time-to-detection; steps=max(raw emit_step minus target "
                "start step, 0); seconds=max(word-end decision time at raw emit_step "
                "minus target event onset time, 0), including overlapping word "
                "intervals; decoded events retain the raw emit_step"
            ),
            "target_mask_definition": (
                "losses and metrics include only target-speaker intervals; the model "
                "observes the complete interaction timeline; invalid intervals close "
                "and cannot open decoded events"
            ),
            "probability_metrics": {
                "score": "softmax probability assigned to positive_class",
                "target": "1[label == positive_class] on target-valid steps",
                "brier_score": "mean squared error between score and binary target",
                "negative_log_likelihood": (
                    "binary cross-entropy of the score after clipping to "
                    f"[{PROBABILITY_EPSILON:g}, {1.0 - PROBABILITY_EPSILON:g}]"
                ),
                "threshold_independent": True,
            },
            "event_f1_protocol": {
                "proposal_source": EVENT_F1_PROPOSAL_SOURCE,
                "uses_frame_threshold": True,
                "used_for": "event precision, recall, F1, early detection, and latency",
            },
            "event_ap_protocol": {
                "proposal_source": EVENT_AP_PROPOSAL_SOURCE,
                "score": EVENT_AP_SCORE,
                "interval": EVENT_AP_INTERVAL,
                "score_threshold": None,
                "nms": "none",
                "ranking": EVENT_AP_RANKING,
            },
        },
        "data": {
            "num_sequences": len(sequences),
            "num_unique_steps": int(sum(sequence["labels"].numel() for sequence in sequences)),
            "num_evaluated_target_steps": int(
                sum(_sequence_target_mask(sequence).sum().item() for sequence in sequences)
            ),
            "num_target_events": int(sum(len(value) for value in targets)),
            "num_predicted_events": int(sum(len(value) for value in predictions)),
            "num_dense_event_proposals": int(sum(len(value) for value in ap_proposals)),
        },
        "frame": frame,
        "boundary": boundary,
        "event": events,
        "early_detection": early,
        "latency": {"mean_detection_delay_seconds": latency_seconds},
        "subject_macro": subject_macro,
    }
    if efficiency is not None:
        metrics["efficiency"] = dict(efficiency)
    return metrics, predictions, targets


@torch.no_grad()
def evaluate(model, loader, config, device, frame_threshold: float | None = None):
    sequences, efficiency = collect_sequences(model, loader, device)
    threshold = (
        float(config["evaluation"]["frame_threshold"])
        if frame_threshold is None
        else float(frame_threshold)
    )
    metrics, _, _ = evaluate_sequences(sequences, config, threshold, efficiency)
    return metrics


def _validate_checkpoint_config(
    checkpoint_config: Mapping[str, Any], requested_config: Mapping[str, Any]
) -> None:
    if checkpoint_config.get("model") != requested_config.get("model"):
        raise ValueError("Checkpoint model configuration does not match --config")
    ignored_data_keys = {"train_manifest", "val_manifest", "test_manifest"}
    saved_data = {
        key: value
        for key, value in checkpoint_config.get("data", {}).items()
        if key not in ignored_data_keys
    }
    requested_data = {
        key: value
        for key, value in requested_config.get("data", {}).items()
        if key not in ignored_data_keys
    }
    if saved_data != requested_data:
        raise ValueError("Checkpoint data tensor contract does not match --config")


def _validate_checkpoint_provenance(
    checkpoint: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    saved = checkpoint.get("provenance")
    if not isinstance(saved, Mapping):
        raise ValueError("Checkpoint is missing the required provenance record")
    saved_identity = saved.get("provenance_sha256")
    current_identity = current.get("provenance_sha256")
    if not isinstance(saved_identity, str) or saved_identity != current_identity:
        raise ValueError(
            "Checkpoint provenance does not match the current executable source and prepared data"
        )


def _write_predictions(
    path: Path,
    sequences: list[dict[str, Any]],
    positive_class: int,
    frame_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sequence in sequences:
            class_probabilities = torch.softmax(sequence["class_logits"], dim=-1)
            boundary_probabilities = torch.sigmoid(sequence["boundary_logits"])
            target_mask = _sequence_target_mask(sequence)
            for step in range(sequence["labels"].shape[0]):
                target_valid = bool(target_mask[step])
                raw_predicted_positive = bool(
                    class_probabilities[step, positive_class] >= frame_threshold
                )
                record: dict[str, Any] = {
                    "sample_id": sequence["sample_id"],
                    "row_index": int(sequence["row_indices"][step]),
                    "label": int(sequence["labels"][step]),
                    "positive_class": positive_class,
                    "target_valid": target_valid,
                    "positive_probability": float(class_probabilities[step, positive_class]),
                    "raw_predicted_positive": raw_predicted_positive,
                    "predicted_positive": target_valid and raw_predicted_positive,
                    "class_probabilities": [float(value) for value in class_probabilities[step]],
                    "boundary_probabilities": [
                        float(value) for value in boundary_probabilities[step]
                    ],
                    "offsets": [float(value) for value in sequence["offsets"][step]],
                }
                if sequence.get("timestamps") is not None:
                    record["start_time_seconds"] = float(sequence["timestamps"][step, 0])
                    record["end_time_seconds"] = float(sequence["timestamps"][step, 1])
                if "read_centers" in sequence:
                    record["read_centers"] = [
                        float(value) for value in sequence["read_centers"][step]
                    ]
                    record["read_widths"] = [
                        float(value) for value in sequence["read_widths"][step]
                    ]
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _write_events(
    path: Path,
    predictions: list[list[dict[str, float]]],
    targets: list[list[dict[str, float]]],
) -> None:
    """Persist decoded events without rewriting their raw online ``emit_step``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"predictions": predictions, "targets": targets},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EPT-Net on unique session timelines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--predictions-output")
    parser.add_argument("--events-output")
    parser.add_argument("--device", help="Override evaluation device")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument(
        "--skip-provenance-check",
        action="store_true",
        help="Skip checkpoint provenance matching; both checkpoint and evaluation hashes are recorded",
    )
    args = parser.parse_args()

    requested_config = load_config(args.config)
    if requested_config["data"].get("input_mode") == "native_streams":
        from .streaming import evaluate as evaluate_streams

        if args.device is not None:
            requested_config["training"]["device"] = args.device
        output = args.output or str(Path(args.checkpoint).parent / "test_metrics.json")
        metrics = evaluate_streams(requested_config, args.checkpoint, output,
            predictions_output=args.predictions_output, events_output=args.events_output,
            no_calibration=args.no_calibration)
        print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.device is not None:
        requested_config["training"]["device"] = args.device
    device = resolve_device(str(requested_config["training"]["device"]))
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise KeyError("Checkpoint must contain model and resolved config")
    checkpoint_config = checkpoint["config"]
    _validate_checkpoint_config(checkpoint_config, requested_config)
    config = dict(checkpoint_config)
    config["evaluation"] = requested_config["evaluation"]
    config["data"] = dict(checkpoint_config["data"])
    config["data"].update(
        {
            key: requested_config["data"][key]
            for key in ("train_manifest", "val_manifest", "test_manifest")
        }
    )
    provenance = build_runtime_provenance(config)
    checkpoint_provenance = checkpoint.get("provenance")
    if args.skip_provenance_check:
        if not isinstance(checkpoint_provenance, Mapping):
            raise ValueError("Checkpoint is missing the provenance record required for audit")
        print(
            "WARNING: explicitly skipping checkpoint provenance hash matching; "
            f"checkpoint={checkpoint_provenance.get('provenance_sha256')} "
            f"evaluation={provenance['provenance_sha256']}"
        )
        provenance["checkpoint_compatibility"] = {
            "provenance_check_skipped": True,
            "checkpoint_provenance_sha256": checkpoint_provenance.get("provenance_sha256"),
            "evaluation_provenance_sha256": provenance["provenance_sha256"],
            "reason": "explicit --skip-provenance-check CLI option",
        }
    else:
        _validate_checkpoint_provenance(checkpoint, provenance)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    evaluation = config["evaluation"]
    frame_threshold = float(evaluation["frame_threshold"])
    calibration_enabled = (
        bool(evaluation.get("calibrate_frame_threshold", False)) and not args.no_calibration
    )
    calibration: dict[str, Any] = {
        "enabled": calibration_enabled,
        "method": (
            "validation_subject_macro_f1_grid" if calibration_enabled else "fixed_threshold"
        ),
        "selection_split": "validation" if calibration_enabled else None,
        "calibration_min_threshold": (
            float(evaluation.get("calibration_min_threshold", 0.1)) if calibration_enabled else None
        ),
        "calibration_max_threshold": (
            float(evaluation.get("calibration_max_threshold", 0.9)) if calibration_enabled else None
        ),
        "calibration_steps": (
            int(evaluation.get("calibration_steps", 81)) if calibration_enabled else None
        ),
    }
    validation = None
    if calibration_enabled:
        validation = StitchedManifestDataset(config["data"]["val_manifest"], require_targets=True)
        validation.materialize()
    test = StitchedManifestDataset(config["data"]["test_manifest"], require_targets=True)
    test.materialize()
    post_materialization_provenance = build_runtime_provenance(config)
    if post_materialization_provenance["provenance_sha256"] != provenance["provenance_sha256"]:
        raise RuntimeError(
            "Executable source or prepared data changed while materializing evaluation inputs"
        )

    if calibration_enabled:
        assert validation is not None
        validation_loader = DataLoader(
            validation, batch_size=1, shuffle=False, collate_fn=collate_multimodal
        )
        validation_sequences, _ = collect_sequences(model, validation_loader, device)
        selected = calibrate_frame_threshold(validation_sequences, evaluation)
        calibration.update(selected)
        frame_threshold = float(selected["frame_threshold"])
    else:
        calibration["frame_threshold"] = frame_threshold
        calibration["disabled_by_cli"] = bool(args.no_calibration)

    test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_multimodal)
    sequences, efficiency = collect_sequences(model, test_loader, device)
    efficiency["parameter_count"] = parameter_count
    metrics, predictions, targets = evaluate_sequences(
        sequences, config, frame_threshold, efficiency
    )
    fixed_metrics, _, _ = evaluate_sequences(sequences, config, 0.5)
    metrics["fixed_threshold_0_5_sensitivity"] = {
        key: fixed_metrics[key]
        for key in (
            "protocol",
            "data",
            "frame",
            "boundary",
            "event",
            "early_detection",
            "latency",
            "subject_macro",
        )
    }
    metrics["checkpoint"] = {
        "path": portable_path(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "seed": int(checkpoint_config["experiment"]["seed"]),
        "experiment": checkpoint_config["experiment"]["name"],
        "training_config_sha256": _training_config_fingerprint(checkpoint_config),
        "training_provenance_sha256": (
            checkpoint_provenance.get("provenance_sha256")
            if isinstance(checkpoint_provenance, Mapping)
            else None
        ),
        "provenance_sha256": provenance["provenance_sha256"],
    }
    metrics["calibration"] = calibration
    metrics["provenance"] = provenance

    output = Path(args.output) if args.output else checkpoint_path.parent / "test_metrics.json"
    predictions_output = (
        Path(args.predictions_output)
        if args.predictions_output
        else output.with_name(f"{output.stem}_predictions.jsonl")
    )
    events_output = (
        Path(args.events_output)
        if args.events_output
        else output.with_name(f"{output.stem}_events.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    _write_predictions(
        predictions_output,
        sequences,
        positive_class=int(evaluation["positive_class"]),
        frame_threshold=frame_threshold,
    )
    _write_events(events_output, predictions, targets)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
