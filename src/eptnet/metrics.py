from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


# Keep probability scoring finite for JSON serialization while retaining a
# severe penalty for an exactly wrong, fully confident prediction.  This is
# also the clipping convention recorded in the evaluation protocol.
PROBABILITY_EPSILON = 1e-7


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate ``y`` over monotonic ``x`` without a NumPy-version dependency."""
    if x.size < 2:
        return 0.0
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5))


def _binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative TP/FP counts at each distinct score threshold."""
    order = np.argsort(-y_score, kind="mergesort")
    truth = y_true[order].astype(np.int64, copy=False)
    scores = y_score[order]
    cumulative_true = np.cumsum(truth, dtype=np.float64)
    cumulative_false = np.cumsum(1 - truth, dtype=np.float64)
    distinct = np.flatnonzero(np.diff(scores))
    threshold_indices = np.r_[distinct, truth.size - 1]
    return cumulative_true[threshold_indices], cumulative_false[threshold_indices]


def _ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    positives = int(y_true.sum())
    negatives = int(y_true.size - positives)
    true_positives, false_positives = _binary_clf_curve(y_true, y_score)

    true_positive_rate = np.r_[0.0, true_positives / positives]
    false_positive_rate = np.r_[0.0, false_positives / negatives]
    auroc = _trapezoid(true_positive_rate, false_positive_rate)

    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / positives
    recall_with_origin = np.r_[0.0, recall]
    precision_with_origin = np.r_[1.0, precision]
    average_precision = float(np.sum(np.diff(recall_with_origin) * precision))
    auprc = _trapezoid(precision_with_origin, recall_with_origin)
    return auroc, auprc, average_precision


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0


def frame_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
    positive_class: int | None = 0,
) -> dict[str, object]:
    """Compute binary frame metrics for scores of the designated positive class.

    ``probabilities`` must contain the probability of the positive class. The
    repository-wide default is class 0 (``deception``), so labels are converted
    with ``labels == 0``. Pass another class index to evaluate that class, or
    pass ``None`` only when ``labels`` are already a binary positive indicator.

    Brier score and binary negative log-likelihood are proper scoring rules for
    the designated positive-class probability and remain defined for a
    single-class target set. Discrimination metrics that require both target
    classes are returned as ``None`` when only one class is present. Confusion
    counts are always returned.
    """
    scores = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels)
    valid = np.asarray(mask, dtype=bool)
    if scores.shape != targets.shape or scores.shape != valid.shape:
        raise ValueError(
            "probabilities, labels, and mask must have identical shapes; "
            f"got {scores.shape}, {targets.shape}, and {valid.shape}"
        )
    scores = scores[valid]
    targets = targets[valid]
    if not np.isfinite(scores).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must lie within [0, 1]")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite probability within [0, 1]")

    if positive_class is None:
        unique_targets = np.unique(targets)
        if not np.all(np.isin(unique_targets, (0, 1))):
            raise ValueError(
                "labels must be binary when positive_class is omitted; "
                f"got values {unique_targets.tolist()}"
            )
        truth = targets.astype(bool, copy=False)
    else:
        truth = targets == positive_class
    predicted = scores >= threshold

    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    true_negative = int(np.sum(~truth & ~predicted))
    false_negative = int(np.sum(truth & ~predicted))
    result: dict[str, object] = {
        "num_frames": int(truth.size),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }

    if truth.size:
        binary_truth = truth.astype(np.float64, copy=False)
        clipped_scores = np.clip(
            scores,
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )
        result.update(
            {
                "brier_score": float(np.mean((scores - binary_truth) ** 2)),
                "negative_log_likelihood": float(
                    -np.mean(
                        binary_truth * np.log(clipped_scores)
                        + (1.0 - binary_truth) * np.log1p(-clipped_scores)
                    )
                ),
            }
        )
    else:
        result.update({"brier_score": None, "negative_log_likelihood": None})

    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    if not positives or not negatives:
        result.update(
            {
                "macro_f1": None,
                "balanced_accuracy": None,
                "auroc": None,
                "auprc": None,
                "average_precision": None,
            }
        )
        return result

    positive_f1 = _f1(true_positive, false_positive, false_negative)
    negative_f1 = _f1(true_negative, false_negative, false_positive)
    sensitivity = true_positive / positives
    specificity = true_negative / negatives
    auroc, auprc, average_precision = _ranking_metrics(truth.astype(np.int64, copy=False), scores)
    result.update(
        {
            "macro_f1": float((positive_f1 + negative_f1) / 2.0),
            "balanced_accuracy": float((sensitivity + specificity) / 2.0),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "average_precision": float(average_precision),
        }
    )
    return result


def _as_batched_boundaries(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[-1] == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [T,2] or [B,T,2], got {array.shape}")
    return array


def _boundary_counts(
    scores: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    tolerance: int,
) -> tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    for sample_scores, sample_targets, sample_valid in zip(scores, targets, valid, strict=False):
        predicted_indices = np.flatnonzero((sample_scores >= threshold) & sample_valid)
        target_indices = np.flatnonzero(sample_targets.astype(bool) & sample_valid)
        ranked_predictions = sorted(
            predicted_indices.tolist(),
            key=lambda index: (-float(sample_scores[index]), int(index)),
        )
        unmatched = set(map(int, target_indices.tolist()))
        for predicted_index in ranked_predictions:
            candidates = [
                target_index
                for target_index in unmatched
                if abs(target_index - predicted_index) <= tolerance
            ]
            if not candidates:
                false_positive += 1
                continue
            matched_index = min(
                candidates,
                key=lambda target_index: (abs(target_index - predicted_index), target_index),
            )
            unmatched.remove(matched_index)
            true_positive += 1
        false_negative += len(unmatched)
    return true_positive, false_positive, false_negative


def _precision_recall_f1(
    true_positive: int, false_positive: int, false_negative: int
) -> tuple[float | None, float | None, float | None]:
    predicted_count = true_positive + false_positive
    target_count = true_positive + false_negative
    precision = true_positive / predicted_count if predicted_count else None
    recall = true_positive / target_count if target_count else None
    if recall is None:
        f1 = None
    elif precision is None:
        f1 = 0.0
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def boundary_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
    tolerance: int = 0,
) -> dict[str, object]:
    """Match predicted start/end boundaries one-to-one within ``±tolerance`` steps."""
    scores = _as_batched_boundaries(probabilities, "probabilities").astype(np.float64)
    truth = _as_batched_boundaries(targets, "targets")
    if scores.shape != truth.shape:
        raise ValueError(
            f"probabilities and targets must match, got {scores.shape} and {truth.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("boundary probabilities contain non-finite values")
    tolerance = int(tolerance)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite probability within [0, 1]")
    if mask is None:
        valid = np.ones(scores.shape[:2], dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.ndim == 1:
            valid = valid[None, ...]
        if valid.shape != scores.shape[:2]:
            raise ValueError(f"mask must have shape {scores.shape[:2]}, got {valid.shape}")

    result: dict[str, object] = {"boundary_tolerance_steps": tolerance}
    class_f1: list[float] = []
    for boundary_index, boundary_name in enumerate(("start", "end")):
        tp, fp, fn = _boundary_counts(
            scores[..., boundary_index],
            truth[..., boundary_index],
            valid,
            threshold,
            tolerance,
        )
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        result.update(
            {
                f"{boundary_name}_tp": tp,
                f"{boundary_name}_fp": fp,
                f"{boundary_name}_fn": fn,
                f"{boundary_name}_precision": precision,
                f"{boundary_name}_recall": recall,
                f"{boundary_name}_f1": f1,
            }
        )
        if f1 is not None:
            class_f1.append(f1)
    result["boundary_macro_f1"] = float(np.mean(class_f1)) if class_f1 else None
    return result


def interval_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    intersection = max(0.0, right - left + 1.0)
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1.0
    return intersection / max(union, 1e-8)


def _interpolated_average_precision(
    true_flags: Sequence[int], false_flags: Sequence[int], target_count: int
) -> float | None:
    if target_count == 0:
        return None
    if not true_flags:
        return 0.0
    cumulative_true = np.cumsum(np.asarray(true_flags, dtype=np.float64))
    cumulative_false = np.cumsum(np.asarray(false_flags, dtype=np.float64))
    recall = cumulative_true / target_count
    precision = cumulative_true / np.maximum(cumulative_true + cumulative_false, 1.0)

    recall = np.r_[0.0, recall, 1.0]
    precision = np.r_[0.0, precision, 0.0]
    for index in range(precision.size - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.flatnonzero(recall[1:] != recall[:-1]) + 1
    return float(np.sum((recall[changes] - recall[changes - 1]) * precision[changes]))


def _match_events(
    predictions: list[list[dict[str, float]]],
    targets: list[list[dict[str, float]]],
    threshold: float,
) -> dict[str, object]:
    """Match proposals and report non-negative onset-to-alert delays.

    ``emit_step`` remains the decoder's raw first threshold-crossing step.  A
    matched proposal that was already active when its target began therefore
    has zero time-to-detection rather than a negative delay.
    """
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("IoU threshold must be finite and within (0, 1]")
    ranked = sorted(
        (
            (float(proposal["score"]), sample_index, proposal_index, proposal)
            for sample_index, sample_predictions in enumerate(predictions)
            for proposal_index, proposal in enumerate(sample_predictions)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used = [set() for _ in targets]
    true_flags: list[int] = []
    false_flags: list[int] = []
    matched_ious: list[float] = []
    delays: list[float] = []

    for _, sample_index, _, proposal in ranked:
        candidates = [
            (
                interval_iou(
                    (proposal["start"], proposal["end"]),
                    (event["start"], event["end"]),
                ),
                target_index,
            )
            for target_index, event in enumerate(targets[sample_index])
            if target_index not in used[sample_index]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            used[sample_index].add(best_index)
            true_flags.append(1)
            false_flags.append(0)
            matched_ious.append(float(best_iou))
            if "emit_step" in proposal:
                raw_delay = float(
                    proposal["emit_step"] - targets[sample_index][best_index]["start"]
                )
                if not np.isfinite(raw_delay):
                    raise ValueError("event detection delay must be finite")
                delays.append(max(raw_delay, 0.0))
        else:
            true_flags.append(0)
            false_flags.append(1)

    true_positive = int(sum(true_flags))
    false_positive = int(sum(false_flags))
    target_count = sum(len(sample_targets) for sample_targets in targets)
    false_negative = int(target_count - true_positive)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-8)),
        "average_precision": _interpolated_average_precision(true_flags, false_flags, target_count),
        "matched_ious": matched_ious,
        "delays": delays,
        "target_count": int(target_count),
    }


def early_detection_recall(
    predictions: list[list[dict[str, float]]],
    targets: list[list[dict[str, float]]],
    allowed_delays: Iterable[float] = (0.0, 1.0, 2.0),
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Recall of matched GT events detected within a non-negative step delay.

    Time-to-detection is ``max(emit_step - target_start, 0)``.  Thus a matched
    alert that is already active at target onset counts as detection at delay
    zero while its raw ``emit_step`` remains unchanged in decoded events.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same number of sequences")
    delays = tuple(float(value) for value in allowed_delays)
    if not delays:
        raise ValueError("allowed_delays must not be empty")
    if any(not np.isfinite(delay) or delay < 0.0 for delay in delays):
        raise ValueError("allowed_delays must be finite and non-negative")
    matched = _match_events(predictions, targets, float(iou_threshold))
    target_count = int(matched["target_count"])
    matched_delays = list(matched["delays"])
    result: dict[str, object] = {}
    for allowed_delay in delays:
        key = f"early_detection_recall_delay_{allowed_delay:g}_steps"
        result[key] = (
            float(sum(delay <= allowed_delay for delay in matched_delays) / target_count)
            if target_count
            else None
        )
    return result


def event_metrics(
    predictions: list[list[dict[str, float]]],
    targets: list[list[dict[str, float]]],
    thresholds: Iterable[float] = (0.3, 0.5, 0.7),
    *,
    ap_proposals: list[list[dict[str, float]]] | None = None,
) -> dict[str, object]:
    """Report operating-point event metrics and proposal-ranking AP.

    ``predictions`` contains events emitted by the thresholded causal decoder;
    it defines precision, recall, F1, overlap, and latency.  ``ap_proposals`` may
    provide a separate, threshold-free ranked proposal set for AP/mAP.  The
    historical behavior is retained when it is omitted, which keeps this
    low-level metric useful for already-decoded proposal lists.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            "predictions and targets must contain the same number of sequences; "
            f"got {len(predictions)} and {len(targets)}"
        )
    ranked_proposals = predictions if ap_proposals is None else ap_proposals
    if len(ranked_proposals) != len(targets):
        raise ValueError(
            "ap_proposals and targets must contain the same number of sequences; "
            f"got {len(ranked_proposals)} and {len(targets)}"
        )
    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values:
        raise ValueError("thresholds must contain at least one IoU value")

    result: dict[str, object] = {}
    average_precisions: list[float] = []
    reference_matches: dict[str, object] | None = None
    for threshold in threshold_values:
        operating_point = _match_events(predictions, targets, threshold)
        ranked = (
            operating_point
            if ap_proposals is None
            else _match_events(ranked_proposals, targets, threshold)
        )
        suffix = f"{threshold:.1f}"
        result[f"event_f1_iou_{suffix}"] = operating_point["f1"]
        result[f"event_precision_iou_{suffix}"] = operating_point["precision"]
        result[f"event_recall_iou_{suffix}"] = operating_point["recall"]
        result[f"event_ap_iou_{suffix}"] = ranked["average_precision"]
        result[f"event_tp_iou_{suffix}"] = operating_point["true_positive"]
        result[f"event_fp_iou_{suffix}"] = operating_point["false_positive"]
        result[f"event_fn_iou_{suffix}"] = operating_point["false_negative"]
        if ranked["average_precision"] is not None:
            average_precisions.append(float(ranked["average_precision"]))
        if abs(threshold - 0.5) < 1e-12:
            reference_matches = operating_point

    result["event_map"] = float(np.mean(average_precisions)) if average_precisions else None
    if reference_matches is None:
        reference_matches = _match_events(predictions, targets, threshold_values[0])
    matched_ious = list(reference_matches["matched_ious"])
    delays = list(reference_matches["delays"])
    result["mean_matched_iou"] = float(np.mean(matched_ious)) if matched_ious else 0.0
    mean_delay = float(np.mean(delays)) if delays else None
    result["mean_detection_delay"] = mean_delay  # Backward-compatible step-valued key.
    result["mean_detection_delay_steps"] = mean_delay
    return result
