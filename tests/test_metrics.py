from __future__ import annotations

import numpy as np
import pytest
import torch

from eptnet.decoding import CausalEventDecoder, decode_events
from eptnet.metrics import (
    boundary_metrics,
    early_detection_recall,
    event_metrics,
    frame_metrics,
)


def test_frame_metrics_support_perfect_class_zero_predictions():
    labels = np.array([0, 1, 0, 1, 1, 0])
    class_zero_probability = np.array([0.99, 0.02, 0.90, 0.10, 0.01, 0.80])
    metrics = frame_metrics(
        class_zero_probability,
        labels,
        np.ones_like(labels, dtype=bool),
    )

    assert metrics["true_positive"] == 3
    assert metrics["true_negative"] == 3
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    for name in ("macro_f1", "balanced_accuracy", "auroc", "auprc", "average_precision"):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(
        np.mean((class_zero_probability - (labels == 0)) ** 2)
    )
    assert metrics["negative_log_likelihood"] == pytest.approx(
        -np.mean(
            (labels == 0) * np.log(class_zero_probability)
            + (labels != 0) * np.log1p(-class_zero_probability)
        )
    )


def test_frame_metrics_return_none_for_single_class_targets():
    metrics = frame_metrics(
        np.array([0.9, 0.8, 0.7]),
        np.array([0, 0, 0]),
        np.ones(3, dtype=bool),
    )

    assert metrics["true_positive"] == 3
    assert metrics["true_negative"] == 0
    for name in ("macro_f1", "balanced_accuracy", "auroc", "auprc", "average_precision"):
        assert metrics[name] is None
    assert metrics["brier_score"] == pytest.approx(np.mean((np.array([0.9, 0.8, 0.7]) - 1) ** 2))
    assert metrics["negative_log_likelihood"] == pytest.approx(
        -np.mean(np.log(np.array([0.9, 0.8, 0.7])))
    )


def test_frame_probability_scores_follow_class_zero_semantics_and_mask():
    scores = np.array([0.8, 0.3, 0.6, 0.99])
    labels = np.array([0, 1, 0, 1])
    mask = np.array([True, True, True, False])

    metrics = frame_metrics(scores, labels, mask, positive_class=0)

    truth = np.array([1.0, 0.0, 1.0])
    retained = scores[:3]
    assert metrics["num_frames"] == 3
    assert metrics["brier_score"] == pytest.approx(np.mean((retained - truth) ** 2))
    assert metrics["negative_log_likelihood"] == pytest.approx(
        -np.mean(truth * np.log(retained) + (1.0 - truth) * np.log1p(-retained))
    )


@pytest.mark.parametrize("invalid_probability", (-0.01, 1.01))
def test_frame_probability_scores_reject_values_outside_unit_interval(invalid_probability):
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        frame_metrics(
            np.array([invalid_probability, 0.5]),
            np.array([0, 1]),
            np.ones(2, dtype=bool),
            positive_class=0,
        )


@pytest.mark.parametrize("threshold", (-0.1, 1.1, np.nan, np.inf))
def test_probability_metrics_reject_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="finite probability"):
        frame_metrics(
            np.array([0.2, 0.8]),
            np.array([0, 1]),
            np.ones(2, dtype=bool),
            threshold=threshold,
        )
    with pytest.raises(ValueError, match="finite probability"):
        boundary_metrics(
            np.zeros((1, 2, 2)),
            np.zeros((1, 2, 2)),
            threshold=threshold,
        )


def test_boundary_metrics_match_with_tolerance_one_to_one():
    probabilities = np.zeros((1, 6, 2), dtype=np.float64)
    targets = np.zeros((1, 6, 2), dtype=np.float64)
    targets[0, 2, 0] = 1.0
    targets[0, 4, 1] = 1.0
    probabilities[0, 1, 0] = 0.9  # Start is one step early.
    probabilities[0, 2, 0] = 0.8  # Duplicate cannot match the same GT twice.
    probabilities[0, 5, 1] = 0.9  # End is one step late.

    metrics = boundary_metrics(probabilities, targets, tolerance=1)
    assert metrics["start_tp"] == 1
    assert metrics["start_fp"] == 1
    assert metrics["start_fn"] == 0
    assert metrics["start_precision"] == pytest.approx(0.5)
    assert metrics["start_recall"] == pytest.approx(1.0)
    assert metrics["end_f1"] == pytest.approx(1.0)


def test_causal_decoder_records_the_actual_online_emit_step():
    decoder = CausalEventDecoder(frame_threshold=0.5, boundary_threshold=0.5)
    assert decoder.step(0.1, (0.0, 0.0), (0.0, 0.0), step_index=0) == []
    assert decoder.step(0.8, (0.9, 0.1), (0.0, 2.0), step_index=1) == []
    assert decoder.is_active and decoder.emit_step == 1
    assert decoder.step(0.9, (0.1, 0.9), (1.0, 0.0), step_index=2) == []
    completed = decoder.step(0.2, (0.0, 0.0), (50.0, 50.0), step_index=3)

    assert len(completed) == 1
    assert completed[0]["emit_step"] == 1.0
    assert completed[0]["close_step"] == 3.0
    assert completed[0]["end"] <= 2.0

    # Later outputs cannot revise an event that was already closed.
    decoder.step(0.1, (1.0, 1.0), (100.0, 100.0), step_index=4)
    assert completed[0]["emit_step"] == 1.0
    assert completed[0]["end"] <= 2.0


@pytest.mark.parametrize("threshold", (-0.1, 1.1, np.nan, np.inf))
def test_causal_decoder_rejects_invalid_probability_thresholds(threshold):
    with pytest.raises(ValueError, match="finite probabilities"):
        CausalEventDecoder(frame_threshold=threshold)
    with pytest.raises(ValueError, match="finite probabilities"):
        CausalEventDecoder(boundary_threshold=threshold)


def test_decode_events_replays_the_online_state_machine():
    outputs = {
        "class_logits": torch.tensor([[[-3.0, 3.0], [3.0, -3.0], [3.0, -3.0], [-3.0, 3.0]]]),
        "boundary_logits": torch.tensor([[[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [10.0, 10.0]]]),
        "offsets": torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [99.0, 99.0]]]),
        "sequence_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    event = decode_events(outputs)[0][0]
    assert event["emit_step"] == 1.0
    assert event["close_step"] == 3.0
    assert event["start"] == 1.0
    assert event["end"] == 2.0


def test_target_mask_gap_closes_and_splits_online_events():
    outputs = {
        "class_logits": torch.tensor(
            [[[3.0, -3.0], [3.0, -3.0], [3.0, -3.0], [3.0, -3.0], [3.0, -3.0]]]
        ),
        "boundary_logits": torch.zeros(1, 5, 2),
        "offsets": torch.zeros(1, 5, 2),
        "sequence_mask": torch.ones(1, 5, dtype=torch.bool),
        "target_mask": torch.tensor([[True, True, False, True, True]]),
    }

    events = decode_events(outputs)[0]

    assert len(events) == 2
    assert events[0]["emit_step"] == 0.0
    assert events[0]["close_step"] == 2.0
    assert events[0]["end"] <= 1.0
    assert events[1]["emit_step"] == 3.0
    assert events[1]["end"] >= 3.0


def test_target_mask_segment_clamps_large_left_offset_after_gap():
    outputs = {
        "class_logits": torch.tensor(
            [[[-3.0, 3.0], [-3.0, 3.0], [3.0, -3.0], [3.0, -3.0], [-3.0, 3.0]]]
        ),
        "boundary_logits": torch.zeros(1, 5, 2),
        "offsets": torch.tensor([[[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [0.0, 0.0]]]),
        "sequence_mask": torch.ones(1, 5, dtype=torch.bool),
        "target_mask": torch.tensor([[False, False, True, True, False]]),
    }

    events = decode_events(outputs)[0]

    assert len(events) == 1
    assert events[0]["emit_step"] == 2.0
    assert events[0]["start"] == 2.0
    assert events[0]["end"] <= 3.0


def test_event_average_precision_is_global_and_one_to_one():
    predictions = [
        [
            {"start": 0.0, "end": 2.0, "score": 0.9, "emit_step": 0.0},
            {"start": 0.0, "end": 2.0, "score": 0.8, "emit_step": 0.0},
        ],
        [{"start": 5.0, "end": 7.0, "score": 0.7, "emit_step": 6.0}],
    ]
    targets = [
        [{"start": 0.0, "end": 2.0}],
        [{"start": 5.0, "end": 7.0}],
    ]

    metrics = event_metrics(predictions, targets, thresholds=(0.5,))
    assert metrics["event_tp_iou_0.5"] == 2
    assert metrics["event_fp_iou_0.5"] == 1
    assert metrics["event_fn_iou_0.5"] == 0
    assert metrics["event_ap_iou_0.5"] == pytest.approx(5.0 / 6.0)
    assert metrics["event_map"] == pytest.approx(5.0 / 6.0)
    assert metrics["mean_detection_delay_steps"] == pytest.approx(0.5)

    early = early_detection_recall(
        predictions, targets, allowed_delays=(0.0, 1.0), iou_threshold=0.5
    )
    assert early["early_detection_recall_delay_0_steps"] == pytest.approx(0.5)
    assert early["early_detection_recall_delay_1_steps"] == pytest.approx(1.0)


def test_event_ap_can_use_threshold_free_proposals_separate_from_operating_point():
    targets = [[{"start": 2.0, "end": 4.0}]]
    dense = [[{"start": 2.0, "end": 4.0, "score": 0.49}]]

    metrics = event_metrics([[]], targets, thresholds=(0.5,), ap_proposals=dense)

    assert metrics["event_f1_iou_0.5"] == pytest.approx(0.0)
    assert metrics["event_ap_iou_0.5"] == pytest.approx(1.0)
    assert metrics["event_map"] == pytest.approx(1.0)


def test_pre_onset_alert_has_zero_ttd_without_rewriting_emit_step():
    proposal = {
        "start": 4.0,
        "end": 7.0,
        "score": 0.9,
        "emit_step": 2.0,
    }
    predictions = [[proposal]]
    targets = [[{"start": 5.0, "end": 7.0}]]

    metrics = event_metrics(predictions, targets, thresholds=(0.5,))
    early = early_detection_recall(
        predictions, targets, allowed_delays=(0.0, 1.0), iou_threshold=0.5
    )

    assert metrics["mean_detection_delay"] == pytest.approx(0.0)
    assert metrics["mean_detection_delay_steps"] == pytest.approx(0.0)
    assert early["early_detection_recall_delay_0_steps"] == pytest.approx(1.0)
    assert early["early_detection_recall_delay_1_steps"] == pytest.approx(1.0)
    assert proposal["emit_step"] == 2.0


@pytest.mark.parametrize("invalid_delay", (-1.0, np.nan, np.inf))
def test_early_detection_rejects_invalid_allowed_delay(invalid_delay):
    with pytest.raises(ValueError, match="finite and non-negative"):
        early_detection_recall([[]], [[]], allowed_delays=(invalid_delay,))


def test_event_latency_rejects_non_finite_emit_step():
    predictions = [[{"start": 0.0, "end": 1.0, "score": 0.9, "emit_step": np.nan}]]
    targets = [[{"start": 0.0, "end": 1.0}]]
    with pytest.raises(ValueError, match="detection delay must be finite"):
        event_metrics(predictions, targets, thresholds=(0.5,))


@pytest.mark.parametrize("invalid_threshold", (0.0, np.nan, np.inf, -0.1, 1.1))
def test_event_metrics_reject_invalid_iou_threshold(invalid_threshold):
    with pytest.raises(ValueError, match="finite and within"):
        event_metrics([[]], [[]], thresholds=(invalid_threshold,))


@pytest.mark.parametrize("invalid_threshold", (0.0, np.nan, np.inf, -0.1, 1.1))
def test_early_detection_rejects_invalid_iou_threshold(invalid_threshold):
    with pytest.raises(ValueError, match="finite and within"):
        early_detection_recall([[]], [[]], iou_threshold=invalid_threshold)
