from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from eptnet.config import load_config
from eptnet.evaluate import (
    _dense_event_proposals,
    _mean_delay_seconds,
    _training_config_fingerprint,
    _write_events,
    calibrate_frame_threshold,
    evaluate_sequences,
)


def test_training_config_fingerprint_ignores_run_placement_but_not_model_changes():
    config = load_config("configs/default.yaml")
    same_training = deepcopy(config)
    same_training["experiment"]["seed"] = 314
    same_training["experiment"]["output_dir"] = "results/elsewhere/seed_314"
    same_training["training"]["device"] = "cuda:0"

    changed_model = deepcopy(config)
    changed_model["model"]["hidden_dim"] += 32

    assert _training_config_fingerprint(same_training) == _training_config_fingerprint(config)
    assert _training_config_fingerprint(changed_model) != _training_config_fingerprint(config)


def test_threshold_calibration_is_subject_balanced_with_deterministic_ties():
    logits = torch.log(torch.tensor([[0.7, 0.3], [0.3, 0.7]]))
    sequences = [
        {
            "class_logits": logits.clone(),
            "labels": torch.tensor([0, 1]),
            "target_mask": torch.tensor([True, True]),
        }
        for _ in range(2)
    ]
    selected = calibrate_frame_threshold(
        sequences,
        {
            "positive_class": 0,
            "calibration_min_threshold": 0.4,
            "calibration_max_threshold": 0.6,
            "calibration_steps": 2,
        },
    )

    assert selected["frame_threshold"] == pytest.approx(0.4)
    assert selected["validation_subject_macro_f1"] == pytest.approx(1.0)
    assert selected["validation_subject_macro_f1_sample_std"] == pytest.approx(0.0)
    assert selected["validation_subject_macro_f1_values"] == [1.0, 1.0]
    assert selected["validation_num_subjects"] == 2
    assert selected["tie_break"].endswith("lower threshold")


def test_timestamp_latency_uses_physical_availability_for_overlapping_words():
    proposal = {"start": 1.0, "end": 3.0, "score": 0.9, "emit_step": 0.0}
    predictions = [[proposal]]
    targets = [[{"start": 2.0, "end": 3.0}]]
    sequences = [
        {
            "timestamps": torch.tensor(
                # The earlier word overlaps the target onset, but its decision
                # is not available until the word ends at 2.25 seconds.
                [[0.0, 2.25], [1.0, 2.5], [2.0, 2.75], [3.0, 3.5]]
            )
        }
    ]

    latency = _mean_delay_seconds(predictions, targets, sequences, iou_threshold=0.5)

    assert latency == pytest.approx(0.25)
    assert proposal["emit_step"] == 0.0


def test_timestamp_latency_rejects_non_finite_values():
    predictions = [[{"start": 0.0, "end": 1.0, "score": 0.9, "emit_step": 1.0}]]
    targets = [[{"start": 0.0, "end": 1.0}]]
    sequences = [{"timestamps": torch.tensor([[0.0, 0.5], [1.0, float("nan")]])}]

    with pytest.raises(ValueError, match="must be finite"):
        _mean_delay_seconds(predictions, targets, sequences, iou_threshold=0.5)


def test_timestamp_latency_uses_word_end_for_late_alert():
    predictions = [[{"start": 1.0, "end": 3.0, "score": 0.9, "emit_step": 3.0}]]
    targets = [[{"start": 2.0, "end": 3.0}]]
    sequences = [{"timestamps": torch.tensor([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])}]

    latency = _mean_delay_seconds(predictions, targets, sequences, iou_threshold=0.5)

    assert latency == pytest.approx(1.5)


@pytest.mark.parametrize("invalid_threshold", (0.0, float("nan"), float("inf"), -0.1, 1.1))
def test_timestamp_latency_rejects_invalid_iou_threshold(invalid_threshold):
    with pytest.raises(ValueError, match="finite and within"):
        _mean_delay_seconds([[]], [[]], [], iou_threshold=invalid_threshold)


def test_protocol_documents_non_negative_ttd_and_raw_emit_step():
    config = load_config("configs/default.yaml")
    sequence = {
        "class_logits": torch.tensor([[-4.0, 4.0], [4.0, -4.0], [4.0, -4.0], [-4.0, 4.0]]),
        "boundary_logits": torch.zeros(4, 2),
        "offsets": torch.zeros(4, 2),
        "labels": torch.tensor([1, 0, 0, 1]),
        "boundaries": torch.zeros(4, 2),
        "timestamps": torch.tensor([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]),
    }

    metrics, _, _ = evaluate_sequences([sequence], config, frame_threshold=0.5)

    definition = metrics["protocol"]["latency_definition"]
    assert "non-negative time-to-detection" in definition
    assert "raw emit_step" in definition
    assert "word-end decision time" in definition
    assert metrics["event"]["mean_detection_delay_steps"] >= 0.0
    assert metrics["latency"]["mean_detection_delay_seconds"] >= 0.0


def test_probability_protocol_and_subject_macro_use_class_zero_event_probability():
    config = load_config("configs/default.yaml")
    sequence = {
        "class_logits": torch.log(torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])),
        "boundary_logits": torch.zeros(3, 2),
        "offsets": torch.zeros(3, 2),
        "labels": torch.tensor([0, 1, 0]),
        "boundaries": torch.zeros(3, 2),
        "target_mask": torch.ones(3, dtype=torch.bool),
    }

    metrics, _, _ = evaluate_sequences([sequence], config, frame_threshold=0.5)

    expected_brier = np.mean((np.array([0.8, 0.3, 0.6]) - np.array([1.0, 0.0, 1.0])) ** 2)
    assert metrics["protocol"]["positive_class"] == 0
    assert metrics["protocol"]["class_encoding"] == {"0": "deception", "1": "truth"}
    assert metrics["protocol"]["probability_metrics"]["threshold_independent"] is True
    assert metrics["frame"]["brier_score"] == pytest.approx(expected_brier)
    assert metrics["subject_macro"]["metrics"]["frame_brier_score"]["mean"] == pytest.approx(
        expected_brier
    )
    assert metrics["subject_macro"]["metrics"]["frame_negative_log_likelihood"][
        "mean"
    ] == pytest.approx(metrics["frame"]["negative_log_likelihood"])


def test_evaluation_excludes_non_target_steps_and_preserves_mask_gaps():
    config = load_config("configs/default.yaml")
    sequence = {
        "class_logits": torch.tensor([[4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [-4.0, 4.0]]),
        "boundary_logits": torch.zeros(4, 2),
        "offsets": torch.zeros(4, 2),
        "labels": torch.tensor([0, 1, 0, 1]),
        "boundaries": torch.zeros(4, 2),
        "target_mask": torch.tensor([True, False, True, True]),
        "timestamps": torch.tensor([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]),
    }

    metrics, predictions, targets = evaluate_sequences([sequence], config, frame_threshold=0.5)

    assert metrics["frame"]["num_frames"] == 3
    assert metrics["data"]["num_unique_steps"] == 4
    assert metrics["data"]["num_evaluated_target_steps"] == 3
    assert len(targets[0]) == 2
    assert targets[0][0] == {"start": 0.0, "end": 0.0}
    assert targets[0][1] == {"start": 2.0, "end": 2.0}
    assert all(event["emit_step"] != 1.0 for event in predictions[0])
    assert metrics["subject_macro"]["num_subjects"] == 1


def test_dense_event_ap_is_frame_threshold_free_while_event_f1_is_not():
    config = load_config("configs/default.yaml")
    probabilities = torch.tensor([[0.1, 0.9], [0.4, 0.6], [0.4, 0.6], [0.1, 0.9]])
    sequence = {
        "class_logits": torch.log(probabilities),
        "boundary_logits": torch.zeros(4, 2),
        "offsets": torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
        "labels": torch.tensor([1, 0, 0, 1]),
        "boundaries": torch.zeros(4, 2),
        "target_mask": torch.ones(4, dtype=torch.bool),
    }

    low, low_predictions, _ = evaluate_sequences([sequence], config, frame_threshold=0.3)
    high, high_predictions, _ = evaluate_sequences([sequence], config, frame_threshold=0.5)

    assert len(low_predictions[0]) == 1
    assert high_predictions[0] == []
    assert low["event"]["event_f1_iou_0.5"] == pytest.approx(1.0)
    assert high["event"]["event_f1_iou_0.5"] == pytest.approx(0.0)
    for key in ("event_ap_iou_0.3", "event_ap_iou_0.5", "event_ap_iou_0.7", "event_map"):
        assert low["event"][key] == pytest.approx(high["event"][key])
    assert low["subject_macro"]["metrics"]["event_map"]["mean"] == pytest.approx(
        high["subject_macro"]["metrics"]["event_map"]["mean"]
    )
    assert low["protocol"]["event_ap_protocol"]["score_threshold"] is None
    assert low["protocol"]["event_ap_protocol"]["nms"] == "none"
    assert low["data"]["num_dense_event_proposals"] == 4


def test_dense_event_proposals_never_cross_target_mask_gap():
    sequence = {
        "class_logits": torch.zeros(5, 2),
        "offsets": torch.full((5, 2), 100.0),
        "labels": torch.tensor([0, 0, 1, 0, 0]),
        "target_mask": torch.tensor([True, True, False, True, True]),
    }

    proposals = _dense_event_proposals(sequence, positive_class=0)

    assert [proposal["anchor_step"] for proposal in proposals] == [0.0, 1.0, 3.0, 4.0]
    assert all(proposal["end"] <= 1.0 for proposal in proposals[:2])
    assert all(proposal["start"] >= 3.0 for proposal in proposals[2:])


def test_events_json_preserves_raw_pre_onset_emit_step(tmp_path):
    predictions = [[{"start": 4.0, "end": 7.0, "score": 0.9, "emit_step": 2.0}]]
    targets = [[{"start": 5.0, "end": 7.0}]]
    output = tmp_path / "events.json"

    _write_events(output, predictions, targets)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted["predictions"][0][0]["emit_step"] == 2.0
    assert predictions[0][0]["emit_step"] == 2.0
