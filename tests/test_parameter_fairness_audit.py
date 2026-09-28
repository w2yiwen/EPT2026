from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit"
    / "audit_parameter_fairness.py"
)
SPEC = importlib.util.spec_from_file_location("audit_parameter_fairness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

REPORT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reporting"
    / "build_release_report.py"
)
REPORT_SPEC = importlib.util.spec_from_file_location("build_release_report", REPORT_SCRIPT)
assert REPORT_SPEC is not None and REPORT_SPEC.loader is not None
REPORT = importlib.util.module_from_spec(REPORT_SPEC)
sys.modules[REPORT_SPEC.name] = REPORT
REPORT_SPEC.loader.exec_module(REPORT)


def test_minimal_cover_is_exact_and_lexicographically_deterministic() -> None:
    coverages = {
        "sample_d": ("eeg_time", "hr"),
        "sample_c": ("eeg_time", "hr"),
        "sample_a": ("eeg_time",),
        "sample_b": ("hr",),
    }

    assert AUDIT.select_minimal_cover(coverages, ("eeg_time", "hr")) == ("sample_c",)


def test_minimal_cover_uses_lexicographic_tie_break_for_equal_size_sets() -> None:
    coverages = {
        "sample_d": ("hr",),
        "sample_c": ("eeg_time",),
        "sample_b": ("hr",),
        "sample_a": ("eeg_time",),
    }

    assert AUDIT.select_minimal_cover(coverages, ("eeg_time", "hr")) == (
        "sample_a",
        "sample_b",
    )


def test_minimal_cover_fails_closed_when_a_required_modality_is_absent() -> None:
    with pytest.raises(ValueError, match="hr"):
        AUDIT.select_minimal_cover({"sample_a": ("eeg_time",)}, ("eeg_time", "hr"))


def test_availability_is_counted_only_inside_target_valid_steps() -> None:
    sample = {
        "sample_id": "session_test_window_0000",
        "physiology_mask": torch.tensor(
            [[True, True, False], [False, False, True], [True, True, True]]
        ),
        "modality_mask": torch.tensor(
            [[True, False, False], [False, True, False], [False, False, True]]
        ),
    }
    target_valid = torch.tensor([True, False, False])

    masks = AUDIT._availability_masks(sample, target_valid)

    assert bool(masks["eeg_time"].any())
    assert bool(masks["eeg_spectral"].any())
    assert bool(masks["video"].any())
    assert not bool(masks["hr"].any())
    assert not bool(masks["audio"].any())
    assert not bool(masks["text"].any())


def test_release_report_accepts_the_new_fairness_schema(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "results" / ("parameter_fairness_bci_subjects.json")
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    output = tmp_path / source.name
    output.write_text(json.dumps(payload), encoding="utf-8")

    records, issues = REPORT._load_parameter_fairness(tmp_path)

    assert not issues
    assert set(records) == set(REPORT.PRIMARY_FAIRNESS_EXPECTED)


def test_release_report_rejects_the_ambiguous_legacy_counting_protocol(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "results" / ("parameter_fairness_bci_subjects.json")
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["records"][0]["counting_protocol"] = (
        "parameters receiving gradients from the full multitask loss"
    )
    output = tmp_path / source.name
    output.write_text(json.dumps(payload), encoding="utf-8")

    records, issues = REPORT._load_parameter_fairness(tmp_path)

    assert not records
    assert any("invalid graph protocol" in issue for issue in issues)


def test_release_report_requires_threshold_free_event_ap_protocol() -> None:
    payload = {
        "protocol": {
            "evaluation_unit": "unique chronological rows reconstructed per session",
            "positive_class": 0,
            "frame_threshold": 0.4,
            "boundary_threshold": 0.5,
            "boundary_tolerance_steps": 1,
            "event_iou_thresholds": [0.3, 0.5, 0.7],
            "early_detection_delays_steps": [0.0, 1.0, 2.0],
            "latency_definition": REPORT.LATENCY_DEFINITION,
            "target_mask_definition": REPORT.TARGET_MASK_DEFINITION,
            "event_f1_protocol": REPORT.EVENT_F1_PROTOCOL,
            "event_ap_protocol": REPORT.EVENT_AP_PROTOCOL,
        }
    }

    REPORT._validate_protocol(payload, primary=True, fixed=False, label="valid")
    payload["protocol"]["event_ap_protocol"] = {
        **REPORT.EVENT_AP_PROTOCOL,
        "score_threshold": 0.4,
    }
    with pytest.raises(ValueError, match="event_ap_protocol"):
        REPORT._validate_protocol(payload, primary=True, fixed=False, label="invalid")


def test_release_report_requires_one_dense_proposal_per_target_step() -> None:
    payload = {
        "data": {
            **REPORT.PRIMARY_COUNTS,
            "num_predicted_events": 1,
            "num_dense_event_proposals": REPORT.PRIMARY_COUNTS["num_evaluated_target_steps"] - 1,
        },
        "frame": {"num_frames": REPORT.PRIMARY_COUNTS["num_evaluated_target_steps"]},
    }

    with pytest.raises(ValueError, match="num_dense_event_proposals"):
        REPORT._validate_data(payload, REPORT.PRIMARY_COUNTS, "invalid")
