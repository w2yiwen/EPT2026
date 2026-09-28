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


def _fairness_payload() -> dict:
    sample_id = "session_fixture_window_0000"
    digest = "0" * 64
    reference_graph = REPORT.PRIMARY_FAIRNESS_EXPECTED["eptnet_bci_subjects_no_text"][
        "graph_participating_parameters"
    ]
    records = []
    for experiment, expected in REPORT.PRIMARY_FAIRNESS_EXPECTED.items():
        graph = expected["graph_participating_parameters"]
        trainable = expected["trainable_parameters"]
        nonzero_parameters = graph - 1
        nonzero_elements = graph - 2
        records.append(
            {
                "experiment": experiment,
                **expected,
                "graph_participating_fraction": graph / trainable,
                "nonzero_gradient_parameters": nonzero_parameters,
                "nonzero_gradient_parameter_fraction": nonzero_parameters / trainable,
                "nonzero_gradient_elements": nonzero_elements,
                "nonzero_gradient_element_fraction": nonzero_elements / trainable,
                "zero_gradient_graph_parameters": 1,
                "mean_audit_loss": 1.0,
                "audit_sample_ids": [sample_id],
                "audit_sample_ids_sha256": digest,
                "manifest_sha256": digest,
                "counting_protocol": REPORT.FAIRNESS_GRAPH_COUNTING_PROTOCOL,
                "nonzero_gradient_parameter_counting_protocol": (
                    REPORT.FAIRNESS_NONZERO_PARAMETER_COUNTING_PROTOCOL
                ),
                "nonzero_gradient_element_counting_protocol": (
                    REPORT.FAIRNESS_NONZERO_ELEMENT_COUNTING_PROTOCOL
                ),
                "executed_parameters": graph,
                "executed_fraction": graph / trainable,
                "executed_parameters_is_alias_of": "graph_participating_parameters",
                "graph_participating_ratio_to_eptnet": graph / reference_graph,
                "executed_parameter_ratio_to_eptnet": graph / reference_graph,
                "nonzero_gradient_parameter_ratio_to_eptnet": (
                    nonzero_parameters / (reference_graph - 1)
                ),
                "nonzero_gradient_element_ratio_to_eptnet": (
                    nonzero_elements / (reference_graph - 2)
                ),
            }
        )
    return {
        "reference_experiment": "eptnet_bci_subjects_no_text",
        "device": "cpu",
        "deterministic_model_seed": 0,
        "selection": {
            "selection_protocol": REPORT.FAIRNESS_SELECTION_PROTOCOL,
            "manifest_path": "data/processed/fixture/manifests/train.jsonl",
            "manifest_sha256": digest,
            "selected_sample_ids": [sample_id],
            "selected_sample_count": 1,
            "selected_sample_ids_sha256": digest,
            "required_cover_modalities": ["eeg_time", "eeg_spectral", "hr", "video"],
            "enabled_but_globally_unavailable_modalities": [],
            "selected_tensor_sha256": {sample_id: digest},
            "selected_samples": [
                {
                    "sample_id": sample_id,
                    "target_valid_steps": 1,
                    "covered_enabled_modalities": [
                        "eeg_time",
                        "eeg_spectral",
                        "hr",
                        "video",
                    ],
                }
            ],
        },
        "records": records,
    }


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
    payload = _fairness_payload()
    output = tmp_path / "parameter_fairness_bci_subjects.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    records, issues = REPORT._load_parameter_fairness(tmp_path)

    assert not issues
    assert set(records) == set(REPORT.PRIMARY_FAIRNESS_EXPECTED)


def test_release_report_rejects_the_ambiguous_legacy_counting_protocol(
    tmp_path: Path,
) -> None:
    payload = _fairness_payload()
    payload["records"][0]["counting_protocol"] = (
        "parameters receiving gradients from the full multitask loss"
    )
    output = tmp_path / "parameter_fairness_bci_subjects.json"
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
