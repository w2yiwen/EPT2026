from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
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


def _audit_input_identity() -> dict:
    manifest_sha256 = "0" * 64
    payload = {
        "schema_version": AUDIT.AUDIT_INPUT_SCHEMA_VERSION,
        "configs": [
            {
                "path": "configs/model.yaml",
                "file_sha256": "1" * 64,
                "dependency_file_sha256": {
                    "configs/base.yaml": "2" * 64,
                    "configs/model.yaml": "1" * 64,
                },
                "resolved_config_sha256": "3" * 64,
            }
        ],
        "implementation_source_sha256": {
            "scripts/audit/audit_parameter_fairness.py": "4" * 64,
        },
        "implementation_source_bundle_sha256": "5" * 64,
        "train_manifest": {
            "path": "data/processed/fixture/manifests/train.jsonl",
            "sha256": manifest_sha256,
        },
    }
    return {**payload, "bundle_sha256": AUDIT._sha256_json(payload)}


def _reference_for_identity(identity: dict) -> dict:
    manifest_sha256 = identity["train_manifest"]["sha256"]
    return {
        "audit_inputs": deepcopy(identity),
        "selection": {"manifest_sha256": manifest_sha256},
        "records": [
            {
                "config": identity["configs"][0]["path"],
                "manifest_sha256": manifest_sha256,
            }
        ],
    }


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


def test_audit_reference_accepts_only_the_exact_fingerprinted_inputs() -> None:
    identity = _audit_input_identity()

    AUDIT.assert_audit_reference_matches(_reference_for_identity(identity), identity)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("configs", "6" * 64),
        ("implementation_source_sha256", "7" * 64),
        ("train_manifest", "8" * 64),
    ],
)
def test_audit_reference_rejects_config_source_or_manifest_drift(
    field: str, replacement: str
) -> None:
    identity = _audit_input_identity()
    current = deepcopy(identity)
    if field == "configs":
        current["configs"][0]["resolved_config_sha256"] = replacement
    elif field == "implementation_source_sha256":
        current[field]["scripts/audit/audit_parameter_fairness.py"] = replacement
    else:
        current[field]["sha256"] = replacement
    unsigned = {key: value for key, value in current.items() if key != "bundle_sha256"}
    current["bundle_sha256"] = AUDIT._sha256_json(unsigned)

    with pytest.raises(ValueError, match=field):
        AUDIT.assert_audit_reference_matches(_reference_for_identity(identity), current)


def test_audit_reference_rejects_legacy_and_internally_inconsistent_reports() -> None:
    identity = _audit_input_identity()
    with pytest.raises(ValueError, match="legacy or incomplete"):
        AUDIT.assert_audit_reference_matches({}, identity)

    reference = _reference_for_identity(identity)
    reference["selection"]["manifest_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="hashes disagree"):
        AUDIT.assert_audit_reference_matches(reference, identity)


def test_config_dependency_fingerprint_covers_the_inheritance_chain(tmp_path: Path) -> None:
    parent = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("value: 1\n", encoding="utf-8")
    child.write_text("inherits: base.yaml\noverride: 2\n", encoding="utf-8")

    assert AUDIT._config_dependency_paths(child) == (parent.resolve(), child.resolve())


def test_manifest_fingerprint_does_not_expose_a_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    declared_directory = repository / "data" / "manifests"
    declared_directory.mkdir(parents=True)
    private_directory = tmp_path / "private-storage"
    private_directory.mkdir()
    private_manifest = private_directory / "train.jsonl"
    private_manifest.write_text('{"sample_id":"fixture"}\n', encoding="utf-8")
    declared_manifest = declared_directory / "train.jsonl"
    declared_manifest.symlink_to(private_manifest)
    monkeypatch.setattr(AUDIT, "PROJECT_ROOT", repository.resolve())

    identity = AUDIT._manifest_identity(
        [{"data": {"train_manifest": str(declared_manifest)}}]
    )

    assert identity["path"] == "data/manifests/train.jsonl"
    assert str(private_directory) not in json.dumps(identity)
    assert identity["sha256"] == AUDIT._sha256_file(private_manifest)


def test_manifest_fingerprint_rejects_an_external_absolute_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    manifest = tmp_path / "private-train.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(AUDIT, "PROJECT_ROOT", repository.resolve())

    with pytest.raises(ValueError, match="repository-relative"):
        AUDIT._manifest_identity([{"data": {"train_manifest": str(manifest)}}])


def test_implementation_fingerprint_covers_models_and_audit_entrypoint() -> None:
    paths = {AUDIT._portable_repository_path(path) for path in AUDIT._implementation_source_paths()}

    assert "scripts/audit/audit_parameter_fairness.py" in paths
    assert "src/eptnet/models/eptnet.py" in paths
    assert "src/eptnet/data/dataset.py" in paths


def test_check_reference_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _audit_input_identity()
    reference_path = tmp_path / "fairness.json"
    reference_path.write_text(
        json.dumps(_reference_for_identity(identity), sort_keys=True), encoding="utf-8"
    )
    original = reference_path.read_bytes()
    monkeypatch.setattr(AUDIT, "load_config", lambda _: {"data": {}})
    monkeypatch.setattr(AUDIT, "_build_audit_inputs", lambda _paths, _configs: identity)

    result = AUDIT.check_audit_reference(["configs/model.yaml"], reference_path)

    assert result["status"] == "matched"
    assert reference_path.read_bytes() == original


def test_direct_audit_cli_refuses_to_overwrite_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fairness.json"
    output.write_text("protected\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_parameter_fairness.py", "--output", str(output)],
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        AUDIT.main()

    assert output.read_text(encoding="utf-8") == "protected\n"


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
