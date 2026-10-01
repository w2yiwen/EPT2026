from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from eptnet.config import load_config


CONFIGS = {
    "configs/eptnet_marlin11_eeg_ppg_video.yaml": (
        "eptnet",
        (True, True, True, True),
    ),
    "configs/gru_marlin11_eeg_ppg_video.yaml": (
        "early_fusion_gru",
        (True, True, True, True),
    ),
    "configs/transformer_marlin11_eeg_ppg_video.yaml": (
        "fusion_transformer",
        (True, True, True, True),
    ),
    "configs/eptnet_marlin11_video_only.yaml": (
        "eptnet",
        (False, False, False, True),
    ),
    "configs/eptnet_marlin11_fixed_reader.yaml": (
        "eptnet",
        (True, True, True, True),
    ),
    "configs/eptnet_marlin11_no_persistent.yaml": (
        "eptnet",
        (True, True, True, True),
    ),
}


def _load_preflight_module():
    script = Path("scripts/experiments/preflight_marlin11_shortpaper.py")
    spec = importlib.util.spec_from_file_location("preflight_marlin11_shortpaper", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_verifier_module():
    script = Path("scripts/experiments/verify_marlin11_shortpaper_results.py")
    spec = importlib.util.spec_from_file_location(
        "verify_marlin11_shortpaper_results", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("path", "expected"), CONFIGS.items())
def test_paper_configs_preserve_frozen_aligned11_contract(path, expected):
    expected_model, expected_modalities = expected
    frozen = load_config("configs/eptnet_marlin11_aligned_windowed_seed42.yaml")
    config = load_config(path)
    model = config["model"]

    assert config["data"] == frozen["data"]
    assert config["training"] == frozen["training"]
    assert config["loss"] == frozen["loss"]
    assert config["evaluation"]["positive_class"] == 0
    assert model["name"] == expected_model
    assert model["use_audio"] is False
    assert model["use_text"] is False
    assert (
        model["use_eeg_time"],
        model["use_eeg_spec"],
        model["use_hr"],
        model["use_behavior_context"] and model["use_video"],
    ) == expected_modalities
    assert config["experiment"]["output_dir"].endswith("_no_text")
    assert "marlin11" in config["experiment"]["output_dir"]


def test_paper_gru_uses_frozen_aligned11_gru_architecture():
    frozen = load_config("configs/gru_marlin11_aligned_windowed_seed42.yaml")
    paper = load_config("configs/gru_marlin11_eeg_ppg_video.yaml")

    assert paper["model"]["name"] == frozen["model"]["name"] == "early_fusion_gru"
    assert paper["model"]["hidden_dim"] == frozen["model"]["hidden_dim"]


def test_paper_diagnostics_change_only_the_named_mechanism():
    full = load_config("configs/eptnet_marlin11_eeg_ppg_video.yaml")
    cases = (
        ("configs/eptnet_marlin11_fixed_reader.yaml", "adaptive_reader"),
        ("configs/eptnet_marlin11_no_persistent.yaml", "persistent_state"),
    )
    for path, key in cases:
        diagnostic = load_config(path)
        assert full["model"][key] is True
        assert diagnostic["model"][key] is False
        normalized = deepcopy(diagnostic)
        normalized["model"][key] = True
        normalized["experiment"] = full["experiment"]
        assert normalized == full


def test_read_only_preflight_hashes_metadata_without_writing_dataset(tmp_path):
    module = _load_preflight_module()

    dataset_root = tmp_path / "frozen"
    (dataset_root / "manifests").mkdir(parents=True)
    (dataset_root / "dataset_summary.json").write_text("{}\n", encoding="utf-8")
    (dataset_root / "manifests" / "sessions_train.jsonl").write_text(
        '{"sample_id":"session_002"}\n', encoding="utf-8"
    )
    before = sorted(path.relative_to(dataset_root) for path in dataset_root.rglob("*"))

    records, digest = module._metadata_fingerprints(dataset_root)

    after = sorted(path.relative_to(dataset_root) for path in dataset_root.rglob("*"))
    assert before == after
    assert digest and len(digest) == 64
    assert {record["path"] for record in records} == {
        "dataset_summary.json",
        "manifests/sessions_train.jsonl",
    }


def test_preflight_parses_session_level_manifests_and_rejects_duplicates(tmp_path):
    module = _load_preflight_module()
    manifest = tmp_path / "sessions_train.jsonl"
    manifest.write_text(
        "\n".join(
            (
                '{"sample_id":"fallback","metadata":{"session_id":"session_002"}}',
                '{"sample_id":"session_003"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert module._manifest_session_ids(manifest) == ["session_002", "session_003"]

    manifest.write_text(
        '{"sample_id":"session_002"}\n{"sample_id":"session_002"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate session IDs"):
        module._manifest_session_ids(manifest)


def test_preflight_verifies_manifest_session_tensor_checksum(tmp_path):
    module = _load_preflight_module()
    dataset_root = tmp_path / "frozen"
    manifest_root = dataset_root / "manifests"
    tensor_root = dataset_root / "sessions" / "session_002"
    manifest_root.mkdir(parents=True)
    tensor_root.mkdir(parents=True)
    tensor_path = tensor_root / "timeline.pt"
    tensor_path.write_bytes(b"frozen-session-tensor")
    digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest = manifest_root / "sessions_train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "session_002",
                "tensor_file": "../sessions/session_002/timeline.pt",
                "sha256": digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert module._manifest_tensor_fingerprints(
        manifest, dataset_root=dataset_root
    ) == [
        {
            "path": "sessions/session_002/timeline.pt",
            "bytes": len(b"frozen-session-tensor"),
            "sha256": digest,
        }
    ]

    tensor_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module._manifest_tensor_fingerprints(manifest, dataset_root=dataset_root)


def test_preflight_rejects_frozen_evidence_drift():
    module = _load_preflight_module()
    reference = {
        "dataset": module.FROZEN_DATASET,
        "label_mapping": {"0": "deception", "1": "truth"},
        "positive_class": 0,
        "expected_session_ids": list(module.FROZEN_SESSION_IDS),
        "excluded_session_ids": list(module.FROZEN_EXCLUDED_IDS),
        "manifests": {"train": {"sha256": "a"}},
        "metadata_files": [{"path": "dataset_summary.json", "sha256": "b"}],
        "metadata_bundle_sha256": "c",
    }
    current = deepcopy(reference)
    module.assert_frozen_evidence_matches(reference, current)

    current["metadata_bundle_sha256"] = "changed"
    with pytest.raises(ValueError, match="metadata_bundle_sha256"):
        module.assert_frozen_evidence_matches(reference, current)


def test_preflight_output_creation_is_exclusive(tmp_path):
    output = tmp_path / "preflight.json"
    command = [
        sys.executable,
        "scripts/experiments/preflight_marlin11_shortpaper.py",
        "configs/eptnet_marlin11_eeg_ppg_video.yaml",
        "--allow-missing-data",
        "--skip-device-check",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    repeated = subprocess.run(command, check=False, capture_output=True, text=True)

    assert repeated.returncode != 0
    assert "Refusing to overwrite preflight record" in repeated.stderr


def test_completed_run_verifier_rejects_stale_identity(tmp_path):
    module = _load_verifier_module()
    config_path = Path("configs/eptnet_marlin11_eeg_ppg_video.yaml")
    config = module.config_module.load_config(str(config_path))
    training_identity = module.provenance_module.training_config_fingerprint(config)
    provenance_identity = "a" * 64
    source_identity = "b" * 64

    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "passed",
                "warnings": [],
                "positive_class": 0,
                "label_mapping": {"0": "deception", "1": "truth"},
                "runtime_provenance": {
                    "provenance_sha256": provenance_identity,
                    "source_tree_sha256": source_identity,
                },
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "seed_42"
    run_dir.mkdir()
    metrics_path = run_dir / "test_metrics.json"
    predictions_path = run_dir / "test_predictions.jsonl"
    events_path = run_dir / "test_events.json"
    metrics = {
        "protocol": {
            "positive_class": 0,
            "class_encoding": {"0": "deception", "1": "truth"},
        },
        "checkpoint": {
            "experiment": config["experiment"]["name"],
            "seed": 42,
            "training_config_sha256": training_identity,
            "provenance_sha256": provenance_identity,
        },
        "provenance": {
            "provenance_sha256": provenance_identity,
            "source_tree": {"sha256": source_identity},
        },
        "data": {
            "num_sequences": 1,
            "num_unique_steps": 2,
            "num_evaluated_target_steps": 2,
            "num_predicted_events": 1,
            "num_target_events": 1,
        },
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    predictions_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "sample_id": "session_018",
                        "row_index": 10,
                        "label": 0,
                        "positive_class": 0,
                        "target_valid": True,
                        "positive_probability": 0.8,
                        "class_probabilities": [0.8, 0.2],
                        "boundary_probabilities": [0.1, 0.9],
                        "offsets": [0.0, 1.0],
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "session_018",
                        "row_index": 11,
                        "label": 1,
                        "positive_class": 0,
                        "target_valid": True,
                        "positive_probability": 0.3,
                        "class_probabilities": [0.3, 0.7],
                        "boundary_probabilities": [0.7, 0.2],
                        "offsets": [1.0, 0.0],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    events_path.write_text(
        json.dumps(
            {
                "predictions": [[{"start": 10.0, "end": 10.0}]],
                "targets": [[{"start": 10.0, "end": 10.0}]],
            }
        ),
        encoding="utf-8",
    )

    verified = module.verify_completed_run(
        config_path=config_path,
        seed=42,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        events_path=events_path,
        preflight_path=preflight,
    )
    assert verified["status"] == "passed"
    assert verified["predictions"]["rows"] == 2

    prediction_rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    prediction_rows[0]["boundary_probabilities"][0] = -0.1
    predictions_path.write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="boundary probability"):
        module.verify_completed_run(
            config_path=config_path,
            seed=42,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            events_path=events_path,
            preflight_path=preflight,
        )

    prediction_rows[0]["boundary_probabilities"][0] = 0.1
    prediction_rows[0]["offsets"][0] = -1.0
    predictions_path.write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="offset"):
        module.verify_completed_run(
            config_path=config_path,
            seed=42,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            events_path=events_path,
            preflight_path=preflight,
        )

    prediction_rows[0]["offsets"][0] = 0.0
    predictions_path.write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )
    metrics["checkpoint"]["training_config_sha256"] = "c" * 64
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration fingerprint is stale"):
        module.verify_completed_run(
            config_path=config_path,
            seed=42,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            events_path=events_path,
            preflight_path=preflight,
        )


def test_aggregate_verifier_recomputes_without_writing(tmp_path):
    module = _load_verifier_module()
    provenance_identity = "d" * 64
    training_identity = "e" * 64
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "passed",
                "warnings": [],
                "positive_class": 0,
                "label_mapping": {"0": "deception", "1": "truth"},
                "runtime_provenance": {
                    "provenance_sha256": provenance_identity,
                    "source_tree_sha256": "f" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    metric_paths = []
    for seed, value in ((13, 0.6), (42, 0.8)):
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "checkpoint": {
                        "experiment": "paper-test",
                        "seed": seed,
                        "training_config_sha256": training_identity,
                        "provenance_sha256": provenance_identity,
                    },
                    "provenance": {"provenance_sha256": provenance_identity},
                    "protocol": {"positive_class": 0, "frame_threshold": 0.5},
                    "data": {
                        "num_sequences": 1,
                        "num_unique_steps": 2,
                        "num_evaluated_target_steps": 2,
                        "num_target_events": 1,
                        "num_dense_event_proposals": 2,
                    },
                    "frame": {"average_precision": value, "brier_score": 1.0 - value},
                }
            ),
            encoding="utf-8",
        )
        metric_paths.append(path)

    expected = module.aggregate_module.aggregate_result_files(metric_paths)
    aggregate_json = tmp_path / "aggregate.json"
    aggregate_csv = tmp_path / "aggregate.csv"
    aggregate_json.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for metric, values in expected["aggregate"].items():
            writer.writerow({"metric": metric, **values})

    verified = module.verify_aggregate(
        metrics_paths=metric_paths,
        aggregate_json_path=aggregate_json,
        aggregate_csv_path=aggregate_csv,
        preflight_path=preflight,
    )
    assert verified["status"] == "passed"
    assert verified["metrics_files"] == 2

    expected["num_runs"] = 99
    aggregate_json.write_text(json.dumps(expected), encoding="utf-8")
    with pytest.raises(ValueError, match="read-only recomputation"):
        module.verify_aggregate(
            metrics_paths=metric_paths,
            aggregate_json_path=aggregate_json,
            aggregate_csv_path=aggregate_csv,
            preflight_path=preflight,
        )


def test_runner_dry_run_requires_no_local_dataset_or_accelerator():
    runner = Path("scripts/experiments/run_marlin11_shortpaper.sh")
    completed = subprocess.run(
        [
            "bash",
            str(runner),
            "--dry-run",
            "--suite",
            "main",
            "--seeds",
            "13 42 73",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "passed_with_warnings" in completed.stdout
    assert "seed_13" in completed.stdout
    assert "seed_42" in completed.stdout
    assert "seed_73" in completed.stdout
    assert "eptnet.train" in completed.stdout
    assert "eptnet.evaluate" in completed.stdout


def test_runner_contains_no_data_preparation_or_migration_command():
    text = Path("scripts/experiments/run_marlin11_shortpaper.sh").read_text(encoding="utf-8")

    forbidden = ("prepare_bci", "scripts/data/", "--overwrite", "migrate", "relabel")
    assert all(token not in text for token in forbidden)
    assert "--resume-partial" in text
    assert 'resume_from="$resume_checkpoint"' in text
    assert 'local resume_checkpoint="$run_dir/last.pt"' in text
    assert 'local best_checkpoint="$run_dir/best.pt"' in text
    assert 'local predictions="$run_dir/test_predictions.jsonl"' in text
    assert 'local events="$run_dir/test_events.json"' in text
    assert "verify_marlin11_shortpaper_results.py run" in text
    assert "verify_marlin11_shortpaper_results.py aggregate" in text
    assert "aggregate_recovered_seeds_" in text
    assert "--no-overwrite" in text
    assert '--check-reference "$fairness_output"' in text
