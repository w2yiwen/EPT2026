from __future__ import annotations

import json
from pathlib import Path

import pytest

from eptnet.train import validate_training_cohort


def _write_contract_fixture(root: Path) -> dict:
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    split_ids = {"train": ["session_002"], "val": ["session_003"], "test": ["session_004"]}
    for split, session_ids in split_ids.items():
        records = [
            {
                "sample_id": session_id,
                "tensor_file": f"../sessions/{session_id}/timeline.pt",
                "metadata": {"session_id": session_id},
            }
            for session_id in session_ids
        ]
        (manifests / f"sessions_{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    for session_id in ("session_002", "session_003", "session_004"):
        session_root = root / "sessions" / session_id
        session_root.mkdir(parents=True)
        (session_root / "session_manifest.json").write_text(
            json.dumps(
                {
                    "availability_steps": {"video": 5, "audio": 5, "text": 4},
                    "alignment": {
                        "whisper_alignment": {"status": "complete"},
                        "video_feature_alignment_method": "legacy_embedding_time_remap_v1",
                    },
                }
            ),
            encoding="utf-8",
        )
    (root / "dataset_summary.json").write_text(
        json.dumps({"dataset": "fixture_aligned3", "num_subjects": 3}), encoding="utf-8"
    )
    return {
        "data": {
            "dataset_name": "fixture_aligned3",
            **{
                f"{split}_manifest": str(manifests / f"sessions_{split}.jsonl")
                for split in split_ids
            },
            "expected_session_ids": ["session_002", "session_003", "session_004"],
            "excluded_session_ids": ["session_011"],
            "require_aligned_behavior_modalities": True,
        }
    }


def test_training_cohort_contract_accepts_exact_aligned_sessions(tmp_path: Path) -> None:
    report = validate_training_cohort(_write_contract_fixture(tmp_path))

    assert report["status"] == "passed"
    assert report["session_count"] == 3
    assert report["excluded_session_ids"] == ["session_011"]
    assert set(report["video_alignment_methods"].values()) == {
        "legacy_embedding_time_remap_v1"
    }


def test_training_cohort_contract_accepts_all_sessions_training_policy(
    tmp_path: Path,
) -> None:
    config = _write_contract_fixture(tmp_path)
    manifests = tmp_path / "manifests"
    all_records = []
    for session_id in ("session_002", "session_003", "session_004"):
        all_records.append(
            {
                "sample_id": session_id,
                "tensor_file": f"../sessions/{session_id}/timeline.pt",
                "metadata": {"session_id": session_id},
            }
        )
    all_manifest = manifests / "sessions_all.jsonl"
    all_manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in all_records),
        encoding="utf-8",
    )
    config["data"]["cohort_policy"] = "all_sessions_training"
    config["data"]["train_manifest"] = str(all_manifest)

    report = validate_training_cohort(config)

    assert report["status"] == "passed"
    assert report["cohort_policy"] == "all_sessions_training"
    assert report["session_count"] == 3
    assert report["split_session_ids"]["train"] == [
        "session_002",
        "session_003",
        "session_004",
    ]


def test_training_cohort_contract_rejects_excluded_session_leak(tmp_path: Path) -> None:
    config = _write_contract_fixture(tmp_path)
    config["data"]["expected_session_ids"] = ["session_002", "session_003", "session_011"]
    test_manifest = Path(config["data"]["test_manifest"])
    test_manifest.write_text(
        json.dumps(
            {
                "sample_id": "session_011",
                "tensor_file": "../sessions/session_011/timeline.pt",
                "metadata": {"session_id": "session_011"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="excluded sessions"):
        validate_training_cohort(config)
