from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eptnet.data.dataset import ManifestDataset
from eptnet.data.prepare_bci_subjects import (
    discover_sessions,
    validate_reused_staged_raw,
)
from eptnet.data.session_package import (
    SessionPackageError,
    atomic_write_jsonl,
    session_to_sample,
    sha256_file,
    verify_session_package,
    write_session_package,
)


def _prepared_session(tmp_path: Path) -> SimpleNamespace:
    source_root = tmp_path / "raw" / "session_001"
    source_root.mkdir(parents=True)
    annotation = source_root / "session_001_annotations.docx"
    annotation.write_bytes(b"annotation")
    labels = np.asarray([1, 0, 0, 1], dtype=np.int64)
    target_mask = np.ones(4, dtype=bool)
    positive_mask = labels == 0
    boundaries = np.zeros((4, 2), dtype=np.float32)
    boundaries[1, 0] = 1
    boundaries[2, 1] = 1
    offsets = np.zeros((4, 2), dtype=np.float32)
    offsets[1] = [0, 1]
    offsets[2] = [1, 0]
    physiology_mask = np.zeros((4, 3), dtype=bool)
    modality_mask = np.zeros((4, 3), dtype=bool)
    modality_mask[:, 2] = True
    return SimpleNamespace(
        dataset_name="unit_dataset",
        session_id="session_001",
        subject_id="private_name",
        step_seconds=1.0,
        labels=labels,
        target_mask=target_mask,
        positive_mask=positive_mask,
        boundaries=boundaries,
        offsets=offsets,
        timestamps=np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.float32),
        physiology_mask=physiology_mask,
        modality_mask=modality_mask,
        features={
            "eeg_time": np.zeros((4, 8), dtype=np.float32),
            "eeg_spectral": np.zeros((4, 40), dtype=np.float32),
            "physiology": np.zeros((4, 40), dtype=np.float32),
            "video": np.zeros((4, 3), dtype=np.float32),
            "audio": np.zeros((4, 50), dtype=np.float32),
            "text": np.ones((4, 768), dtype=np.float32),
        },
        step_text=["a", "b", "c", "d"],
        behavior_provenance={"text_backend": "unit"},
        source=SimpleNamespace(
            annotation=annotation,
            face_csv=None,
            openface_csv=None,
            eeg_file=None,
            audio_file=None,
            video_file=None,
            ppg_files=None,
        ),
    )


def test_session_package_is_complete_verified_and_directly_loadable(tmp_path: Path) -> None:
    session = _prepared_session(tmp_path)
    output_root = tmp_path / "processed"
    record = write_session_package(session, "train", output_root)
    manifest_path = output_root / "sessions" / "session_001" / "session_manifest.json"

    verification = verify_session_package(manifest_path)
    assert verification["num_steps"] == 4
    assert all(verification["checks"].values())

    manifest_path_for_loader = output_root / "manifests" / "sessions_train.jsonl"
    atomic_write_jsonl(manifest_path_for_loader, [record])
    dataset = ManifestDataset(str(manifest_path_for_loader))
    sample = dataset[0]
    assert sample["sample_id"] == "session_001"
    assert torch.equal(sample["row_indices"], torch.arange(4))
    assert sample["metadata"]["participant_id"] == "session_001"


def test_session_package_rejects_target_leakage(tmp_path: Path) -> None:
    sample = session_to_sample(_prepared_session(tmp_path))
    sample["positive_mask"][0] = True
    tensor_path = tmp_path / "timeline.pt"
    torch.save(sample, tensor_path)
    manifest = {
        "schema_version": 1,
        "session_id": "session_001",
        "tensor_file": "timeline.pt",
        "tensor_sha256": sha256_file(tensor_path),
    }
    (tmp_path / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SessionPackageError, match="positive_mask"):
        verify_session_package(tmp_path / "session_manifest.json")


def test_manifest_loader_rejects_path_escape(tmp_path: Path) -> None:
    dataset_root = tmp_path / "processed"
    manifest_root = dataset_root / "manifests"
    manifest_root.mkdir(parents=True)
    outside = tmp_path / "outside.pt"
    torch.save({"labels": torch.ones(1, dtype=torch.long)}, outside)
    record = {"sample_id": "escape", "tensor_file": "../../outside.pt"}
    atomic_write_jsonl(manifest_root / "sessions.jsonl", [record])

    dataset = ManifestDataset(str(manifest_root / "sessions.jsonl"))
    with pytest.raises(ValueError, match="escapes dataset root"):
        dataset[0]


def test_staged_discovery_preserves_session_id_and_ppg_roles(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "_audit").mkdir(parents=True)
    session_root = raw_root / "session_019"
    session_root.mkdir()
    (session_root / "session_019_annotations.docx").write_bytes(b"docx")
    (session_root / "session_019_ppg_channel_1.txt").write_text("1", encoding="utf-8")
    (session_root / "session_019_ppg_channel_2.txt").write_text("2", encoding="utf-8")
    metadata = {
        "session_id": "session_019",
        "subject_id": "private_name",
        "annotation_mode": "yellow_highlight",
        "available_sources": {"annotated_transcript": True},
        "files": [],
    }
    (session_root / "session_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    sessions = discover_sessions(raw_root)
    assert len(sessions) == 1
    assert sessions[0].session_id == "session_019"
    assert sessions[0].subject_id == "session_019"
    assert sessions[0].ppg_files is not None


def test_staged_discovery_and_validation_support_centralized_facial_csv(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    session_root = raw_root / "session_019"
    facial_root = raw_root / "facial_csv" / "session_019"
    session_root.mkdir(parents=True)
    facial_root.mkdir(parents=True)
    annotation = session_root / "session_019_annotations.docx"
    face_csv = facial_root / "session_019_facial_actions.csv"
    annotation.write_bytes(b"docx")
    face_csv.write_text(
        "timestamp,blink,pressed_lips,furrow_brow\n0,0,0,0\n",
        encoding="utf-8",
    )
    metadata = {
        "dataset": "test_dataset",
        "session_id": "session_019",
        "subject_id": "private_name",
        "annotation_mode": "yellow_highlight",
        "available_sources": {
            "annotated_transcript": True,
            "facial_actions": True,
        },
        "files": [
            {
                "staged_relative_path": "session_019/session_019_annotations.docx",
                "sha256": sha256_file(annotation),
            },
            {
                "staged_relative_path": ("facial_csv/session_019/session_019_facial_actions.csv"),
                "sha256": sha256_file(face_csv),
            },
        ],
    }
    (session_root / "session_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    sessions = discover_sessions(raw_root)

    assert len(sessions) == 1
    assert sessions[0].face_csv == face_csv.resolve()
    manifest = validate_reused_staged_raw(raw_root, sessions, dataset_name="test_dataset")
    staged_paths = {record["staged_relative_path"] for record in manifest["files"]}
    assert "facial_csv/session_019/session_019_facial_actions.csv" in staged_paths

    no_facial_manifest = validate_reused_staged_raw(
        raw_root,
        sessions,
        dataset_name="test_dataset_no_facial",
        include_facial_csv=False,
    )
    no_facial_paths = {record["staged_relative_path"] for record in no_facial_manifest["files"]}
    assert not any("facial" in path for path in no_facial_paths)
    assert no_facial_manifest["source_policy"] == {
        "include_facial_csv": False,
        "excluded_facial_csv_records": 1,
    }
