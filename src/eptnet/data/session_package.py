from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

SESSION_PACKAGE_SCHEMA_VERSION = 1
FEATURE_KEYS = ("eeg_time", "eeg_spectral", "physiology", "video", "audio", "text")
MASK_KEYS = ("physiology_mask", "modality_mask", "target_mask", "positive_mask")
TARGET_KEYS = ("labels", "boundaries", "offsets")


class SessionPackageError(ValueError):
    """Raised when a session package violates the on-disk research contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise OSError(f"Atomic writer produced no data for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    def write(target: Path) -> None:
        target.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )

    _atomic_write(path, write)


def atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    def write(target: Path) -> None:
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    _atomic_write(path, write)


def atomic_torch_save(path: Path, value: object) -> None:
    _atomic_write(path, lambda target: torch.save(value, target))


def _tensor(values: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(values).copy())


def session_to_sample(session: Any) -> dict[str, Any]:
    """Convert one normalized session to a complete, non-overlapping timeline."""

    num_steps = len(session.labels)
    sample: dict[str, Any] = {
        **{name: _tensor(session.features[name]) for name in FEATURE_KEYS},
        "physiology_mask": _tensor(session.physiology_mask),
        "modality_mask": _tensor(session.modality_mask),
        "target_mask": _tensor(session.target_mask),
        "labels": _tensor(session.labels),
        "boundaries": _tensor(session.boundaries),
        "offsets": _tensor(session.offsets),
        "positive_mask": _tensor(session.positive_mask),
        "timestamps": _tensor(session.timestamps),
        "row_indices": torch.arange(num_steps, dtype=torch.long),
        "words": list(session.step_text),
    }
    return sample


def _expected_event_targets(positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.zeros((len(positive), 2), dtype=np.float32)
    offsets = np.zeros((len(positive), 2), dtype=np.float32)
    index = 0
    while index < len(positive):
        if not positive[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(positive) and positive[index + 1]:
            index += 1
        end = index
        boundaries[start, 0] = 1.0
        boundaries[end, 1] = 1.0
        positions = np.arange(start, end + 1)
        offsets[start : end + 1, 0] = positions - start
        offsets[start : end + 1, 1] = end - positions
        index += 1
    return boundaries, offsets


def validate_session_sample(sample: Mapping[str, Any], *, session_id: str) -> dict[str, bool]:
    """Fail closed on shape, dtype, missingness, time, and target inconsistencies."""

    required = {
        *FEATURE_KEYS,
        *MASK_KEYS,
        *TARGET_KEYS,
        "timestamps",
        "row_indices",
        "words",
    }
    missing = sorted(required - set(sample))
    if missing:
        raise SessionPackageError(f"{session_id}: missing fields {missing}")
    labels = sample["labels"]
    if not isinstance(labels, Tensor) or labels.ndim != 1 or labels.dtype != torch.int64:
        raise SessionPackageError(f"{session_id}: labels must be rank-1 int64")
    num_steps = int(labels.shape[0])
    if num_steps <= 0:
        raise SessionPackageError(f"{session_id}: empty session")

    for name in FEATURE_KEYS:
        values = sample[name]
        if not isinstance(values, Tensor) or values.ndim != 2 or values.shape[0] != num_steps:
            raise SessionPackageError(f"{session_id}: invalid feature shape for {name}")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise SessionPackageError(f"{session_id}: {name} must contain finite floats")
    for name in MASK_KEYS:
        values = sample[name]
        if not isinstance(values, Tensor) or values.shape[0] != num_steps:
            raise SessionPackageError(f"{session_id}: invalid mask shape for {name}")
        if values.dtype != torch.bool:
            raise SessionPackageError(f"{session_id}: {name} must use bool dtype")
    if sample["physiology_mask"].shape != (num_steps, 3):
        raise SessionPackageError(f"{session_id}: physiology_mask must have shape [T,3]")
    if sample["modality_mask"].shape != (num_steps, 3):
        raise SessionPackageError(f"{session_id}: modality_mask must have shape [T,3]")
    feature_masks = {
        "eeg_time": sample["physiology_mask"][:, 0],
        "eeg_spectral": sample["physiology_mask"][:, 1],
        "physiology": sample["physiology_mask"][:, 2],
        "video": sample["modality_mask"][:, 0],
        "audio": sample["modality_mask"][:, 1],
        "text": sample["modality_mask"][:, 2],
    }
    for name, mask in feature_masks.items():
        if torch.any(sample[name][~mask] != 0):
            raise SessionPackageError(
                f"{session_id}: unavailable {name} rows must be exact zero placeholders"
            )

    timestamps = sample["timestamps"]
    if (
        not isinstance(timestamps, Tensor)
        or timestamps.shape != (num_steps, 2)
        or not timestamps.is_floating_point()
        or not torch.isfinite(timestamps).all()
    ):
        raise SessionPackageError(f"{session_id}: timestamps must be finite [T,2] floats")
    if torch.any(timestamps[:, 1] <= timestamps[:, 0]):
        raise SessionPackageError(f"{session_id}: every timestamp interval must be positive")
    if num_steps > 1:
        if torch.any(timestamps[1:, 0] <= timestamps[:-1, 0]):
            raise SessionPackageError(f"{session_id}: timestamp starts must increase strictly")
        if not torch.allclose(timestamps[:-1, 1], timestamps[1:, 0], atol=1e-6, rtol=0):
            raise SessionPackageError(f"{session_id}: timeline contains gaps or overlap")

    row_indices = sample["row_indices"]
    if (
        not isinstance(row_indices, Tensor)
        or row_indices.dtype != torch.int64
        or row_indices.ndim != 1
        or not torch.equal(row_indices, torch.arange(num_steps, dtype=torch.long))
    ):
        raise SessionPackageError(f"{session_id}: row_indices must be contiguous 0..T-1")
    words = sample["words"]
    if not isinstance(words, list) or len(words) != num_steps or not all(
        isinstance(value, str) for value in words
    ):
        raise SessionPackageError(f"{session_id}: words must be a length-T string list")

    if not torch.all((labels == 0) | (labels == 1)):
        raise SessionPackageError(f"{session_id}: labels must use only 0/1")
    target_mask = sample["target_mask"]
    expected_positive = (labels == 0) & target_mask
    if not torch.equal(sample["positive_mask"], expected_positive):
        raise SessionPackageError(f"{session_id}: positive_mask leaks outside target_mask")
    boundaries = sample["boundaries"]
    offsets = sample["offsets"]
    if (
        not isinstance(boundaries, Tensor)
        or boundaries.shape != (num_steps, 2)
        or not isinstance(offsets, Tensor)
        or offsets.shape != (num_steps, 2)
        or not torch.isfinite(boundaries).all()
        or not torch.isfinite(offsets).all()
    ):
        raise SessionPackageError(f"{session_id}: boundaries/offsets must be finite [T,2]")
    expected_boundaries, expected_offsets = _expected_event_targets(
        expected_positive.cpu().numpy()
    )
    if not torch.equal(boundaries.cpu(), torch.from_numpy(expected_boundaries)):
        raise SessionPackageError(f"{session_id}: boundary targets are inconsistent")
    if not torch.equal(offsets.cpu(), torch.from_numpy(expected_offsets)):
        raise SessionPackageError(f"{session_id}: offset targets are inconsistent")

    return {
        "required_fields": True,
        "aligned_shapes": True,
        "finite_features": True,
        "explicit_missingness": True,
        "monotonic_contiguous_timeline": True,
        "contiguous_row_indices": True,
        "target_mask_no_future_leakage": True,
        "event_targets_recomputed": True,
    }


def _source_paths(session: Any) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = [("annotation", Path(session.source.annotation))]
    for role, value in (
        ("facial_actions", session.source.face_csv),
        ("openface_features", session.source.openface_csv),
        ("eeg", session.source.eeg_file),
        ("audio", session.source.audio_file),
        ("video", session.source.video_file),
    ):
        if value is not None:
            sources.append((role, Path(value)))
    if session.source.ppg_files is not None:
        sources.extend(
            (f"ppg_channel_{index}", Path(path))
            for index, path in enumerate(session.source.ppg_files, 1)
        )
    return sources


def write_session_package(session: Any, split: str, output_root: Path) -> dict[str, object]:
    """Write one complete model-ready timeline and its integrity manifest."""

    sample = session_to_sample(session)
    checks = validate_session_sample(sample, session_id=str(session.session_id))
    session_root = output_root / "sessions" / session.session_id
    tensor_path = session_root / "timeline.pt"
    atomic_torch_save(tensor_path, sample)
    tensor_hash = sha256_file(tensor_path)
    source_records = [
        {
            "role": role,
            "bytes": path.stat().st_size,
        }
        for role, path in _source_paths(session)
    ]
    manifest: dict[str, object] = {
        "schema_version": SESSION_PACKAGE_SCHEMA_VERSION,
        "dataset": str(session.dataset_name),
        "session_id": str(session.session_id),
        "participant_id": str(session.session_id),
        "split": split,
        "num_steps": int(sample["labels"].shape[0]),
        "step_seconds": float(session.step_seconds),
        "tensor_file": "timeline.pt",
        "tensor_bytes": tensor_path.stat().st_size,
        "tensor_sha256": tensor_hash,
        "feature_shapes": {
            name: list(sample[name].shape) for name in (*FEATURE_KEYS, "timestamps")
        },
        "feature_dtypes": {
            name: str(sample[name].dtype)
            for name in (*FEATURE_KEYS, *MASK_KEYS, *TARGET_KEYS, "timestamps", "row_indices")
        },
        "availability_steps": {
            "eeg_time": int(sample["physiology_mask"][:, 0].sum()),
            "eeg_spectral": int(sample["physiology_mask"][:, 1].sum()),
            "physiology": int(sample["physiology_mask"][:, 2].sum()),
            "video": int(sample["modality_mask"][:, 0].sum()),
            "audio": int(sample["modality_mask"][:, 1].sum()),
            "text": int(sample["modality_mask"][:, 2].sum()),
        },
        "target_valid_steps": int(sample["target_mask"].sum()),
        "deception_steps": int(sample["positive_mask"].sum()),
        "deception_events": int(sample["boundaries"][:, 0].sum()),
        "integrity_checks": checks,
        "source_artifacts": source_records,
        "source_hash_ledger": "../../source_manifest.json",
        "behavior_provenance": dict(session.behavior_provenance),
        "alignment": dict(getattr(session, "alignment", {})),
    }
    manifest_path = session_root / "session_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "sample_id": str(session.session_id),
        "tensor_file": f"../sessions/{session.session_id}/timeline.pt",
        "sha256": tensor_hash,
        "metadata": {
            "dataset": str(session.dataset_name),
            "session_id": str(session.session_id),
            "participant_id": str(session.session_id),
            "split": split,
            "num_steps": int(sample["labels"].shape[0]),
            "target_valid_steps": int(sample["target_mask"].sum()),
        },
    }


def verify_session_package(manifest_path: Path) -> dict[str, object]:
    """Verify checksum and semantic contents without unsafe pickle loading."""

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SessionPackageError(f"Manifest must be an object: {manifest_path}")
    if manifest.get("schema_version") != SESSION_PACKAGE_SCHEMA_VERSION:
        raise SessionPackageError(f"Unsupported schema version in {manifest_path}")
    tensor_file = manifest.get("tensor_file")
    if not isinstance(tensor_file, str) or Path(tensor_file).name != tensor_file:
        raise SessionPackageError(f"Unsafe tensor_file in {manifest_path}")
    tensor_path = (manifest_path.parent / tensor_file).resolve()
    if tensor_path.parent != manifest_path.parent:
        raise SessionPackageError(f"Tensor escapes session directory: {tensor_path}")
    expected_hash = manifest.get("tensor_sha256")
    if not isinstance(expected_hash, str) or sha256_file(tensor_path) != expected_hash:
        raise SessionPackageError(f"Tensor checksum mismatch: {tensor_path}")
    sample = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(sample, dict):
        raise SessionPackageError(f"Tensor payload must be a dictionary: {tensor_path}")
    checks = validate_session_sample(sample, session_id=str(manifest.get("session_id")))
    return {
        "session_id": manifest.get("session_id"),
        "tensor_sha256": expected_hash,
        "num_steps": int(sample["labels"].shape[0]),
        "checks": checks,
    }
