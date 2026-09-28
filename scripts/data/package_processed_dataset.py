#!/usr/bin/env python3
"""Validate and create a self-contained archive for a processed EPT-Net dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch

REQUIRED_ROOT_FILES = {
    "_SUCCESS.json",
    "alignment_report.json",
    "dataset_integrity.json",
    "dataset_summary.json",
    "feature_schema.json",
    "normalization_stats.npz",
    "README.md",
    "session_index.json",
    "source_manifest.json",
    "split_strategy.json",
}
REQUIRED_TENSOR_FIELDS = {
    "eeg_time",
    "eeg_spectral",
    "physiology",
    "video",
    "audio",
    "text",
    "physiology_mask",
    "modality_mask",
    "target_mask",
    "labels",
    "boundaries",
    "offsets",
    "positive_mask",
    "timestamps",
    "row_indices",
    "words",
}
FORBIDDEN_SOURCE_SUFFIXES = {".csv", ".docx", ".mp4", ".wav", ".txt"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _manifest_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Manifest line {line_number} is not an object: {path}")
            records.append(record)
    return records


def _contained_file(root: Path, manifest: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Invalid tensor_file in {manifest}: {reference!r}")
    referenced = Path(reference)
    if referenced.is_absolute():
        raise ValueError(f"Absolute tensor path is not portable: {reference}")
    resolved = (manifest.parent / referenced).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Tensor path escapes dataset root: {reference}") from error
    if not resolved.is_file() or resolved.is_symlink() or resolved.suffix.lower() != ".pt":
        raise FileNotFoundError(f"Expected a regular .pt tensor file: {resolved}")
    return resolved


def validate_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    missing = sorted(name for name in REQUIRED_ROOT_FILES if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing required dataset files: {missing}")

    package_files = sorted(path for path in root.rglob("*") if path.is_file())
    symbolic = [path.relative_to(root).as_posix() for path in package_files if path.is_symlink()]
    if symbolic:
        raise ValueError(f"Symbolic links are forbidden in a portable dataset: {symbolic}")
    source_like = [
        path.relative_to(root).as_posix()
        for path in package_files
        if path.suffix.lower() in FORBIDDEN_SOURCE_SUFFIXES
    ]
    if source_like:
        raise ValueError(f"Raw/source-like files leaked into processed package: {source_like}")

    summary = _json(root / "dataset_summary.json")
    integrity = _json(root / "dataset_integrity.json")
    alignment = _json(root / "alignment_report.json")
    manifest = root / "manifests" / "sessions_all.jsonl"
    records = _manifest_records(manifest)
    expected_sessions = int(summary["num_subjects"])
    if len(records) != expected_sessions or int(integrity["session_count"]) != expected_sessions:
        raise ValueError(
            "Session count mismatch among summary, integrity manifest, and sessions_all manifest"
        )

    referenced_tensors: set[Path] = set()
    total_steps = 0
    for record in records:
        tensor_path = _contained_file(root, manifest, record.get("tensor_file"))
        if tensor_path in referenced_tensors:
            raise ValueError(f"Duplicate tensor reference: {tensor_path}")
        referenced_tensors.add(tensor_path)
        expected_hash = record.get("sha256")
        actual_hash = _sha256(tensor_path)
        if expected_hash != actual_hash:
            raise ValueError(f"Tensor checksum mismatch: {tensor_path}")
        sample = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(sample, dict):
            raise TypeError(f"Tensor file does not contain a dictionary: {tensor_path}")
        missing_fields = sorted(REQUIRED_TENSOR_FIELDS.difference(sample))
        if missing_fields:
            raise ValueError(f"Missing tensor fields in {tensor_path}: {missing_fields}")
        length = int(sample["labels"].shape[0])
        if length != int(record["metadata"]["num_steps"]):
            raise ValueError(f"Timeline length mismatch: {tensor_path}")
        for name, value in sample.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"Non-finite tensor {name!r}: {tensor_path}")
        total_steps += length

    actual_tensors = set((root / "sessions").glob("session_*/timeline.pt"))
    if referenced_tensors != actual_tensors:
        missing_refs = sorted(
            path.relative_to(root).as_posix() for path in actual_tensors - referenced_tensors
        )
        missing_files = sorted(
            path.relative_to(root).as_posix() for path in referenced_tensors - actual_tensors
        )
        raise ValueError(
            f"Tensor inventory mismatch; unreferenced={missing_refs}, missing={missing_files}"
        )
    if total_steps != int(summary["num_steps"]):
        raise ValueError("Total timeline steps do not match dataset_summary.json")
    if int(alignment["aggregate"]["facial_source_records"]) != 0:
        raise ValueError("Facial CSV records are present in the processed package")

    return {
        "dataset": str(summary["dataset"]),
        "session_count": expected_sessions,
        "timeline_steps": total_steps,
        "tensor_count": len(referenced_tensors),
        "alignment_mode": str(summary["alignment_mode"]),
        "facial_source_records": 0,
    }


def _write_portable_metadata(root: Path, validation: dict[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        **validation,
        "package_kind": "self_contained_processed_timelines",
        "entry_manifest": "manifests/sessions_all.jsonl",
        "split_manifests": {
            split: f"manifests/sessions_{split}.jsonl" for split in ("train", "val", "test")
        },
        "tensor_format": "PyTorch weights-only dictionary",
        "paths_are_relative": True,
        "raw_source_files_included": False,
        "facial_csv_included": False,
        "checksum_manifest": "FILES.sha256",
        "redistribution_note": (
            "Technical portability does not grant redistribution rights. Confirm participant "
            "consent, data governance, and the selected license before public release."
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (root / "PORTABLE_PACKAGE.json").write_text(text, encoding="utf-8", newline="\n")


def _write_file_checksums(root: Path) -> tuple[int, int]:
    checksum_path = root / "FILES.sha256"
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != checksum_path
    )
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(files) + 1, sum(path.stat().st_size for path in files) + checksum_path.stat().st_size


def _write_deterministic_zip(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as handle:
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                relative = Path(root.name) / path.relative_to(root)
                info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                with path.open("rb") as source:
                    handle.writestr(info, source.read(), compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temporary_path, archive)
    finally:
        temporary_path.unlink(missing_ok=True)


def package_dataset(root: Path, archive: Path | None = None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    validation = validate_dataset(root)
    _write_portable_metadata(root, validation)
    file_count, package_bytes = _write_file_checksums(root)
    archive = (
        archive.expanduser().resolve()
        if archive is not None
        else root.parent / f"{root.name}.zip"
    )
    if archive == root or root in archive.parents:
        raise ValueError("Archive must be written outside the dataset directory")
    _write_deterministic_zip(root, archive)
    archive_hash = _sha256(archive)
    checksum_path = archive.with_suffix(f"{archive.suffix}.sha256")
    checksum_path.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8", newline="\n"
    )
    return {
        "status": "passed",
        **validation,
        "dataset_root": root.as_posix(),
        "file_count": file_count,
        "package_bytes": package_bytes,
        "archive": archive.as_posix(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "archive_checksum_file": checksum_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(package_dataset(args.dataset_root, args.archive), ensure_ascii=False))


if __name__ == "__main__":
    main()
