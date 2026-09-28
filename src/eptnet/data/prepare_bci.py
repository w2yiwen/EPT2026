from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

DATASET_NAME = "bci_truth_deception_v1"
SESSION_ID = "session_001"
# Source-side participant filenames are intentionally represented by strict
# patterns rather than embedded personal identifiers.  Every pattern must
# resolve to exactly one file; ambiguity is an error rather than an arbitrary
# first-match choice.
SOURCE_FILE_SPECS = (
    ("README.md", "dataset_source_notes.md"),
    ("01_raw_acquisition/*.mp4", f"{SESSION_ID}_video.mp4"),
    ("01_raw_acquisition/converted.wav", f"{SESSION_ID}_audio.wav"),
    ("01_raw_acquisition/EEG.txt", f"{SESSION_ID}_eeg.csv"),
    ("01_raw_acquisition/ch2.txt", f"{SESSION_ID}_ppg_red.txt"),
    ("01_raw_acquisition/ch3.txt", f"{SESSION_ID}_ppg_infrared.txt"),
    ("02_alignment_and_labels/word_timestamps.csv", f"{SESSION_ID}_word_timestamps.csv"),
    (
        "02_alignment_and_labels/word_timestamps_improved.csv",
        f"{SESSION_ID}_word_timestamps_refined.csv",
    ),
    ("02_alignment_and_labels/1.csv", f"{SESSION_ID}_labels_primary.csv"),
    (
        "02_alignment_and_labels/final_result.csv",
        f"{SESSION_ID}_labels_alternative.csv",
    ),
    (
        "02_alignment_and_labels/*_marked.docx",
        f"{SESSION_ID}_manual_annotations.docx",
    ),
    (
        "03_model_direct_input/multi_final.csv",
        f"{SESSION_ID}_multimodal_features.csv",
    ),
)
EXPECTED_GROUP_SIZES = {
    "audio": 50,
    "video": 324,
    "text": 768,
    "physiology": 40,
    "eeg": 55,
}
EEG_TIME_FEATURE_COUNT = 8
EEG_SPECTRAL_FEATURE_COUNT = 40
EXCLUDED_EEG_PREFIXES = ("EEG_CSP_", "EEG_LDA_")
EXCLUDED_FEATURE_REASON = (
    "CSP and LDA are label-supervised transforms whose fit scope is not available "
    "in the staged bundle; excluding them prevents unverified target leakage."
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_file(source_root: Path, source_pattern: str) -> Path:
    """Resolve one required source artifact without encoding participant names."""
    matches = sorted(path for path in source_root.glob(source_pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"Required source pattern has no file: {source_pattern!r} under {source_root}"
        )
    if len(matches) > 1:
        relative_matches = [path.relative_to(source_root).as_posix() for path in matches]
        raise ValueError(
            f"Required source pattern {source_pattern!r} is ambiguous; "
            f"expected one file, found {relative_matches}"
        )
    return matches[0]


def stage_source_bundle(source_root: Path, raw_root: Path, force: bool = False) -> dict:
    """Copy the immutable source bundle to normalized, provenance-tracked names."""
    source_root = source_root.expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for source_pattern, normalized_name in SOURCE_FILE_SPECS:
        source = _resolve_source_file(source_root, source_pattern)
        source_relative = source.relative_to(source_root).as_posix()
        destination = raw_root / normalized_name
        source_hash = sha256_file(source)
        if destination.exists():
            destination_hash = sha256_file(destination)
            if destination_hash != source_hash:
                if not force:
                    raise FileExistsError(
                        f"Normalized file exists with a different hash: {destination}. "
                        "Use --force only after verifying the source bundle."
                    )
                shutil.copy2(source, destination)
        else:
            shutil.copy2(source, destination)
        copied_hash = sha256_file(destination)
        if copied_hash != source_hash:
            raise OSError(f"Checksum mismatch after copying {source} to {destination}")
        entries.append(
            {
                "source_relative_path": source_relative,
                "normalized_name": normalized_name,
                "bytes": source.stat().st_size,
                "sha256": source_hash,
            }
        )
    manifest = {
        "dataset": DATASET_NAME,
        "session_id": SESSION_ID,
        "source_root": str(source_root),
        "files": entries,
    }
    (raw_root / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def load_staged_source_bundle(raw_root: Path) -> dict:
    """Validate and reuse an already staged immutable source bundle."""
    manifest_path = raw_root / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No staged source manifest found at {manifest_path}; provide --source once"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET_NAME or manifest.get("session_id") != SESSION_ID:
        raise ValueError(f"Unexpected staged source identity in {manifest_path}")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError(f"Invalid files list in {manifest_path}")
    by_name = {entry.get("normalized_name"): entry for entry in entries}
    expected_names = {normalized_name for _, normalized_name in SOURCE_FILE_SPECS}
    if set(by_name) != expected_names:
        raise ValueError(
            f"Staged source files do not match the expected bundle: "
            f"expected {sorted(expected_names)}, got {sorted(by_name)}"
        )
    for normalized_name in sorted(expected_names):
        record = by_name[normalized_name]
        path = raw_root / normalized_name
        if not path.is_file():
            raise FileNotFoundError(f"Staged source file is missing: {path}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise OSError(f"Staged source size mismatch: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise OSError(f"Staged source checksum mismatch: {path}")
    return manifest


def parse_tempo(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value)
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if len(matches) != 1:
        raise ValueError(f"Expected one numeric tempo value, got {value!r}")
    return float(matches[0])


def discover_feature_groups(columns: Sequence[str]) -> dict[str, list[str]]:
    names = list(map(str, columns))
    sentinels = {
        "audio": names.index("duration"),
        "video": names.index("Landmark_1_X_mean"),
        "text": names.index("0"),
        "physiology": names.index("HR_Mean"),
        "eeg": names.index("EEG_Channel_1"),
        "end": len(names),
    }
    raw_groups = {
        "audio": names[sentinels["audio"] : sentinels["video"]],
        "video": names[sentinels["video"] : sentinels["text"]],
        "text": names[sentinels["text"] : sentinels["physiology"]],
        "physiology": names[sentinels["physiology"] : sentinels["eeg"]],
        "eeg": names[sentinels["eeg"] : sentinels["end"]],
    }
    actual = {name: len(values) for name, values in raw_groups.items()}
    if actual != EXPECTED_GROUP_SIZES:
        raise ValueError(
            f"Unexpected feature schema: expected {EXPECTED_GROUP_SIZES}, got {actual}"
        )
    eeg_time = [name for name in raw_groups["eeg"] if name.startswith("EEG_Channel_")]
    eeg_spectral = [name for name in raw_groups["eeg"] if name.startswith("EEG_Advanced_")]
    excluded = [name for name in raw_groups["eeg"] if name.startswith(EXCLUDED_EEG_PREFIXES)]
    recognized = set(eeg_time) | set(eeg_spectral) | set(excluded)
    unknown = [name for name in raw_groups["eeg"] if name not in recognized]
    if unknown:
        raise ValueError(f"Unrecognized EEG features: {unknown}")
    if len(eeg_time) != EEG_TIME_FEATURE_COUNT or len(eeg_spectral) != EEG_SPECTRAL_FEATURE_COUNT:
        raise ValueError(
            "Expected 8 EEG_Channel and 40 EEG_Advanced features, got "
            f"{len(eeg_time)} and {len(eeg_spectral)}"
        )
    groups = dict(raw_groups)
    groups["eeg"] = eeg_time + eeg_spectral
    return groups


def excluded_eeg_features(columns: Sequence[str]) -> list[str]:
    return [name for name in map(str, columns) if name.startswith(EXCLUDED_EEG_PREFIXES)]


def numeric_group(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    selected = frame.loc[:, list(columns)].copy()
    if "tempo" in selected.columns:
        selected["tempo"] = selected["tempo"].map(parse_tempo)
    selected = selected.apply(pd.to_numeric, errors="coerce")
    return selected.to_numpy(dtype=np.float64)


def fit_preprocessor(
    train_frame: pd.DataFrame, groups: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for name, columns in groups.items():
        values = numeric_group(train_frame, columns)
        median = np.nanmedian(values, axis=0)
        if np.isnan(median).any():
            missing = np.flatnonzero(np.isnan(median)).tolist()
            raise ValueError(f"All training values are missing in {name} columns {missing}")
        filled = np.where(np.isnan(values), median[None, :], values)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale[scale < 1e-8] = 1.0
        stats[name] = {"median": median, "mean": mean, "scale": scale}
    return stats


def transform_features(
    frame: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    stats: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    transformed: dict[str, np.ndarray] = {}
    for name, columns in groups.items():
        values = numeric_group(frame, columns)
        group_stats = stats[name]
        values = np.where(np.isnan(values), group_stats["median"][None, :], values)
        values = (values - group_stats["mean"][None, :]) / group_stats["scale"][None, :]
        values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values remain after preprocessing group {name}")
        transformed[name] = values
    eeg = transformed.pop("eeg")
    transformed["eeg_time"] = eeg[:, :EEG_TIME_FEATURE_COUNT]
    transformed["eeg_spectral"] = eeg[:, EEG_TIME_FEATURE_COUNT:]
    return transformed


def build_event_targets(labels: np.ndarray, event_label: int = 0) -> dict[str, np.ndarray]:
    positive = labels.astype(np.int64) == event_label
    boundaries = np.zeros((len(labels), 2), dtype=np.float32)
    offsets = np.zeros((len(labels), 2), dtype=np.float32)
    index = 0
    while index < len(labels):
        if not positive[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(labels) and positive[index + 1]:
            index += 1
        end = index
        boundaries[start, 0] = 1.0
        boundaries[end, 1] = 1.0
        positions = np.arange(start, end + 1)
        offsets[start : end + 1, 0] = positions - start
        offsets[start : end + 1, 1] = end - positions
        index += 1
    return {
        "labels": labels.astype(np.int64),
        "boundaries": boundaries,
        "offsets": offsets,
        "positive_mask": positive,
    }


def snap_split_boundary(
    labels: np.ndarray,
    nominal_index: int,
    minimum_index: int,
    maximum_index: int,
    event_label: int = 0,
) -> int:
    """Move a split point to the nearest boundary that does not cut an event."""
    labels = np.asarray(labels, dtype=np.int64)
    minimum_index = max(1, int(minimum_index))
    maximum_index = min(len(labels) - 1, int(maximum_index))
    if minimum_index > maximum_index:
        raise ValueError(
            f"No legal split interval: minimum {minimum_index}, maximum {maximum_index}"
        )
    candidates = [
        index
        for index in range(minimum_index, maximum_index + 1)
        if not (labels[index - 1] == event_label and labels[index] == event_label)
    ]
    if not candidates:
        raise ValueError(
            f"No event-safe split boundary between {minimum_index} and {maximum_index}"
        )
    return min(candidates, key=lambda index: (abs(index - nominal_index), index))


def window_starts(length: int, window_size: int, stride: int, min_window_size: int) -> list[int]:
    if length < min_window_size:
        raise ValueError(f"Split length {length} is smaller than min_window_size {min_window_size}")
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def write_windows(
    split_name: str,
    split_start: int,
    split_end: int,
    features: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    frame: pd.DataFrame,
    processed_root: Path,
    window_size: int,
    stride: int,
    min_window_size: int,
) -> list[dict]:
    window_dir = processed_root / "sessions" / SESSION_ID / "windows" / split_name
    manifest_dir = processed_root / "manifests"
    window_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    split_length = split_end - split_start
    records: list[dict] = []

    def as_compact_tensor(values: np.ndarray) -> torch.Tensor:
        # A view would make torch.save serialize the full session-level backing storage.
        return torch.from_numpy(np.ascontiguousarray(values).copy())

    for window_index, local_start in enumerate(
        window_starts(split_length, window_size, stride, min_window_size)
    ):
        local_end = min(local_start + window_size, split_length)
        global_start = split_start + local_start
        global_end = split_start + local_end
        sample_id = f"{SESSION_ID}_{split_name}_window_{window_index:04d}"
        filename = f"{sample_id}.pt"
        tensor_path = window_dir / filename
        sample = {
            "eeg_time": as_compact_tensor(features["eeg_time"][global_start:global_end]),
            "eeg_spectral": as_compact_tensor(features["eeg_spectral"][global_start:global_end]),
            "physiology": as_compact_tensor(features["physiology"][global_start:global_end]),
            "video": as_compact_tensor(features["video"][global_start:global_end]),
            "audio": as_compact_tensor(features["audio"][global_start:global_end]),
            "text": as_compact_tensor(features["text"][global_start:global_end]),
            "modality_mask": torch.ones(global_end - global_start, 3, dtype=torch.bool),
            "labels": as_compact_tensor(targets["labels"][global_start:global_end]),
            "boundaries": as_compact_tensor(targets["boundaries"][global_start:global_end]),
            "offsets": as_compact_tensor(targets["offsets"][global_start:global_end]),
            "positive_mask": as_compact_tensor(targets["positive_mask"][global_start:global_end]),
            "timestamps": as_compact_tensor(
                frame.loc[global_start : global_end - 1, ["Start_Time", "End_Time"]].to_numpy(
                    dtype=np.float32
                )
            ),
            "row_indices": torch.arange(global_start, global_end, dtype=torch.long),
            "words": frame.loc[global_start : global_end - 1, "Word"].astype(str).tolist(),
        }
        torch.save(sample, tensor_path)
        records.append(
            {
                "sample_id": sample_id,
                "tensor_file": (f"../sessions/{SESSION_ID}/windows/{split_name}/{filename}"),
                "metadata": {
                    "dataset": DATASET_NAME,
                    "session_id": SESSION_ID,
                    "split": split_name,
                    "source_row_start": global_start,
                    "source_row_end_exclusive": global_end,
                    "start_time_seconds": float(frame.iloc[global_start]["Start_Time"]),
                    "end_time_seconds": float(frame.iloc[global_end - 1]["End_Time"]),
                },
            }
        )
    manifest_path = manifest_dir / f"{split_name}.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def prepare_dataset(
    source_root: Path | None,
    data_root: Path,
    window_size: int = 64,
    stride: int = 32,
    min_window_size: int = 16,
    force: bool = False,
) -> dict:
    raw_root = data_root / "raw" / DATASET_NAME / SESSION_ID
    processed_root = data_root / "processed" / DATASET_NAME
    if source_root is None:
        source_manifest = load_staged_source_bundle(raw_root)
    else:
        source_manifest = stage_source_bundle(source_root, raw_root, force=force)

    feature_path = raw_root / f"{SESSION_ID}_multimodal_features.csv"
    label_path = raw_root / f"{SESSION_ID}_labels_primary.csv"
    frame = pd.read_csv(feature_path, encoding="utf-8-sig", low_memory=False)
    labels = pd.read_csv(label_path, encoding="utf-8-sig")
    required_metadata = ["Word", "Start_Time", "End_Time", "Label"]
    if list(frame.columns[:4]) != required_metadata or list(labels.columns) != required_metadata:
        raise ValueError("Unexpected metadata columns in feature or primary-label table")
    if len(frame) != len(labels):
        raise ValueError("Feature and primary-label row counts differ")
    if not frame["Word"].astype(str).equals(labels["Word"].astype(str)):
        raise ValueError("Feature and primary-label words are not aligned")
    if not frame["Label"].equals(labels["Label"]):
        raise ValueError("Feature and primary-label targets are not aligned")
    if not np.allclose(frame[["Start_Time", "End_Time"]], labels[["Start_Time", "End_Time"]]):
        raise ValueError("Feature and primary-label timestamps are not aligned")
    if (frame["End_Time"] < frame["Start_Time"]).any() or (
        frame["Start_Time"].diff().dropna() < 0
    ).any():
        raise ValueError("Primary timestamps are invalid or non-monotonic")

    all_columns = list(map(str, frame.columns))
    groups = discover_feature_groups(all_columns)
    excluded_features = excluded_eeg_features(all_columns)
    total = len(frame)
    nominal_train_end = math.floor(total * 0.70)
    nominal_val_end = math.floor(total * 0.85)
    nominal_split_ranges = {
        "train": (0, nominal_train_end),
        "val": (nominal_train_end, nominal_val_end),
        "test": (nominal_val_end, total),
    }
    labels_array = frame["Label"].to_numpy(dtype=np.int64)
    train_end = snap_split_boundary(
        labels_array,
        nominal_train_end,
        min_window_size,
        total - 2 * min_window_size,
        event_label=0,
    )
    val_end = snap_split_boundary(
        labels_array,
        nominal_val_end,
        train_end + min_window_size,
        total - min_window_size,
        event_label=0,
    )
    split_ranges = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total),
    }
    stats = fit_preprocessor(frame.iloc[:train_end], groups)
    features = transform_features(frame, groups, stats)
    targets = {
        "labels": labels_array.copy(),
        "boundaries": np.zeros((total, 2), dtype=np.float32),
        "offsets": np.zeros((total, 2), dtype=np.float32),
        "positive_mask": np.zeros(total, dtype=bool),
    }
    for split_start, split_end in split_ranges.values():
        local_targets = build_event_targets(labels_array[split_start:split_end], event_label=0)
        for key in ("boundaries", "offsets", "positive_mask"):
            targets[key][split_start:split_end] = local_targets[key]

    processed_root.mkdir(parents=True, exist_ok=True)
    np.savez(
        processed_root / "normalization_stats.npz",
        **{
            f"{group}_{stat_name}": value
            for group, group_stats in stats.items()
            for stat_name, value in group_stats.items()
        },
    )
    feature_schema = {
        "dataset": DATASET_NAME,
        "session_id": SESSION_ID,
        "metadata_columns": required_metadata,
        "feature_groups": groups,
        "model_mapping": {
            "eeg_time": groups["eeg"][:EEG_TIME_FEATURE_COUNT],
            "eeg_spectral": groups["eeg"][EEG_TIME_FEATURE_COUNT:],
            "physiology": groups["physiology"],
            "video": groups["video"],
            "audio": groups["audio"],
            "text": groups["text"],
        },
        "label_mapping": {"0": "deception", "1": "truth"},
        "event_positive_label": 0,
        "excluded_features": excluded_features,
        "exclusion_reason": EXCLUDED_FEATURE_REASON,
    }
    (processed_root / "feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    manifests = {}
    for split_name, (split_start, split_end) in split_ranges.items():
        manifests[split_name] = write_windows(
            split_name,
            split_start,
            split_end,
            features,
            targets,
            frame,
            processed_root,
            window_size,
            stride,
            min_window_size,
        )
    all_manifest_path = processed_root / "manifests" / "all.jsonl"
    with all_manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for split_name in ("train", "val", "test"):
            for record in manifests[split_name]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dataset": DATASET_NAME,
        "session_id": SESSION_ID,
        "num_steps": total,
        "time_range_seconds": [float(frame["Start_Time"].min()), float(frame["End_Time"].max())],
        "label_counts": {
            str(label): int(count)
            for label, count in frame["Label"].value_counts().sort_index().items()
        },
        "feature_dimensions": {
            "eeg_time": EEG_TIME_FEATURE_COUNT,
            "eeg_spectral": EEG_SPECTRAL_FEATURE_COUNT,
            "physiology": 40,
            "video": 324,
            "audio": 50,
            "text": 768,
        },
        "nominal_split_ranges": {
            name: [start, end] for name, (start, end) in nominal_split_ranges.items()
        },
        "actual_split_ranges": {name: [start, end] for name, (start, end) in split_ranges.items()},
        "split_ranges": {name: [start, end] for name, (start, end) in split_ranges.items()},
        "windows_per_split": {name: len(records) for name, records in manifests.items()},
        "window_size": window_size,
        "stride": stride,
        "source_file_count": len(source_manifest["files"]),
        "limitations": [
            "single participant/session",
            "chronological within-session split is not subject-independent evaluation",
            "precomputed features are used for the executable path",
        ],
    }
    (processed_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    session_root = processed_root / "sessions" / SESSION_ID
    session_root.mkdir(parents=True, exist_ok=True)
    (session_root / "session_summary.json").write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "session_id": SESSION_ID,
                "num_steps": total,
                "split_ranges": {name: [start, end] for name, (start, end) in split_ranges.items()},
                "windows_per_split": {name: len(records) for name, records in manifests.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (processed_root / "session_index.json").write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "sessions": [
                    {
                        "session_id": SESSION_ID,
                        "raw_path": f"raw/{DATASET_NAME}/{SESSION_ID}",
                        "processed_path": f"processed/{DATASET_NAME}/sessions/{SESSION_ID}",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    raw_dataset_root = raw_root.parent
    (raw_dataset_root / "session_index.json").write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "sessions": [
                    {
                        "session_id": SESSION_ID,
                        "raw_path": f"raw/{DATASET_NAME}/{SESSION_ID}",
                        "processed_path": f"processed/{DATASET_NAME}/sessions/{SESSION_ID}",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    legacy_window_root = (processed_root / "windows").resolve()
    if legacy_window_root.parent != processed_root.resolve():
        raise ValueError(f"Refusing to remove unexpected legacy path: {legacy_window_root}")
    if legacy_window_root.exists():
        shutil.rmtree(legacy_window_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage and prepare the supplied BCI bundle")
    parser.add_argument(
        "--source",
        help=(
            "Path to BCI_original_inputs_bundle. Omit it to validate and reuse the "
            "already staged raw/source_manifest.json bundle."
        ),
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min-window-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = prepare_dataset(
        Path(args.source) if args.source else None,
        Path(args.data_root),
        window_size=args.window_size,
        stride=args.stride,
        min_window_size=args.min_window_size,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
