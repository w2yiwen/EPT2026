from __future__ import annotations

import csv
import json
from pathlib import Path

from eptnet.data.audit import build_data_audit, render_markdown


def _write_manifest(path: Path, ranges: list[tuple[int, int]], split: str) -> None:
    records = [
        {
            "sample_id": f"{split}_{index}",
            "tensor_file": f"unused_{index}.pt",
            "metadata": {
                "dataset": "bci_truth_deception_v1",
                "session_id": "session_001",
                "split": split,
                "source_row_start": start,
                "source_row_end_exclusive": end,
            },
        }
        for index, (start, end) in enumerate(ranges)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_build_data_audit_is_read_only_and_counts_unique_rows(tmp_path: Path):
    data_root = tmp_path / "data"
    raw = data_root / "raw" / "bci_truth_deception_v1" / "session_001"
    processed = data_root / "processed" / "bci_truth_deception_v1"
    manifests = processed / "manifests"
    raw.mkdir(parents=True)
    manifests.mkdir(parents=True)

    labels = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    primary_path = raw / "session_001_labels_primary.csv"
    feature_path = raw / "session_001_multimodal_features.csv"
    header = [
        "Word",
        "Start_Time",
        "End_Time",
        "Label",
        "eeg_t",
        "eeg_s",
        "phys",
        "video",
        "audio",
        "text",
    ]
    with (
        primary_path.open("w", encoding="utf-8", newline="") as primary,
        feature_path.open("w", encoding="utf-8", newline="") as features,
    ):
        primary_writer = csv.writer(primary)
        feature_writer = csv.writer(features)
        primary_writer.writerow(header[:4])
        feature_writer.writerow(header)
        for index, label in enumerate(labels):
            metadata = [f"w{index}", float(index), float(index) + 0.5, label]
            primary_writer.writerow(metadata)
            repeated = float(index % 4)
            feature_writer.writerow(
                metadata + [index, index % 3, index / 2, index, repeated, repeated]
            )

    split_ranges = {"train": [0, 4], "val": [4, 8], "test": [8, 12]}
    summary = {
        "dataset": "bci_truth_deception_v1",
        "num_steps": len(labels),
        "label_counts": {"0": 5, "1": 7},
        "feature_dimensions": {
            name: 1 for name in ("eeg_time", "eeg_spectral", "physiology", "video", "audio", "text")
        },
        "nominal_split_ranges": split_ranges,
        "actual_split_ranges": split_ranges,
        "split_ranges": split_ranges,
        "windows_per_split": {"train": 2, "val": 1, "test": 1},
        "limitations": ["synthetic unit fixture"],
    }
    schema = {
        "event_positive_label": 0,
        "model_mapping": {
            "eeg_time": ["eeg_t"],
            "eeg_spectral": ["eeg_s"],
            "physiology": ["phys"],
            "video": ["video"],
            "audio": ["audio"],
            "text": ["text"],
        },
    }
    (processed / "dataset_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (processed / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    _write_manifest(manifests / "train.jsonl", [(0, 3), (2, 4)], "train")
    _write_manifest(manifests / "val.jsonl", [(4, 8)], "val")
    _write_manifest(manifests / "test.jsonl", [(8, 12)], "test")

    report = build_data_audit(data_root)
    assert report["label_integrity"]["labels_modified"] is False
    assert report["overall"]["unique_steps"] == 12
    assert report["overall"]["event_runs"] == 3
    assert report["all_split_boundaries_event_safe"] is True
    assert report["splits"]["train"]["windows"]["duplicate_observation_rate"] == 0.2
    assert "combined_model_input" in report["feature_leakage_screen"]
    markdown = render_markdown(report)
    assert "no smoothing, merging, or relabeling" in markdown
    assert "No completed seed-42 metric file" in markdown
