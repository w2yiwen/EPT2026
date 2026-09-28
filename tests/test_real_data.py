from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from eptnet.config import load_config
from eptnet.data import ManifestDataset, collate_multimodal, validate_batch
from eptnet.train import _build_continuous_session_loader

DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "raw" / "bci_truth_deception_v1" / "session_001"
PROCESSED_ROOT = DATA_ROOT / "processed" / "bci_truth_deception_v1"

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "source_manifest.json").is_file()
    or not (PROCESSED_ROOT / "dataset_summary.json").is_file(),
    reason=(
        "private BCI artifacts are not distributed with the public repository; "
        "run data preparation before the real-data integration tests"
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_prepared_summary_and_schema():
    summary = json.loads((PROCESSED_ROOT / "dataset_summary.json").read_text())
    schema = json.loads((PROCESSED_ROOT / "feature_schema.json").read_text())
    assert summary["num_steps"] == 1015
    assert summary["label_counts"] == {"0": 592, "1": 423}
    assert summary["windows_per_split"] == {"train": 22, "val": 4, "test": 4}
    assert summary["split_ranges"] == {
        "train": [0, 709],
        "val": [709, 859],
        "test": [859, 1015],
    }
    assert summary["feature_dimensions"] == {
        "eeg_time": 8,
        "eeg_spectral": 40,
        "physiology": 40,
        "video": 324,
        "audio": 50,
        "text": 768,
    }
    assert schema["label_mapping"] == {"0": "deception", "1": "truth"}
    assert schema["event_positive_label"] == 0
    assert schema["excluded_features"] == [
        *(f"EEG_CSP_{index}" for index in range(1, 7)),
        "EEG_LDA_1",
    ]


def test_normalized_source_checksums():
    manifest = json.loads((RAW_ROOT / "source_manifest.json").read_text())
    assert len(manifest["files"]) == 12
    for record in manifest["files"]:
        normalized = RAW_ROOT / record["normalized_name"]
        assert normalized.is_file()
        assert normalized.stat().st_size == record["bytes"]
        assert _sha256(normalized) == record["sha256"]


def test_manifests_stay_inside_split_ranges():
    summary = json.loads((PROCESSED_ROOT / "dataset_summary.json").read_text())
    for split, (split_start, split_end) in summary["split_ranges"].items():
        path = PROCESSED_ROOT / "manifests" / f"{split}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        assert len(records) == summary["windows_per_split"][split]
        for record in records:
            metadata = record["metadata"]
            assert split_start <= metadata["source_row_start"]
            assert metadata["source_row_end_exclusive"] <= split_end
        assert records[0]["metadata"]["source_row_start"] == split_start
        assert records[-1]["metadata"]["source_row_end_exclusive"] == split_end


def test_real_batch_contract_and_finite_values():
    dataset = ManifestDataset(
        str(PROCESSED_ROOT / "manifests" / "train.jsonl"), require_targets=True
    )
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_multimodal)))
    validate_batch(batch, require_targets=True, input_mode="precomputed_features")
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all()
    assert batch["eeg_time"].shape[-1] == 8
    assert batch["eeg_spectral"].shape[-1] == 40
    assert batch["physiology"].shape[-1] == 40
    assert batch["video"].shape[-1] == 324
    assert batch["audio"].shape[-1] == 50
    assert batch["text"].shape[-1] == 768


def test_training_loader_uses_each_of_the_709_train_rows_once():
    config = load_config("configs/default.yaml")
    dataset, loader = _build_continuous_session_loader(
        config["data"]["train_manifest"],
        settings=config["training"],
        device=torch.device("cpu"),
        seed=int(config["experiment"]["seed"]),
    )

    assert len(dataset) == 1
    assert len(loader) == 1
    batch = next(iter(loader))
    rows = batch["row_indices"][batch["sequence_mask"]]
    assert rows.tolist() == list(range(709))
    assert rows.unique().numel() == 709


def test_split_edges_are_valid_event_boundaries():
    for split in ("train", "val", "test"):
        dataset = ManifestDataset(
            str(PROCESSED_ROOT / "manifests" / f"{split}.jsonl"), require_targets=True
        )
        first = dataset[0]
        last = dataset[len(dataset) - 1]
        if bool(first["positive_mask"][0]):
            assert float(first["boundaries"][0, 0]) == 1.0
            assert float(first["offsets"][0, 0]) == 0.0
        if bool(last["positive_mask"][-1]):
            assert float(last["boundaries"][-1, 1]) == 1.0
            assert float(last["offsets"][-1, 1]) == 0.0
