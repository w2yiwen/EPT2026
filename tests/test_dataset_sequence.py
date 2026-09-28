from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler

from eptnet.data import CausalTrainingWindowDataset, StitchedManifestDataset, collate_multimodal
from eptnet.data.prepare_bci import (
    discover_feature_groups,
    excluded_eeg_features,
    snap_split_boundary,
)
from eptnet.train import (
    _build_causal_window_loader,
    _build_continuous_session_loader,
    _resolve_automatic_loss_weights,
)


def _window(rows: list[int]) -> dict:
    index = torch.tensor(rows, dtype=torch.long)
    values = index.float()
    length = len(rows)
    labels = (index % 2).long()
    return {
        "eeg_time": values[:, None].repeat(1, 2),
        "eeg_spectral": values[:, None].repeat(1, 3),
        "physiology": values[:, None].repeat(1, 2),
        "video": values[:, None].repeat(1, 2),
        "audio": values[:, None].repeat(1, 2),
        "text": values[:, None].repeat(1, 2),
        "modality_mask": torch.ones(length, 3, dtype=torch.bool),
        "labels": labels,
        "boundaries": torch.stack(
            (index.remainder(3).eq(0), index.remainder(3).eq(1)), dim=-1
        ).float(),
        "offsets": torch.zeros(length, 2),
        "positive_mask": labels == 0,
        "target_mask": torch.ones(length, dtype=torch.bool),
        "timestamps": torch.stack((values, values + 0.5), dim=-1),
        "row_indices": index,
        "words": [f"word-{row}" for row in rows],
    }


def _manifest(tmp_path: Path, windows: list[tuple[str, str, dict]]) -> Path:
    records = []
    for sample_id, session_id, sample in windows:
        tensor_file = tmp_path / f"{sample_id}.pt"
        torch.save(sample, tensor_file)
        records.append(
            {
                "sample_id": sample_id,
                "tensor_file": tensor_file.name,
                "metadata": {
                    "dataset": "unit_test",
                    "session_id": session_id,
                    "split": "test",
                },
            }
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def test_stitched_manifest_deduplicates_sorts_and_collates_metadata(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        [
            ("late", "session_b", _window([12, 13, 14])),
            ("early", "session_b", _window([10, 11, 12])),
        ],
    )
    dataset = StitchedManifestDataset(str(manifest), require_targets=True)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["row_indices"].tolist() == [10, 11, 12, 13, 14]
    assert sample["words"] == [f"word-{row}" for row in range(10, 15)]
    assert sample["metadata"]["source_window_count"] == 2
    assert sample["eeg_time"].shape == (5, 2)

    batch = collate_multimodal([sample])
    assert batch["row_indices"].shape == (1, 5)
    assert batch["row_indices"][0].tolist() == [10, 11, 12, 13, 14]
    assert batch["timestamps"].shape == (1, 5, 2)
    assert torch.equal(batch["timestamps"][0], sample["timestamps"])
    assert torch.equal(batch["target_mask"], torch.ones(1, 5, dtype=torch.bool))


def test_stitched_manifest_materialization_freezes_the_run_inputs(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        [("only", "session_a", _window([0, 1, 2]))],
    )
    dataset = StitchedManifestDataset(str(manifest), require_targets=True)
    dataset.materialize()
    (tmp_path / "only.pt").unlink()

    assert dataset[0]["row_indices"].tolist() == [0, 1, 2]


def test_stitched_manifest_rejects_inconsistent_duplicate_field(tmp_path: Path):
    first = _window([0, 1, 2])
    second = _window([2, 3, 4])
    second["audio"][0, 0] += 1.0
    manifest = _manifest(
        tmp_path,
        [("first", "session_a", first), ("second", "session_a", second)],
    )
    dataset = StitchedManifestDataset(str(manifest))
    with pytest.raises(ValueError, match="Inconsistent duplicate row 2 field 'audio'"):
        _ = dataset[0]


def test_stitched_manifest_rejects_non_contiguous_rows(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        [
            ("first", "session_a", _window([0, 1])),
            ("second", "session_a", _window([3, 4])),
        ],
    )
    dataset = StitchedManifestDataset(str(manifest))
    with pytest.raises(ValueError, match="expected 2, found 3"):
        _ = dataset[0]


def test_training_loader_uses_one_unshuffled_709_step_session(tmp_path: Path):
    total_steps = 709
    window_length = 64
    starts = list(range(0, total_steps - window_length + 1, 32))
    if starts[-1] + window_length < total_steps:
        starts.append(total_steps - window_length)
    manifest = _manifest(
        tmp_path,
        [
            (
                f"window-{start:04d}",
                "session_train",
                _window(list(range(start, min(start + window_length, total_steps)))),
            )
            for start in starts
        ],
    )
    settings = {
        "sequence_protocol": "continuous_session",
        "batch_size": 1,
        "num_workers": 0,
    }

    dataset, loader = _build_continuous_session_loader(
        str(manifest), settings=settings, device=torch.device("cpu"), seed=42
    )

    assert len(dataset) == 1
    assert len(loader) == 1
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.drop_last is False
    batch = next(iter(loader))
    valid_rows = batch["row_indices"][batch["sequence_mask"]]
    assert valid_rows.tolist() == list(range(total_steps))
    assert valid_rows.unique().numel() == total_steps
    assert int(batch["sequence_mask"].sum()) == total_steps


def test_continuous_loader_rejects_other_or_multi_session_batch_protocol(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        [("window", "session_train", _window([0, 1, 2]))],
    )
    base_settings = {
        "sequence_protocol": "continuous_session",
        "batch_size": 1,
        "num_workers": 0,
    }
    windowed = {**base_settings, "sequence_protocol": "overlap_windows"}
    with pytest.raises(ValueError, match=r"sequence_protocol must be 'continuous_session'"):
        _build_continuous_session_loader(
            str(manifest), settings=windowed, device=torch.device("cpu"), seed=42
        )

    batched = {**base_settings, "batch_size": 2}
    with pytest.raises(ValueError, match="batch_size must be 1"):
        _build_continuous_session_loader(
            str(manifest), settings=batched, device=torch.device("cpu"), seed=42
        )


def test_causal_window_loader_masks_warmup_and_keeps_validation_rows_unique(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        [("session", "session_train", _window(list(range(160))))],
    )
    settings = {
        "sequence_protocol": "causal_windows",
        "batch_size": 1,
        "num_workers": 0,
        "window_size": 128,
        "window_stride": 32,
        "window_warmup_steps": 32,
        "positive_window_oversample": 1,
    }
    dataset, loader, unique = _build_causal_window_loader(
        str(manifest),
        settings=settings,
        positive_class=0,
        device=torch.device("cpu"),
        seed=42,
    )

    assert isinstance(dataset, CausalTrainingWindowDataset)
    assert len(unique) == 1
    assert len(dataset) == 2
    assert len(loader) == 2
    first = dataset[0]
    second = dataset[1]
    assert first["row_indices"].tolist() == list(range(128))
    assert bool(first["target_mask"][:32].all())
    assert second["row_indices"].tolist() == list(range(32, 160))
    assert not bool(second["target_mask"][:32].any())
    assert bool(second["target_mask"][32:].all())
    assert unique[0]["row_indices"].unique().numel() == 160


def test_positive_causal_windows_can_be_oversampled_without_changing_unique_data(
    tmp_path: Path,
):
    sample = _window(list(range(96)))
    sample["labels"].fill_(1)
    sample["positive_mask"].fill_(False)
    sample["labels"][60] = 0
    sample["positive_mask"][60] = True
    manifest = _manifest(tmp_path, [("session", "session_train", sample)])
    unique = StitchedManifestDataset(str(manifest))
    windows = CausalTrainingWindowDataset(
        unique,
        window_size=64,
        stride=32,
        warmup_steps=16,
        positive_class=0,
        positive_window_oversample=2,
    )

    assert len(unique) == 1
    assert len(windows) == 4
    assert sum(item["metadata"]["contains_supervised_positive"] for item in windows) == 4
    assert {item["metadata"]["oversample_replica"] for item in windows} == {0, 1}


def test_automatic_loss_weights_count_each_overlapping_row_once(tmp_path: Path):
    manifest = _manifest(
        tmp_path,
        [
            ("first", "session_train", _window([0, 1, 2])),
            ("second", "session_train", _window([2, 3, 4])),
        ],
    )
    dataset = StitchedManifestDataset(str(manifest), require_targets=True)
    config = {
        "data": {"num_classes": 2},
        "loss": {"class_weights": "auto", "boundary_pos_weight": "auto"},
    }

    statistics = _resolve_automatic_loss_weights(config, dataset)

    assert statistics["unique_training_steps"] == 5
    assert statistics["class_counts"] == [3, 2]
    assert statistics["boundary_positive_counts"] == [2, 2]
    assert statistics["class_weights"] == pytest.approx([5 / 6, 5 / 4])
    assert statistics["boundary_pos_weight"] == pytest.approx([3 / 2, 3 / 2])


def test_event_safe_split_snaps_to_nearest_complete_event_boundary():
    labels = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
    assert snap_split_boundary(labels, 4, 1, 8, event_label=0) == 5
    assert snap_split_boundary(labels, 8, 1, 9, event_label=0) == 7


def test_feature_discovery_excludes_unverified_supervised_eeg_features():
    columns = (
        ["Word", "Start_Time", "End_Time", "Label"]
        + ["duration"]
        + [f"audio_{index}" for index in range(49)]
        + ["Landmark_1_X_mean"]
        + [f"video_{index}" for index in range(323)]
        + ["0"]
        + [str(index) for index in range(1, 768)]
        + ["HR_Mean"]
        + [f"physiology_{index}" for index in range(39)]
        + [f"EEG_Channel_{index}" for index in range(1, 9)]
        + [f"EEG_Advanced_{index}" for index in range(1, 41)]
        + [f"EEG_CSP_{index}" for index in range(1, 7)]
        + ["EEG_LDA_1"]
    )
    groups = discover_feature_groups(columns)
    excluded = excluded_eeg_features(columns)
    assert len(groups["eeg"]) == 48
    assert groups["eeg"][:8] == [f"EEG_Channel_{index}" for index in range(1, 9)]
    assert groups["eeg"][8:] == [f"EEG_Advanced_{index}" for index in range(1, 41)]
    assert excluded == [f"EEG_CSP_{index}" for index in range(1, 7)] + ["EEG_LDA_1"]
