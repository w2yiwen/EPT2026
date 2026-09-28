"""Audit entry point for behavior-feature migration equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from eptnet.config import load_config
from eptnet.data.dataset import ManifestDataset, collate_multimodal
from eptnet.models import build_model

UNCHANGED_TENSOR_FIELDS = (
    "eeg_time",
    "eeg_spectral",
    "physiology",
    "video",
    "audio",
    "physiology_mask",
    "modality_mask",
    "target_mask",
    "labels",
    "boundaries",
    "offsets",
    "positive_mask",
    "timestamps",
    "row_indices",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_minimal_migration(new_root: Path, legacy_root: Path) -> dict[str, object]:
    windows = 0
    available_text_rows = 0
    changed_text_rows = 0
    for split in ("train", "val", "test"):
        new_dataset = ManifestDataset(str(new_root / "manifests" / f"{split}.jsonl"))
        legacy_dataset = ManifestDataset(str(legacy_root / "manifests" / f"{split}.jsonl"))
        if len(new_dataset) != len(legacy_dataset):
            raise ValueError(f"{split} window count changed during behavior migration")
        for index in range(len(new_dataset)):
            new_sample = new_dataset[index]
            legacy_sample = legacy_dataset[index]
            if new_sample["sample_id"] != legacy_sample["sample_id"]:
                raise ValueError(f"{split} sample order changed at window {index}")
            if new_sample["words"] != legacy_sample["words"]:
                raise ValueError(f"{split} transcript alignment changed at window {index}")
            for field in UNCHANGED_TENSOR_FIELDS:
                if not torch.equal(new_sample[field], legacy_sample[field]):
                    raise ValueError(
                        f"Field {field!r} changed outside the requested text migration "
                        f"for {new_sample['sample_id']}"
                    )
            text_mask = new_sample["modality_mask"][:, 2].bool()
            available_text_rows += int(text_mask.sum().item())
            changed_text_rows += int(
                torch.any(new_sample["text"] != legacy_sample["text"], dim=-1)[text_mask]
                .sum()
                .item()
            )
            windows += 1
    if available_text_rows == 0 or changed_text_rows != available_text_rows:
        raise ValueError(
            "MacBERT replacement did not change every available legacy text row: "
            f"{changed_text_rows}/{available_text_rows}"
        )
    return {
        "status": "passed",
        "windows_compared": windows,
        "unchanged_tensor_fields": list(UNCHANGED_TENSOR_FIELDS),
        "transcript_alignment_unchanged": True,
        "available_window_text_rows_replaced": changed_text_rows,
        "available_window_text_rows_total": available_text_rows,
    }


def verify(config_path: Path, legacy_root: Path) -> dict[str, object]:
    config = load_config(str(config_path))
    manifest = Path(config["data"]["val_manifest"])
    dataset = ManifestDataset(str(manifest))
    sample = dataset[0]
    batch = collate_multimodal([sample])

    expected_dimensions = {
        "video": config["data"]["video_dim"],
        "audio": config["data"]["audio_dim"],
        "text": config["data"]["text_dim"],
    }
    actual_dimensions = {name: int(batch[name].shape[-1]) for name in expected_dimensions}
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            f"Prepared behavior dimensions {actual_dimensions} do not match config "
            f"{expected_dimensions}"
        )

    requested_device = str(config["training"]["device"])
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["experiment"]["seed"]))
    model = build_model(config).to(device).eval()
    tensor_batch = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    with torch.inference_mode():
        output = model(tensor_batch)

    tensor_outputs = {
        name: list(value.shape) for name, value in output.items() if isinstance(value, torch.Tensor)
    }
    for name, value in output.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Model output {name} contains a non-finite value")

    schema_path = manifest.parent.parent / "feature_schema.json"
    new_root = manifest.parent.parent
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    text_mask = batch["modality_mask"][..., 2].bool()
    audio_mask = batch["modality_mask"][..., 1].bool()
    text_norms = torch.linalg.vector_norm(batch["text"], dim=-1)
    report: dict[str, object] = {
        "status": "passed",
        "config": str(config_path.as_posix()),
        "dataset": config["data"]["dataset_name"],
        "manifest": str(manifest.as_posix()),
        "sample_id": sample["sample_id"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "behavior_dimensions": actual_dimensions,
        "behavior_backends": schema["behavior_features"],
        "available_steps_in_checked_window": {
            "video": int(batch["modality_mask"][..., 0].sum().item()),
            "audio": int(audio_mask.sum().item()),
            "text": int(text_mask.sum().item()),
        },
        "text_embedding_norm": {
            "minimum_on_available_steps": float(text_norms[text_mask].min().item()),
            "maximum_on_available_steps": float(text_norms[text_mask].max().item()),
        },
        "model_outputs": tensor_outputs,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "feature_schema_sha256": _sha256(schema_path),
        "minimal_migration_parity": _verify_minimal_migration(new_root, legacy_root),
    }
    if report["behavior_backends"]["text_backend"] != "macbert":
        raise ValueError("The selected dataset does not declare the MacBERT backend")
    if int(report["available_steps_in_checked_window"]["text"]) == 0:
        raise ValueError("The checked validation window contains no available text features")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the official behavior-feature migration")
    parser.add_argument("--config", default="configs/bci_subjects_official_text.yaml")
    parser.add_argument("--output", default="results/behavior_migration_verification.json")
    parser.add_argument("--legacy-root", default="data/processed/bci_subjects_ept_v1")
    args = parser.parse_args()
    report = verify(Path(args.config), Path(args.legacy_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
