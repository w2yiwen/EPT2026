from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import load_config, save_resolved_config
from .data import StitchedManifestDataset, collate_multimodal
from .engine import Trainer
from .losses import EPTNetLoss
from .models import build_model
from .provenance import (
    build_provenance_fingerprint,
    portable_path,
    portable_python_command,
)
from .training_artifacts import (
    estimate_full_training_runtime,
    generate_training_artifacts,
    load_training_history,
)

CONTINUOUS_SESSION_PROTOCOL = "continuous_session"


def validate_training_cohort(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail before training unless manifests match the declared aligned cohort."""
    data = config.get("data", {})
    expected = data.get("expected_session_ids")
    if expected is None:
        return {"status": "not_declared"}
    expected_ids = {str(value) for value in expected}
    excluded_ids = {str(value) for value in data.get("excluded_session_ids", [])}
    split_ids: dict[str, list[str]] = {}
    all_ids: list[str] = []
    manifest_directory: Path | None = None
    for split in ("train", "val", "test"):
        path = Path(str(data[f"{split}_manifest"])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cohort manifest is missing: {path}")
        if manifest_directory is None:
            manifest_directory = path.parent
        elif path.parent != manifest_directory:
            raise ValueError("Cohort manifests do not share one directory")
        ids: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                metadata = record.get("metadata", {})
                session_id = str(metadata.get("session_id", record.get("sample_id", "")))
                if not session_id:
                    raise ValueError(f"Missing session ID in {path}:{line_number}")
                ids.append(session_id)
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sessions in {split} manifest: {ids}")
        split_ids[split] = ids
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"A session occurs in multiple splits: {all_ids}")
    actual_ids = set(all_ids)
    if actual_ids != expected_ids:
        raise ValueError(
            "Prepared cohort does not match data.expected_session_ids; "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}"
        )
    leaked = actual_ids.intersection(excluded_ids)
    if leaked:
        raise ValueError(f"Explicitly excluded sessions reached training: {sorted(leaked)}")

    assert manifest_directory is not None
    dataset_root = manifest_directory.parent
    summary_path = dataset_root / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Dataset summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset") != data.get("dataset_name"):
        raise ValueError("Dataset summary name does not match the training configuration")
    if int(summary.get("num_subjects", -1)) != len(expected_ids):
        raise ValueError("Dataset summary subject count does not match the declared cohort")

    behavior_steps: dict[str, dict[str, int]] = {}
    video_alignment_methods: dict[str, str] = {}
    if bool(data.get("require_aligned_behavior_modalities", False)):
        for session_id in sorted(actual_ids):
            session_manifest_path = (
                dataset_root / "sessions" / session_id / "session_manifest.json"
            )
            if not session_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Session integrity manifest is missing: {session_manifest_path}"
                )
            session_manifest = json.loads(
                session_manifest_path.read_text(encoding="utf-8")
            )
            availability = session_manifest.get("availability_steps", {})
            steps = {
                name: int(availability.get(name, 0))
                for name in ("video", "audio", "text")
            }
            missing_modalities = [name for name, count in steps.items() if count <= 0]
            if missing_modalities:
                raise ValueError(
                    f"{session_id} has no valid aligned steps for {missing_modalities}"
                )
            alignment = session_manifest.get("alignment", {})
            whisper_alignment = alignment.get("whisper_alignment")
            if not isinstance(whisper_alignment, dict):
                raise ValueError(f"{session_id} lacks verified Whisper alignment metadata")
            if whisper_alignment.get("status") != "complete":
                raise ValueError(f"{session_id} Whisper alignment is incomplete")
            video_method = str(alignment.get("video_feature_alignment_method", ""))
            allowed_video_methods = {
                "direct_real_time_frame_sampling_v2",
                "legacy_embedding_time_remap_v1",
            }
            if video_method not in allowed_video_methods:
                raise ValueError(
                    f"{session_id} has an unsupported video alignment method: {video_method!r}"
                )
            behavior_steps[session_id] = steps
            video_alignment_methods[session_id] = video_method
    return {
        "status": "passed",
        "session_count": len(actual_ids),
        "session_ids": sorted(actual_ids),
        "excluded_session_ids": sorted(excluded_ids),
        "split_session_ids": split_ids,
        "behavior_available_steps": behavior_steps,
        "video_alignment_methods": video_alignment_methods,
    }


def build_runtime_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint executable source and every prepared split tensor."""

    project_root = Path(__file__).resolve().parents[2]
    manifest_paths: list[Path] = []
    for key in ("train_manifest", "val_manifest", "test_manifest"):
        value = config.get("data", {}).get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"data.{key} must identify a prepared split manifest")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        manifest_paths.append(candidate.resolve())

    manifest_directories = {path.parent for path in manifest_paths}
    if len(manifest_directories) != 1:
        raise ValueError("All split manifests must share one manifests directory")
    manifest_directory = next(iter(manifest_directories))
    if manifest_directory.name != "manifests":
        raise ValueError("Prepared split manifests must be stored below a manifests directory")
    dataset_root = manifest_directory.parent
    return build_provenance_fingerprint(
        project_root,
        dataset_root,
        manifest_paths=manifest_paths,
        requirements_lock_path=project_root / "requirements-lock.txt",
        include_git=True,
    )


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({value}) but CUDA is not available")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is invalid; detected {torch.cuda.device_count()} device(s)"
            )
        return torch.device("cuda", index)
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS device requested but MPS is not available")
    return device


def set_seed(seed: int, settings: Mapping[str, Any]) -> None:
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    deterministic = bool(settings.get("deterministic", True))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    warn_only = bool(settings.get("deterministic_warn_only", True))
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    if getattr(torch.backends, "cudnn", None) is not None:
        torch.backends.cudnn.deterministic = bool(
            settings.get("cudnn_deterministic", deterministic)
        )
        torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", False))
        torch.backends.cudnn.allow_tf32 = bool(settings.get("allow_tf32", False))
    if getattr(torch.backends, "cuda", None) is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", False))


def seed_worker(_worker_id: int) -> None:
    """Seed NumPy and Python in each DataLoader worker from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _seed_output_dir(base_output_dir: str | Path, seed: int) -> Path:
    base = Path(base_output_dir)
    seed_name = f"seed_{seed}"
    return base if base.name == seed_name else base / seed_name


def _reject_accidental_overwrite(output_dir: Path, resume_from: str | None) -> None:
    """Protect completed or partial run evidence from silent replacement."""
    if resume_from is not None:
        return
    artifact_names = (
        "best.pt",
        "last.pt",
        "history.json",
        "resolved_config.yaml",
        "run_metadata.json",
    )
    existing = [name for name in artifact_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing run artifacts in {output_dir}: {existing}. "
            "Use --resume with that run's trusted last.pt or choose a new output directory/seed."
        )


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested_backend": device.type,
        "resolved_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "cuda_compute_capability": f"{properties.major}.{properties.minor}",
                "cuda_total_memory_bytes": properties.total_memory,
            }
        )
    return metadata


def _build_run_metadata(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    device: torch.device,
    train_sessions: int,
    val_sessions: int,
    train_unique_steps: int,
    train_batches_per_epoch: int,
    val_batches_per_epoch: int,
    model: torch.nn.Module,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": portable_python_command(sys.executable, "eptnet.train", sys.argv[1:]),
        "working_directory": portable_path(Path.cwd()),
        "config_path": portable_path(args.config),
        "resume_from": portable_path(args.resume) if args.resume else None,
        "seed": int(config["experiment"]["seed"]),
        "output_dir": portable_path(config["experiment"]["output_dir"]),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": _device_metadata(device),
        "determinism": {
            "deterministic_algorithms": bool(training.get("deterministic", True)),
            "warn_only": bool(training.get("deterministic_warn_only", True)),
            "cudnn_deterministic": bool(training.get("cudnn_deterministic", True)),
            "cudnn_benchmark": bool(training.get("cudnn_benchmark", False)),
            "allow_tf32": bool(training.get("allow_tf32", False)),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "amp_enabled": bool(training.get("amp", False)),
        "sequence_protocol": {
            "name": str(training["sequence_protocol"]),
            "dataset_adapter": "StitchedManifestDataset",
            "unit": "complete_session",
            "sampler": "SequentialSampler",
            "shuffle": False,
            "drop_last": False,
            "session_batch_size": int(training["batch_size"]),
            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
            "train_unique_steps": train_unique_steps,
            "train_batches_per_epoch": train_batches_per_epoch,
            "val_batches_per_epoch": val_batches_per_epoch,
            "row_accounting": "one unique (session_id, row_index) per epoch",
            "epoch_loss_aggregation": "unweighted mean over complete-session batches",
            "checkpoint_selection": "minimum participant-mean validation total loss",
            "target_step_weighted_losses": "reported as diagnostics only",
        },
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "provenance": dict(provenance),
    }


def _build_continuous_session_loader(
    manifest_path: str,
    *,
    settings: Mapping[str, Any],
    device: torch.device,
    seed: int,
    maximum_sessions: int | None = None,
) -> tuple[Dataset, DataLoader]:
    """Build an ordered loader whose every item is one complete session timeline.

    A batch size of one is an explicit protocol constraint rather than an incidental
    consequence of the current single-session dataset.  It prevents future
    multi-session datasets from silently introducing padding-dependent batching or
    optimizer-step changes into the trusted protocol.
    """
    protocol = str(settings.get("sequence_protocol", ""))
    if protocol != CONTINUOUS_SESSION_PROTOCOL:
        raise ValueError(
            "training.sequence_protocol must be 'continuous_session'; "
            "windowed training is incompatible with the trusted evaluation timeline"
        )
    batch_size = int(settings["batch_size"])
    if batch_size != 1:
        raise ValueError(
            "training.batch_size must be 1 for sequence_protocol='continuous_session' "
            "so each optimization batch is one complete, unpadded session"
        )

    complete_dataset = StitchedManifestDataset(manifest_path, require_targets=True)
    if maximum_sessions is None:
        complete_dataset.materialize()
        dataset: Dataset = complete_dataset
    else:
        if maximum_sessions <= 0:
            raise ValueError("maximum_sessions must be positive")
        selected_indices = list(range(min(maximum_sessions, len(complete_dataset))))
        for index in selected_indices:
            complete_dataset[index]
        dataset = Subset(complete_dataset, selected_indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        generator=generator,
        num_workers=int(settings["num_workers"]),
        collate_fn=collate_multimodal,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
    )
    return dataset, loader


def _resolve_automatic_loss_weights(
    config: dict[str, Any], dataset: Dataset
) -> dict[str, Any]:
    """Resolve imbalance weights from unique training rows, never overlapping windows."""
    labels = []
    boundaries = []
    for index in range(len(dataset)):
        sample = dataset[index]
        target_mask = sample.get("target_mask")
        if target_mask is None:
            target_mask = torch.ones_like(sample["labels"], dtype=torch.bool)
        else:
            target_mask = target_mask.bool()
        if not bool(target_mask.any().item()):
            raise ValueError(f"Training session {index} contains no target-valid steps")
        labels.append(sample["labels"].long()[target_mask])
        boundaries.append(sample["boundaries"].float()[target_mask])
    all_labels = torch.cat(labels)
    all_boundaries = torch.cat(boundaries)
    num_classes = int(config["data"]["num_classes"])
    counts = torch.bincount(all_labels, minlength=num_classes).to(torch.float64)
    if counts.numel() != num_classes or bool((counts <= 0).any()):
        raise ValueError(f"Every class must occur in training data; counts={counts.tolist()}")
    positive_counts = all_boundaries.sum(dim=0, dtype=torch.float64)
    negative_counts = all_boundaries.shape[0] - positive_counts
    if bool((positive_counts <= 0).any()):
        raise ValueError(
            "Every boundary head needs positive training targets; "
            f"counts={positive_counts.tolist()}"
        )

    loss = config.setdefault("loss", {})
    if loss.get("class_weights") == "auto":
        weights = all_labels.numel() / (num_classes * counts)
        loss["class_weights"] = [float(value) for value in weights]
    if loss.get("boundary_pos_weight") == "auto":
        weights = negative_counts / positive_counts
        loss["boundary_pos_weight"] = [float(value) for value in weights]
    return {
        "unique_training_steps": int(all_labels.numel()),
        "class_counts": [int(value) for value in counts],
        "class_weights": loss.get("class_weights"),
        "boundary_positive_counts": [int(value) for value in positive_counts],
        "boundary_pos_weight": loss.get("boundary_pos_weight"),
    }


def _dataset_step_count(dataset: Dataset) -> int:
    total = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        labels = sample.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            raise ValueError(f"Dataset item {index} has no rank-1 labels tensor")
        total += int(labels.shape[0])
    if total <= 0:
        raise ValueError("Dataset contains no temporal steps")
    return total


def _manifest_step_count(path: str | Path) -> int:
    total = 0
    sessions: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            session_id = str(metadata.get("session_id", "")).strip()
            if not session_id:
                raise ValueError(f"Manifest line {line_number} has no metadata.session_id")
            if session_id in sessions:
                raise ValueError(
                    "Runtime estimation requires one complete-session record per manifest line"
                )
            sessions.add(session_id)
            steps = metadata.get("num_steps")
            if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
                raise ValueError(f"Manifest line {line_number} has invalid metadata.num_steps")
            total += steps
    if total <= 0:
        raise ValueError(f"Manifest contains no sessions: {path}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EPT-Net")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, help="Override experiment.seed")
    parser.add_argument("--output-dir", help="Override the base experiment output directory")
    parser.add_argument("--device", help="Override training.device (for example cuda:0 or cpu)")
    parser.add_argument("--resume", help="Resume from a trusted last.pt checkpoint")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show nested epoch/train/validation progress bars (default: auto-detect terminal)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run one epoch on one complete real train session and one complete real "
            "validation session; never reads the test split or reports performance"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = config["experiment"]
    settings = config["training"]
    configured_maximum_epochs = int(settings["epochs"])
    configured_patience = int(settings["patience"])
    if args.smoke:
        if args.resume:
            raise ValueError("--smoke cannot be combined with --resume")
        experiment["name"] = f"{experiment['name']}_smoke"
        settings["epochs"] = 1
        settings["patience"] = 1
    if args.seed is not None:
        experiment["seed"] = args.seed
    if args.device is not None:
        settings["device"] = args.device
    seed = int(experiment["seed"])
    if args.output_dir:
        base_output_dir = args.output_dir
    elif args.smoke:
        base_output_dir = Path("results") / "smoke" / str(experiment["name"])
    else:
        base_output_dir = experiment["output_dir"]
    output_dir = _seed_output_dir(base_output_dir, seed)
    experiment["output_dir"] = str(output_dir)
    _reject_accidental_overwrite(output_dir, args.resume)

    cohort_contract = validate_training_cohort(config)
    print(
        json.dumps({"event": "cohort_contract", **cohort_contract}, ensure_ascii=False),
        flush=True,
    )
    provenance = build_runtime_provenance(config)
    set_seed(seed, settings)
    device = resolve_device(str(settings["device"]))
    show_progress = bool(sys.stderr.isatty()) if args.progress is None else bool(args.progress)
    runtime = _device_metadata(device)
    print(
        json.dumps(
            {
                "event": "runtime",
                "seed": seed,
                "device": runtime,
                "output_dir": portable_path(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    train_manifest = config["data"].get("train_manifest")
    val_manifest = config["data"].get("val_manifest")
    if not train_manifest or not val_manifest:
        raise ValueError(
            "train_manifest and val_manifest must point to prepared real-data manifests"
        )
    train_dataset, train_loader = _build_continuous_session_loader(
        train_manifest,
        settings=settings,
        device=device,
        seed=seed,
        maximum_sessions=1 if args.smoke else None,
    )
    val_dataset, val_loader = _build_continuous_session_loader(
        val_manifest,
        settings=settings,
        device=device,
        seed=(seed + 1) % 2**32,
        maximum_sessions=1 if args.smoke else None,
    )
    target_statistics = _resolve_automatic_loss_weights(config, train_dataset)
    post_materialization_provenance = build_runtime_provenance(config)
    if post_materialization_provenance["provenance_sha256"] != provenance["provenance_sha256"]:
        raise RuntimeError(
            "Executable source or prepared data changed while materializing the training run"
        )

    model = build_model(config)
    criterion = EPTNetLoss(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, str(output_dir / "resolved_config.yaml"))
    metadata = _build_run_metadata(
        args=args,
        config=config,
        device=device,
        train_sessions=len(train_dataset),
        val_sessions=len(val_dataset),
        train_unique_steps=int(target_statistics["unique_training_steps"]),
        train_batches_per_epoch=len(train_loader),
        val_batches_per_epoch=len(val_loader),
        model=model,
        provenance=provenance,
    )
    metadata["cohort_contract"] = cohort_contract
    metadata["target_statistics"] = target_statistics
    metadata["progress"] = {
        "enabled": show_progress,
        "requested": args.progress,
        "mode": "explicit" if args.progress is not None else "terminal_auto_detection",
    }
    metadata["run_kind"] = "real_session_smoke" if args.smoke else "formal_training"
    if args.smoke:
        metadata["smoke_contract"] = {
            "epochs": 1,
            "train_sessions": 1,
            "validation_sessions": 1,
            "complete_sessions_not_truncated": True,
            "test_split_accessed": False,
            "full_experiment_started": False,
            "scientific_scope": "engineering execution and timing only",
        }
    metadata_path = output_dir / "run_metadata.json"
    _atomic_write_json(metadata, metadata_path)

    try:
        trainer = Trainer(
            model,
            criterion,
            train_loader,
            val_loader,
            config,
            device,
            provenance=provenance,
            show_progress=show_progress,
        )
        result = trainer.fit(resume_from=args.resume)
        training_artifacts = generate_training_artifacts(output_dir)
        result["training_artifacts"] = training_artifacts
        if args.smoke:
            history = load_training_history(output_dir / "history.json")
            runtime_estimate = estimate_full_training_runtime(
                history,
                observed_train_steps=_dataset_step_count(train_dataset),
                observed_val_steps=_dataset_step_count(val_dataset),
                full_train_steps=_manifest_step_count(train_manifest),
                full_val_steps=_manifest_step_count(val_manifest),
                maximum_epochs=configured_maximum_epochs,
                patience=configured_patience,
            )
            runtime_estimate["device"] = _device_metadata(device)
            _atomic_write_json(runtime_estimate, output_dir / "runtime_estimate.json")
            result["runtime_estimate"] = runtime_estimate
    except BaseException as error:
        metadata.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _atomic_write_json(metadata, metadata_path)
        raise

    metadata.update(
        {
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
    )
    _atomic_write_json(metadata, metadata_path)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
