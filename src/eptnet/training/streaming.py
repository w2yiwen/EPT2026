"""Joint AV/event/boundary/offset training with chronological TBPTT."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eptnet.config import save_resolved_config
from eptnet.data.streaming import StreamManifestDataset, normalization_statistics
from eptnet.models import build_model
from eptnet.provenance import portable_path
from .trainer import resolve_device, set_seed


class StreamEPTNetLoss(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.settings = config["loss"]

    def forward(self, outputs, batch):
        valid = batch["sequence_mask"].bool() & batch["target_mask"].bool()
        av_valid = valid & outputs["av_mask"].bool()
        positive = valid & batch["positive_mask"].bool()
        zero = outputs["class_logits"].sum() * 0.0
        def classification(logits, mask):
            if not bool(mask.any()):
                return zero
            return F.cross_entropy(logits[mask], batch["labels"][mask],
                label_smoothing=float(self.settings.get("label_smoothing", 0.0)))
        event = classification(outputs["class_logits"], valid)
        av = classification(outputs["av_class_logits"], av_valid)
        boundary = (F.binary_cross_entropy_with_logits(outputs["boundary_logits"][av_valid],
                    batch["boundaries"][av_valid]) if bool(av_valid.any()) else zero)
        offset = (F.smooth_l1_loss(outputs["offsets"][positive], batch["offsets"][positive])
                  if bool(positive.any()) else zero)
        total = (float(self.settings.get("av_weight", 1)) * av
                 + float(self.settings.get("cls_weight", 1)) * event
                 + float(self.settings.get("boundary_weight", 1)) * boundary
                 + float(self.settings.get("offset_weight", 0.5)) * offset)
        return {"total": total, "audiovisual": av.detach(), "classification": event.detach(),
                "boundary": boundary.detach(), "offset": offset.detach()}


def stream_datasets(config):
    datasets = {split: StreamManifestDataset(config["data"][f"{split}_manifest"])
                for split in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = datasets[left].subjects & datasets[right].subjects
        if overlap:
            raise ValueError(f"Participants shared by {left} and {right}: {sorted(overlap)}")
    return datasets


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_provenance(config, datasets):
    root = Path(__file__).resolve().parents[3]
    sources = {path.relative_to(root).as_posix(): _sha256(path)
               for path in sorted((root / "src").rglob("*.py"))}
    for name in ("pyproject.toml", "requirements-lock.txt", "requirements-paper.txt"):
        sources[name] = _sha256(root / name)
    records = {}
    for split, dataset in datasets.items():
        records[split] = {"manifest_sha256": _sha256(dataset.path), "subjects": sorted(dataset.subjects),
            "sessions": [{"sample_id": row.get("sample_id"), "subject_id": str(row["subject_id"]),
                "sha256": _sha256(Path(row["path"]) if Path(row["path"]).is_absolute()
                                  else dataset.path.parent / row["path"])} for row in dataset.records]}
    return {"source_sha256": sources, "datasets": records, "encoders": config["encoders"],
            "protocol": "0.5s native-time read-update-write; participant-disjoint chronological TBPTT"}


def _phase(model, dataset, criterion, config, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals, count = {}, 0
    chunk = int(config["training"]["chunk_steps"])
    for session_index in range(len(dataset)):
        session = dataset[session_index]
        runtime = model.initial_state()
        for start in range(0, len(session), chunk):
            stop = min(start + chunk, len(session))
            targets = session.targets(start, stop, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                outputs = model(session, runtime, start, stop)
                losses = criterion(outputs, targets)
                if not bool(torch.isfinite(losses["total"])):
                    raise FloatingPointError(f"Nonfinite streaming loss in {session.sample_id}, chunk {start}:{stop}")
                supervised = bool((targets["sequence_mask"] & targets["target_mask"]).any())
                if training and supervised:
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]),
                                             error_if_nonfinite=True)
                    optimizer.step()
            runtime.detach()
            if supervised:
                count += 1
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
    if not count:
        raise ValueError("The manifest contains no annotated decision steps")
    return {name: value / count for name, value in totals.items()}


def _save_checkpoint(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config, resume=None):
    settings = config["training"]
    seed = int(config["experiment"]["seed"])
    set_seed(seed, settings)
    device = resolve_device(settings["device"])
    datasets = stream_datasets(config)
    output = Path(config["experiment"]["output_dir"]) / f"seed_{seed}"
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run already exists: {output}; choose another output_dir or resume it")
    output.mkdir(parents=True, exist_ok=True)
    provenance = stream_provenance(config, datasets)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    criterion = StreamEPTNetLoss(config)
    first, best, stale, history = 1, float("inf"), 0, []
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        # A resume restores the full run, rather than resetting optimizer or RNG.
        for section in ("data", "model", "loss", "encoders", "experiment"):
            if saved["config"][section] != config[section]:
                raise ValueError(f"Resume {section} does not match the saved run")
        old_training = {key: value for key, value in saved["config"]["training"].items()
                        if key not in ("epochs", "device")}
        new_training = {key: value for key, value in settings.items() if key not in ("epochs", "device")}
        if old_training != new_training or saved["provenance"] != provenance:
            raise ValueError("Resume requires the same training protocol, source and prepared recordings")
        if not (output / "best.pt").is_file():
            raise FileNotFoundError(f"The run's best.pt is required in {output}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        statistics = saved["normalization"]
        first, best, stale = int(saved["epoch"]) + 1, saved["best_val_loss"], saved["stale_epochs"]
        history = saved["history"]
        rng = saved["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["cpu"].cpu())
        if rng["cuda"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    else:
        stats_path = Path(config["data"]["normalization_stats"])
        statistics = (json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists()
                      else normalization_statistics(datasets["train"]))
        if set(statistics["subjects"]) != datasets["train"].subjects:
            raise ValueError("Normalization must be fitted only to the current training participants")
        if statistics.get("manifest_sha256") != _sha256(datasets["train"].path):
            raise ValueError("Normalization belongs to another training manifest")
        if statistics.get("recording_sha256") != [row["sha256"] for row in provenance["datasets"]["train"]["sessions"]]:
            raise ValueError("Normalization must match the prepared training recordings")
        model.set_normalization(statistics)
    save_resolved_config(config, str(output / "resolved_config.yaml"))
    (output / "normalization_stats.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for epoch in range(first, int(settings["epochs"]) + 1):
        if stale >= int(settings["patience"]):
            break
        train_loss = _phase(model, datasets["train"], criterion, config, device, optimizer)
        val_loss = _phase(model, datasets["val"], criterion, config, device)
        improved = val_loss["total"] < best
        best = min(best, val_loss["total"])
        stale = 0 if improved else stale + 1
        record = {"epoch": epoch, "train": train_loss, "val": val_loss, "best_val_loss": best,
                  "stale_epochs": stale}
        history.append(record)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "config": copy.deepcopy(config), "normalization": statistics, "provenance": provenance,
            "history": history, "best_val_loss": best, "stale_epochs": stale,
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                          "cpu": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        if improved:
            _save_checkpoint(payload, output / "best.pt")
        _save_checkpoint(payload, output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(record, allow_nan=False), flush=True)
    return {"output_dir": portable_path(output), "best_val_loss": best, "epochs_recorded": len(history)}
