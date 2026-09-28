from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from .provenance import portable_path


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _atomic_write_json(payload: Any, path: Path) -> None:
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


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class Trainer:
    CHECKPOINT_SCHEMA_VERSION = 3

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: Iterable,
        val_loader: Iterable,
        config: Mapping,
        device: torch.device,
        provenance: Mapping[str, Any],
        show_progress: bool | None = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        provenance_identity = provenance.get("provenance_sha256")
        if not isinstance(provenance_identity, str) or len(provenance_identity) != 64:
            raise ValueError("Trainer requires a valid provenance_sha256 record")
        self.provenance = copy.deepcopy(dict(provenance))
        self.show_progress = (
            bool(sys.stderr.isatty()) if show_progress is None else bool(show_progress)
        )
        training = config["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        self.grad_clip_norm = float(training.get("grad_clip_norm", 1.0))
        if self.grad_clip_norm <= 0:
            raise ValueError("training.grad_clip_norm must be positive")
        self.amp_enabled = bool(training.get("amp", False))
        if self.amp_enabled and device.type != "cuda":
            raise ValueError("training.amp=true is currently supported only on CUDA")
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_enabled else None
        self.scheduler = self._build_scheduler(training.get("scheduler", {}))
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_scheduler(self, scheduler_config: Mapping[str, Any]):
        name = str(scheduler_config.get("name", "reduce_on_plateau")).lower()
        if name in {"none", "off", "disabled"}:
            return None
        if name != "reduce_on_plateau":
            raise ValueError(f"Unsupported scheduler: {name}")
        mode = str(scheduler_config.get("mode", "min")).lower()
        if mode != "min":
            raise ValueError(
                "ReduceLROnPlateau mode must be 'min' because validation loss is monitored"
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=mode,
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 3)),
            threshold=float(scheduler_config.get("threshold", 1e-4)),
            threshold_mode=str(scheduler_config.get("threshold_mode", "rel")),
            cooldown=int(scheduler_config.get("cooldown", 0)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
            eps=float(scheduler_config.get("eps", 1e-8)),
        )

    @staticmethod
    def _ensure_finite_losses(losses: Mapping[str, Tensor], *, training: bool) -> None:
        if "total" not in losses:
            raise KeyError("Criterion output must contain a 'total' loss")
        invalid = [
            name
            for name, value in losses.items()
            if not isinstance(value, Tensor)
            or not bool(torch.isfinite(value.detach()).all().item())
        ]
        if invalid:
            phase = "training" if training else "validation"
            raise FloatingPointError(f"Non-finite {phase} loss detected: {invalid}")

    def _ensure_finite_gradients(self) -> None:
        invalid = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad.detach()).all().item())
        ]
        if invalid:
            preview = ", ".join(invalid[:8])
            suffix = " ..." if len(invalid) > 8 else ""
            raise FloatingPointError(f"Non-finite gradients detected in: {preview}{suffix}")

    def _epoch(
        self,
        loader: Iterable,
        training: bool,
        *,
        epoch: int | None = None,
        maximum_epochs: int | None = None,
    ) -> dict[str, float]:
        phase_start = time.perf_counter()
        self.model.train(training)
        batch_accumulated: dict[str, float] = {}
        step_accumulated: dict[str, float] = {}
        batches = 0
        total_valid_steps = 0
        phase = "train" if training else "valid"
        if epoch is not None and maximum_epochs is not None:
            description = f"Epoch {epoch:03d}/{maximum_epochs:03d} | {phase:<5}"
        else:
            description = phase
        try:
            total = len(loader)  # type: ignore[arg-type]
        except TypeError:
            total = None
        progress = tqdm(
            loader,
            total=total,
            desc=description,
            unit="batch",
            position=1 if epoch is not None else 0,
            leave=False,
            dynamic_ncols=True,
            disable=not getattr(self, "show_progress", False),
        )
        with progress:
            for batch_index, raw_batch in enumerate(progress):
                batch = move_to_device(raw_batch, self.device)
                if "sequence_mask" not in batch:
                    raise KeyError("Batch is missing sequence_mask")
                decision_mask = batch["sequence_mask"].bool()
                if "target_mask" in batch:
                    decision_mask = decision_mask & batch["target_mask"].bool()
                valid_steps = int(decision_mask.sum().item())
                if valid_steps <= 0:
                    raise RuntimeError(
                        f"Batch {batch_index} contains no target-valid decision steps"
                    )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.amp_enabled
                    else nullcontext()
                )
                with torch.set_grad_enabled(training):
                    with autocast_context:
                        outputs = self.model(batch)
                        losses = self.criterion(outputs, batch)
                    self._ensure_finite_losses(losses, training=training)
                    if training:
                        if self.scaler is None:
                            losses["total"].backward()
                        else:
                            self.scaler.scale(losses["total"]).backward()
                            self.scaler.unscale_(self.optimizer)
                        self._ensure_finite_gradients()
                        gradient_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                        if not bool(torch.isfinite(gradient_norm).all().item()):
                            raise FloatingPointError("Non-finite gradient norm detected")
                        if self.scaler is None:
                            self.optimizer.step()
                        else:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                for name, value in losses.items():
                    scalar = float(value.detach().item())
                    batch_accumulated[name] = batch_accumulated.get(name, 0.0) + scalar
                    step_accumulated[name] = step_accumulated.get(name, 0.0) + scalar * valid_steps
                total_valid_steps += valid_steps
                batches += 1
                postfix: dict[str, str] = {"loss": f"{batch_accumulated['total'] / batches:.4f}"}
                if training:
                    postfix["lr"] = f"{float(self.optimizer.param_groups[0]['lr']):.2e}"
                progress.set_postfix(postfix, refresh=False)
        if batches == 0 or total_valid_steps == 0:
            raise RuntimeError("DataLoader produced zero valid batches")
        # The primary loss is an unweighted mean over optimization batches.  A
        # training batch may be a complete session or a declared causal window;
        # validation remains one complete subject/session per batch.
        averaged = {name: value / batches for name, value in batch_accumulated.items()}
        averaged.update(
            {
                f"target_step_weighted_{name}": value / total_valid_steps
                for name, value in step_accumulated.items()
            }
        )
        averaged["num_valid_steps"] = float(total_valid_steps)
        averaged["num_batches"] = float(batches)
        averaged["elapsed_seconds"] = time.perf_counter() - phase_start
        return averaged

    def train_one_epoch(
        self, *, epoch: int | None = None, maximum_epochs: int | None = None
    ) -> dict[str, float]:
        return self._epoch(
            self.train_loader,
            training=True,
            epoch=epoch,
            maximum_epochs=maximum_epochs,
        )

    @torch.no_grad()
    def validate(
        self, *, epoch: int | None = None, maximum_epochs: int | None = None
    ) -> dict[str, float]:
        return self._epoch(
            self.val_loader,
            training=False,
            epoch=epoch,
            maximum_epochs=maximum_epochs,
        )

    def _capture_rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_loaders": {},
        }
        for name, loader in (("train", self.train_loader), ("val", self.val_loader)):
            generator = getattr(loader, "generator", None)
            if generator is not None:
                state["data_loaders"][name] = generator.get_state()
        return state

    def _restore_rng_state(self, state: Mapping[str, Any]) -> None:
        required = {"python", "numpy", "torch_cpu", "data_loaders"}
        missing = sorted(required.difference(state))
        if missing:
            raise KeyError(f"Checkpoint RNG state is missing keys: {missing}")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
            if len(cuda_state) != torch.cuda.device_count():
                raise RuntimeError(
                    "CUDA device count differs from the checkpoint; exact RNG restoration is impossible"
                )
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])
        loader_states = state["data_loaders"]
        for name, loader in (("train", self.train_loader), ("val", self.val_loader)):
            generator = getattr(loader, "generator", None)
            if generator is not None:
                if name not in loader_states:
                    raise KeyError(f"Checkpoint is missing the {name} DataLoader generator state")
                generator.set_state(loader_states[name].cpu())

    def _validate_resume_config(self, saved_config: Mapping[str, Any]) -> None:
        for section in ("data", "model", "loss"):
            if saved_config.get(section) != self.config.get(section):
                raise ValueError(f"Resume rejected because the '{section}' configuration changed")
        # A last checkpoint only stores the current model state, not a separate
        # copy of the earlier best model.  Resuming into another directory could
        # therefore finish without a valid ``best.pt`` in the destination.
        for key in ("name", "seed", "output_dir"):
            if saved_config.get("experiment", {}).get(key) != self.config.get("experiment", {}).get(
                key
            ):
                raise ValueError(f"Resume rejected because experiment.{key} changed")
        exact_training_keys = (
            "sequence_protocol",
            "window_size",
            "window_stride",
            "window_warmup_steps",
            "positive_window_oversample",
            "batch_size",
            "patience",
            "learning_rate",
            "weight_decay",
            "grad_clip_norm",
            "num_workers",
            "amp",
            "deterministic",
            "deterministic_warn_only",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "allow_tf32",
            "scheduler",
        )
        saved_training = saved_config.get("training", {})
        current_training = self.config.get("training", {})
        changed = [
            key
            for key in exact_training_keys
            if saved_training.get(key) != current_training.get(key)
        ]
        if changed:
            raise ValueError(f"Resume rejected because training settings changed: {changed}")

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        best_val_loss: float,
        stale_epochs: int,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": epoch,
            "config": copy.deepcopy(dict(self.config)),
            "best_val_loss": best_val_loss,
            "stale_epochs": stale_epochs,
            "rng_state": self._capture_rng_state(),
            "history": copy.deepcopy(history),
            "provenance": copy.deepcopy(self.provenance),
        }

    def _validate_resume_provenance(self, saved: Any) -> None:
        if not isinstance(saved, Mapping):
            raise ValueError("Resume rejected because provenance metadata is missing")
        saved_identity = saved.get("provenance_sha256")
        current_identity = self.provenance.get("provenance_sha256")
        if not isinstance(saved_identity, str) or saved_identity != current_identity:
            raise ValueError(
                "Resume rejected because executable source or prepared-data provenance changed"
            )

    def _load_checkpoint(self, path: str | Path) -> tuple[int, float, int, list[dict[str, Any]]]:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        required = {
            "schema_version",
            "model",
            "optimizer",
            "scheduler",
            "epoch",
            "config",
            "best_val_loss",
            "stale_epochs",
            "rng_state",
            "history",
            "provenance",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise KeyError(f"Resume checkpoint is missing keys: {missing}")
        if int(checkpoint["schema_version"]) != self.CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint schema {checkpoint['schema_version']}; "
                f"expected {self.CHECKPOINT_SCHEMA_VERSION}"
            )
        self._validate_resume_config(checkpoint["config"])
        self._validate_resume_provenance(checkpoint["provenance"])
        best_path = self.output_dir / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(
                f"Resume requires the run's existing best checkpoint at {best_path}; "
                "a last checkpoint alone cannot reconstruct an earlier best model."
            )
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is None:
            if checkpoint["scheduler"] is not None:
                raise ValueError(
                    "Checkpoint has scheduler state but the current scheduler is disabled"
                )
        else:
            if checkpoint["scheduler"] is None:
                raise ValueError("Checkpoint is missing scheduler state")
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler is None:
            if checkpoint.get("scaler") is not None:
                raise ValueError("Checkpoint has AMP scaler state but AMP is disabled")
        else:
            if checkpoint.get("scaler") is None:
                raise ValueError("Checkpoint is missing AMP scaler state")
            self.scaler.load_state_dict(checkpoint["scaler"])
        self._restore_rng_state(checkpoint["rng_state"])
        epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint["best_val_loss"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        if history and int(history[-1]["epoch"]) != epoch:
            raise ValueError("Checkpoint history does not end at the checkpoint epoch")
        print(
            json.dumps(
                {
                    "event": "resume",
                    "checkpoint": portable_path(checkpoint_path),
                    "completed_epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "stale_epochs": stale_epochs,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return epoch + 1, best_val_loss, stale_epochs, history

    def fit(self, resume_from: str | Path | None = None) -> dict[str, Any]:
        training = self.config["training"]
        maximum_epochs = int(training["epochs"])
        patience = int(training["patience"])
        if maximum_epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if patience <= 0:
            raise ValueError("training.patience must be positive")

        start_epoch = 1
        best_loss = float("inf")
        stale_epochs = 0
        history: list[dict[str, Any]] = []
        if resume_from is not None:
            start_epoch, best_loss, stale_epochs, history = self._load_checkpoint(resume_from)
        epochs_this_run = 0
        stopped_early = stale_epochs >= patience

        epoch_progress = tqdm(
            range(start_epoch, maximum_epochs + 1),
            total=maximum_epochs,
            initial=start_epoch - 1,
            desc="Epochs",
            unit="epoch",
            position=0,
            leave=True,
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        with epoch_progress:
            for epoch in epoch_progress:
                if stale_epochs >= patience:
                    stopped_early = True
                    break
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                epoch_start = time.perf_counter()
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                train_loss = self.train_one_epoch(epoch=epoch, maximum_epochs=maximum_epochs)
                val_loss = self.validate(epoch=epoch, maximum_epochs=maximum_epochs)
                monitored_loss = float(val_loss["total"])
                if not np.isfinite(monitored_loss):
                    raise FloatingPointError("Validation total loss is not finite")

                improved = monitored_loss < best_loss
                if improved:
                    best_loss = monitored_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if self.scheduler is not None:
                    self.scheduler.step(monitored_loss)
                next_learning_rate = float(self.optimizer.param_groups[0]["lr"])
                elapsed_seconds = time.perf_counter() - epoch_start
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "train": train_loss,
                    "val": val_loss,
                    "learning_rate": learning_rate,
                    "next_learning_rate": next_learning_rate,
                    "best_val_loss": best_loss,
                    "stale_epochs": stale_epochs,
                    "improved": improved,
                    "elapsed_seconds": elapsed_seconds,
                }
                if self.device.type == "cuda":
                    record["max_cuda_memory_bytes"] = int(
                        torch.cuda.max_memory_allocated(self.device)
                    )
                history.append(record)
                epochs_this_run += 1

                payload = self._checkpoint_payload(
                    epoch=epoch,
                    best_val_loss=best_loss,
                    stale_epochs=stale_epochs,
                    history=history,
                )
                if improved:
                    _atomic_torch_save(payload, self.output_dir / "best.pt")
                _atomic_torch_save(payload, self.output_dir / "last.pt")
                _atomic_write_json(history, self.output_dir / "history.json")
                epoch_progress.set_postfix(
                    {
                        "train": f"{float(train_loss['total']):.4f}",
                        "valid": f"{monitored_loss:.4f}",
                        "best": f"{best_loss:.4f}",
                        "lr": f"{next_learning_rate:.2e}",
                        "stale": f"{stale_epochs}/{patience}",
                    },
                    refresh=True,
                )
                if self.show_progress:
                    tqdm.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False),
                        file=sys.stdout,
                    )
                else:
                    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

                if stale_epochs >= patience:
                    stopped_early = True
                    break

        last_epoch = int(history[-1]["epoch"]) if history else 0
        return {
            "best_val_loss": best_loss,
            "last_epoch": last_epoch,
            "epochs_this_run": epochs_this_run,
            "epochs_recorded": len(history),
            "stopped_early": stopped_early,
            "output_dir": portable_path(self.output_dir),
        }
