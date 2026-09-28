from __future__ import annotations

import itertools
import json
import random
from argparse import Namespace
from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import eptnet.engine as engine_module
from eptnet.config import load_config
from eptnet.engine import Trainer
from eptnet.train import _build_run_metadata, _reject_accidental_overwrite


class _ToySequenceDataset(Dataset):
    """Small deterministic dataset; stochasticity comes from shuffle and dropout."""

    def __init__(self) -> None:
        features = torch.linspace(-1.5, 1.5, steps=6 * 4 * 3, dtype=torch.float32)
        self.features = features.reshape(6, 4, 3)
        self.targets = (0.4 * self.features[..., :1]) - (0.2 * self.features[..., 1:2])

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "targets": self.targets[index],
            "sequence_mask": torch.ones(self.features.shape[1], dtype=torch.bool),
        }


class _ToyDropoutRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.35)
        self.projection = nn.Linear(3, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, object]:
        return {
            "prediction": self.projection(self.dropout(batch["features"])),
            "training": self.training,
        }


class _ToyCriterion(nn.Module):
    def forward(
        self, outputs: dict[str, object], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        prediction = outputs["prediction"]
        assert isinstance(prediction, torch.Tensor)
        if bool(outputs["training"]):
            squared_error = (prediction - batch["targets"]).square().squeeze(-1)
            mask = batch["sequence_mask"].to(dtype=squared_error.dtype)
            total = (squared_error * mask).sum() / mask.sum()
        else:
            # A fixed validation signal deliberately exercises scheduler and
            # stale-epoch restoration while retaining a valid autograd graph.
            total = prediction.sum() * 0.0 + 1.0
        return {"total": total}


class _ReportedLossModel(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"anchor": batch["reported_loss"] * 0.0}


class _ReportedLossCriterion(nn.Module):
    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {"total": batch["reported_loss"] + outputs["anchor"]}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_toy_trainer(
    output_dir,
    *,
    epochs: int,
    seed: int = 31415,
    show_progress: bool = False,
) -> Trainer:
    _seed_everything(seed)
    dataset = _ToySequenceDataset()
    train_generator = torch.Generator().manual_seed(seed + 1)
    val_generator = torch.Generator().manual_seed(seed + 2)
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        generator=val_generator,
    )
    config = {
        "experiment": {
            "name": "resume_equivalence_toy",
            "seed": seed,
            "output_dir": str(output_dir),
        },
        "data": {"name": "deterministic_toy"},
        "model": {"name": "dropout_linear"},
        "loss": {"name": "masked_mse"},
        "training": {
            "sequence_protocol": "continuous_session",
            "epochs": epochs,
            "patience": 20,
            "batch_size": 2,
            "learning_rate": 0.05,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0,
            "num_workers": 0,
            "amp": False,
            "deterministic": True,
            "deterministic_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "allow_tf32": False,
            "scheduler": {
                "name": "reduce_on_plateau",
                "mode": "min",
                "factor": 0.5,
                "patience": 0,
                "threshold": 0.0,
                "threshold_mode": "abs",
                "cooldown": 0,
                "min_lr": 1e-5,
                "eps": 1e-12,
            },
        },
    }
    return Trainer(
        model=_ToyDropoutRegressor(),
        criterion=_ToyCriterion(),
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=torch.device("cpu"),
        provenance={"provenance_sha256": "a" * 64},
        show_progress=show_progress,
    )


def _assert_nested_equal(actual, expected) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(actual, np.ndarray):
        assert isinstance(expected, np.ndarray)
        assert np.array_equal(actual, expected)
    elif isinstance(actual, Mapping):
        assert isinstance(expected, Mapping)
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)):
        assert isinstance(expected, Sequence)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_resume_rejects_output_directory_change():
    saved = load_config("configs/default.yaml")
    current = deepcopy(saved)
    current["experiment"]["output_dir"] = "results/another_run/seed_42"
    trainer = object.__new__(Trainer)
    trainer.config = current

    with pytest.raises(ValueError, match=r"experiment\.output_dir changed"):
        trainer._validate_resume_config(saved)


def test_validation_loss_is_session_mean_with_step_weighted_diagnostic():
    trainer = object.__new__(Trainer)
    trainer.model = _ReportedLossModel()
    trainer.criterion = _ReportedLossCriterion()
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    trainer.val_loader = [
        {
            "sequence_mask": torch.tensor([[True]]),
            "reported_loss": torch.tensor(1.0),
        },
        {
            "sequence_mask": torch.tensor([[True, True, True]]),
            "reported_loss": torch.tensor(3.0),
        },
    ]

    metrics = trainer.validate()

    assert metrics["total"] == pytest.approx(2.0)
    assert metrics["target_step_weighted_total"] == pytest.approx(2.5)
    assert metrics["num_valid_steps"] == 4.0
    assert metrics["num_batches"] == 2.0


def test_progress_bars_render_and_do_not_change_training_state(tmp_path, capsys):
    quiet_dir = tmp_path / "quiet"
    progress_dir = tmp_path / "progress"

    quiet_trainer = _build_toy_trainer(quiet_dir, epochs=2, show_progress=False)
    quiet_trainer.fit()
    capsys.readouterr()

    progress_trainer = _build_toy_trainer(progress_dir, epochs=2, show_progress=True)
    progress_trainer.fit()
    captured = capsys.readouterr()

    assert "Epochs" in captured.err
    assert "train" in captured.err
    assert "valid" in captured.err

    quiet_checkpoint = torch.load(quiet_dir / "last.pt", map_location="cpu", weights_only=False)
    progress_checkpoint = torch.load(
        progress_dir / "last.pt", map_location="cpu", weights_only=False
    )
    for key in ("model", "optimizer", "scheduler", "rng_state"):
        _assert_nested_equal(progress_checkpoint[key], quiet_checkpoint[key])


def test_resume_rejects_sequence_protocol_change():
    saved = load_config("configs/default.yaml")
    current = deepcopy(saved)
    current["training"]["sequence_protocol"] = "overlap_windows"
    trainer = object.__new__(Trainer)
    trainer.config = current

    with pytest.raises(ValueError, match="sequence_protocol"):
        trainer._validate_resume_config(saved)


def test_resume_rejects_early_stopping_patience_change():
    saved = load_config("configs/default.yaml")
    current = deepcopy(saved)
    current["training"]["patience"] = int(saved["training"]["patience"]) + 1
    trainer = object.__new__(Trainer)
    trainer.config = current

    with pytest.raises(ValueError, match="patience"):
        trainer._validate_resume_config(saved)


def test_run_metadata_freezes_continuous_session_loader_contract():
    config = load_config("configs/default.yaml")
    metadata = _build_run_metadata(
        args=Namespace(config="configs/default.yaml", resume=None),
        config=config,
        device=torch.device("cpu"),
        train_sessions=1,
        val_sessions=1,
        train_unique_steps=709,
        train_batches_per_epoch=1,
        val_batches_per_epoch=1,
        model=torch.nn.Linear(2, 2),
        provenance={"provenance_sha256": "a" * 64},
    )

    protocol = metadata["sequence_protocol"]
    assert protocol["name"] == "continuous_session"
    assert protocol["dataset_adapter"] == "StitchedManifestDataset"
    assert protocol["unit"] == "complete_session"
    assert protocol["sampler"] == "SequentialSampler"
    assert protocol["shuffle"] is False
    assert protocol["drop_last"] is False
    assert protocol["session_batch_size"] == 1
    assert protocol["train_sessions"] == 1
    assert protocol["val_sessions"] == 1
    assert protocol["train_unique_steps"] == 709
    assert protocol["train_batches_per_epoch"] == 1
    assert protocol["val_batches_per_epoch"] == 1
    assert protocol["row_accounting"] == "one unique (session_id, row_index) per epoch"


def test_run_metadata_freezes_causal_window_training_and_complete_validation():
    config = load_config("configs/eptnet_v6_marlin11_4060_windowed_seed42.yaml")
    metadata = _build_run_metadata(
        args=Namespace(
            config="configs/eptnet_v6_marlin11_4060_windowed_seed42.yaml", resume=None
        ),
        config=config,
        device=torch.device("cpu"),
        train_sessions=7,
        val_sessions=1,
        train_unique_steps=1628,
        train_batches_per_epoch=80,
        val_batches_per_epoch=1,
        model=torch.nn.Linear(2, 2),
        provenance={"provenance_sha256": "a" * 64},
    )

    protocol = metadata["sequence_protocol"]
    assert protocol["name"] == "causal_windows"
    assert protocol["dataset_adapter"] == "CausalTrainingWindowDataset"
    assert protocol["unit"] == "causal_window"
    assert protocol["validation_unit"] == "complete_session"
    assert protocol["shuffle"] is True
    assert protocol["window_size"] == 128
    assert protocol["window_stride"] == 32
    assert protocol["window_warmup_steps"] == 32
    assert protocol["evaluation_uses_windows"] is False


def test_new_run_refuses_to_overwrite_existing_evidence(tmp_path):
    output_dir = tmp_path / "seed_42"
    output_dir.mkdir()
    (output_dir / "history.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _reject_accidental_overwrite(output_dir, resume_from=None)

    _reject_accidental_overwrite(output_dir, resume_from=str(output_dir / "last.pt"))


def test_cpu_resume_is_bitwise_equivalent_to_uninterrupted_training(tmp_path, monkeypatch):
    """N epochs and K epochs + resume must produce the same training state.

    The shuffled loader and dropout layer make the assertion sensitive to both
    DataLoader-generator and global Torch RNG restoration.  The fixed validation
    signal forces ReduceLROnPlateau and stale-epoch state to progress every epoch.
    Wall-clock timing is replaced because it is observability metadata rather
    than mathematical training state.
    """

    fake_clock = itertools.count()
    monkeypatch.setattr(
        engine_module.time,
        "perf_counter",
        lambda: next(fake_clock) * 0.125,
    )

    total_epochs = 4
    split_epoch = 2
    full_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"

    full_trainer = _build_toy_trainer(full_dir, epochs=total_epochs)
    full_result = full_trainer.fit()

    first_leg_trainer = _build_toy_trainer(resumed_dir, epochs=split_epoch)
    first_leg_result = first_leg_trainer.fit()
    resumed_trainer = _build_toy_trainer(resumed_dir, epochs=total_epochs)
    resumed_result = resumed_trainer.fit(resume_from=resumed_dir / "last.pt")

    assert full_result["last_epoch"] == resumed_result["last_epoch"] == total_epochs
    assert full_result["epochs_recorded"] == resumed_result["epochs_recorded"] == total_epochs
    assert first_leg_result["last_epoch"] == split_epoch
    assert resumed_result["epochs_this_run"] == total_epochs - split_epoch

    full_checkpoint = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    resumed_checkpoint = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)

    for key in (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "epoch",
        "best_val_loss",
        "stale_epochs",
        "rng_state",
        "history",
    ):
        _assert_nested_equal(resumed_checkpoint[key], full_checkpoint[key])

    full_history = json.loads((full_dir / "history.json").read_text(encoding="utf-8"))
    resumed_history = json.loads((resumed_dir / "history.json").read_text(encoding="utf-8"))
    assert resumed_history == full_history == full_checkpoint["history"]

    # This is not a vacuous scheduler-state comparison: the learning rate is
    # reduced after every non-improving validation epoch.
    assert [record["learning_rate"] for record in full_history] == [
        0.05,
        0.05,
        0.025,
        0.0125,
    ]
    assert [record["next_learning_rate"] for record in full_history] == [
        0.05,
        0.025,
        0.0125,
        0.00625,
    ]
