from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from eptnet.engine import Trainer
from eptnet.evaluate import _validate_checkpoint_provenance
from eptnet.training import trainer as trainer_module


def _provenance(
    identity: str,
    *,
    source_identity: str = "1" * 64,
    data_identity: str = "2" * 64,
) -> dict:
    return {
        "schema_version": 1,
        "provenance_sha256": identity,
        "source_tree": {"sha256": source_identity},
        "required_artifacts": {},
        "manifests": [
            {
                "path": "data/processed/fixture/manifests/train.jsonl",
                "manifest_and_tensors_sha256": data_identity,
            }
        ],
    }


@pytest.mark.parametrize(
    "current",
    (
        _provenance("b" * 64, source_identity="3" * 64),
        _provenance("c" * 64, data_identity="4" * 64),
    ),
    ids=("source-change", "prepared-data-change"),
)
def test_evaluation_rejects_changed_source_or_prepared_data(current):
    checkpoint = {"provenance": _provenance("a" * 64)}

    with pytest.raises(ValueError, match="provenance does not match"):
        _validate_checkpoint_provenance(checkpoint, current)


@pytest.mark.parametrize(
    "current",
    (
        _provenance("b" * 64, source_identity="3" * 64),
        _provenance("c" * 64, data_identity="4" * 64),
    ),
    ids=("source-change", "prepared-data-change"),
)
def test_resume_rejects_changed_source_or_prepared_data(current):
    trainer = object.__new__(Trainer)
    trainer.provenance = current

    with pytest.raises(ValueError, match="source or prepared-data provenance changed"):
        trainer._validate_resume_provenance(_provenance("a" * 64))


def test_matching_provenance_is_accepted_for_evaluation_and_resume():
    saved = _provenance("a" * 64)
    current = deepcopy(saved)
    # Git availability/state is diagnostic metadata and is intentionally not
    # part of the portable content identity.
    current["git"] = {"commit": "f" * 40, "dirty": False}

    _validate_checkpoint_provenance({"provenance": saved}, current)
    trainer = object.__new__(Trainer)
    trainer.provenance = current
    trainer._validate_resume_provenance(saved)


def test_evaluation_and_resume_reject_missing_provenance():
    current = _provenance("a" * 64)
    with pytest.raises(ValueError, match="missing the required provenance"):
        _validate_checkpoint_provenance({}, current)

    trainer = object.__new__(Trainer)
    trainer.provenance = current
    with pytest.raises(ValueError, match="provenance metadata is missing"):
        trainer._validate_resume_provenance(None)


def test_checkpoint_schema_v3_persists_an_independent_provenance_record():
    trainer = object.__new__(Trainer)
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scheduler = None
    trainer.scaler = None
    trainer.config = {"experiment": {"seed": 42}}
    trainer.train_loader = []
    trainer.val_loader = []
    trainer.provenance = _provenance("a" * 64)

    payload = trainer._checkpoint_payload(
        epoch=1,
        best_val_loss=0.5,
        stale_epochs=0,
        history=[],
    )

    assert payload["schema_version"] == Trainer.CHECKPOINT_SCHEMA_VERSION == 3
    assert payload["provenance"] == trainer.provenance
    assert payload["provenance"] is not trainer.provenance


def test_training_provenance_uses_repository_root(tmp_path, monkeypatch):
    manifests = tmp_path / "dataset" / "manifests"
    manifests.mkdir(parents=True)
    paths = {}
    for split in ("train", "val", "test"):
        path = manifests / f"{split}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        paths[f"{split}_manifest"] = str(path)

    observed = {}

    def capture(project_root, dataset_root, **kwargs):
        observed["project_root"] = Path(project_root)
        observed["dataset_root"] = Path(dataset_root)
        return {"status": "captured"}

    monkeypatch.setattr(trainer_module, "build_provenance_fingerprint", capture)
    result = trainer_module.build_runtime_provenance({"data": paths})

    assert result == {"status": "captured"}
    assert observed["project_root"] == Path(__file__).resolve().parents[1]
    assert observed["dataset_root"] == manifests.parent
