from __future__ import annotations

import json
from pathlib import Path

from eptnet.training_artifacts import (
    estimate_full_training_runtime,
    generate_training_artifacts,
    load_training_history,
)


def _record(epoch: int) -> dict:
    train = {
        "total": 1.0 / epoch,
        "classification": 0.5 / epoch,
        "boundary": 0.3 / epoch,
        "offset": 0.2 / epoch,
        "smooth": 0.1 / epoch,
        "elapsed_seconds": 2.0,
    }
    val = {
        "total": 1.2 / epoch,
        "classification": 0.6 / epoch,
        "boundary": 0.4 / epoch,
        "offset": 0.2 / epoch,
        "smooth": 0.1 / epoch,
        "elapsed_seconds": 0.5,
    }
    return {
        "epoch": epoch,
        "train": train,
        "val": val,
        "learning_rate": 3e-4,
        "next_learning_rate": 3e-4,
        "best_val_loss": val["total"],
        "stale_epochs": 0,
        "improved": True,
        "elapsed_seconds": 2.5,
        "max_cuda_memory_bytes": 512 * 1024**2,
    }


def test_training_figure_exports_pdf_png_csv_and_manifest(tmp_path: Path):
    history = [_record(1), _record(2)]
    (tmp_path / "history.json").write_text(
        json.dumps(history, allow_nan=False), encoding="utf-8"
    )

    manifest = generate_training_artifacts(tmp_path)

    figure_dir = tmp_path / "figures"
    expected = {
        "training_history.csv",
        "fig_training_dynamics.pdf",
        "fig_training_dynamics.png",
    }
    assert expected == set(manifest["outputs"])
    assert (figure_dir / "fig_training_dynamics.pdf").read_bytes().startswith(b"%PDF")
    assert (figure_dir / "fig_training_dynamics.png").read_bytes().startswith(b"\x89PNG")
    assert (figure_dir / "training_figure_manifest.json").is_file()
    assert manifest["scientific_scope"].startswith("Optimization diagnostics only")


def test_runtime_estimate_scales_train_and_validation_steps_separately():
    estimate = estimate_full_training_runtime(
        [_record(1)],
        observed_train_steps=100,
        observed_val_steps=50,
        full_train_steps=1000,
        full_val_steps=200,
        maximum_epochs=100,
        patience=15,
        uncertainty_fraction=0.25,
    )

    # 2s * 10 train scaling + 0.5s * 4 validation scaling.
    assert estimate["estimated_seconds_per_epoch"]["point_seconds"] == 22.0
    assert estimate["estimated_until_earliest_patience_stop"]["epochs"] == 16
    assert estimate["estimated_configured_maximum"]["point_seconds"] == 2200.0


def test_history_loader_rejects_non_contiguous_epochs(tmp_path: Path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([_record(2)]), encoding="utf-8")

    try:
        load_training_history(path)
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("Expected a non-contiguous history error")
