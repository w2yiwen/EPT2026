#!/usr/bin/env python3
"""Render the measured EPT-Net training/validation total-loss trajectory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FIG_ROOT = Path(__file__).resolve().parents[1]
if str(FIG_ROOT) not in sys.path:
    sys.path.insert(0, str(FIG_ROOT))

from _paper import (  # noqa: E402
    COLORS,
    canvas_inches,
    configure_matplotlib,
    export_and_check,
    finite,
    project_root,
)

FIGURE_ID = "fig01_training_loss"
WIDTH_MM = 90.0
HEIGHT_MM = 90.0
DEFAULT_ROOT = (
    "results/dual4090_seed42_3way/"
    "eptnet_marlin11_eeg_ppg_video_no_text"
)


@dataclass(frozen=True)
class LossRun:
    seed: int
    history_path: Path
    epochs: np.ndarray
    train_total: np.ndarray
    val_total: np.ndarray
    selected_epoch: int


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _seed(path: Path) -> int:
    match = re.fullmatch(r"seed_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Run directory must be named seed_<integer>: {path}")
    return int(match.group(1))


def _load_run(run_dir: Path) -> LossRun:
    run_dir = run_dir.resolve()
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing training history: {history_path}")
    history = _load_json(history_path)
    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history must be a non-empty JSON list: {history_path}")

    epochs: list[int] = []
    train_total: list[float] = []
    val_total: list[float] = []
    for index, record in enumerate(history):
        location = f"{history_path}[{index}]"
        if not isinstance(record, dict):
            raise TypeError(f"{location} must be a JSON object")
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, float)):
            raise TypeError(f"{location}.epoch must be an integer")
        epoch = int(epoch)
        if epoch != index + 1:
            raise ValueError(f"{history_path} epochs must be contiguous and one-indexed")
        try:
            train = finite(record["train"]["total"], f"{location}.train.total")
            validation = finite(record["val"]["total"], f"{location}.val.total")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing train/val total loss at {location}") from exc
        epochs.append(epoch)
        train_total.append(train)
        val_total.append(validation)

    val_array = np.asarray(val_total, dtype=float)
    return LossRun(
        seed=_seed(run_dir),
        history_path=history_path,
        epochs=np.asarray(epochs, dtype=int),
        train_total=np.asarray(train_total, dtype=float),
        val_total=val_array,
        selected_epoch=int(np.argmin(val_array)) + 1,
    )


def _discover(specs: Iterable[str] | None) -> list[Path]:
    root = project_root()
    raw_paths = list(specs or [DEFAULT_ROOT])
    run_dirs: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        path = path if path.is_absolute() else root / path
        if path.name.startswith("seed_"):
            run_dirs.append(path)
        else:
            run_dirs.extend(sorted(p for p in path.glob("seed_*") if p.is_dir()))
    unique = sorted({path.resolve() for path in run_dirs})
    if not unique:
        raise FileNotFoundError(f"No seed_<integer> run directories found under {raw_paths}")
    return unique


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    mean = np.mean(values, axis=0)
    if values.shape[0] == 1:
        return mean, None
    return mean, np.std(values, axis=0, ddof=1)


def generate(*, run_specs: Iterable[str] | None = None) -> dict[str, Any]:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    runs = [_load_run(path) for path in _discover(run_specs)]
    seeds = [run.seed for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seed identifiers: {seeds}")

    common_epochs = min(len(run.epochs) for run in runs)
    epochs = np.arange(1, common_epochs + 1)
    train = np.stack([run.train_total[:common_epochs] for run in runs])
    validation = np.stack([run.val_total[:common_epochs] for run in runs])
    train_mean, train_std = _mean_std(train)
    val_mean, val_std = _mean_std(validation)

    figure, axis = plt.subplots(figsize=canvas_inches(WIDTH_MM, HEIGHT_MM))
    axis.plot(epochs, train_mean, color=COLORS["blue"], linewidth=1.35, label="Training")
    axis.plot(epochs, val_mean, color=COLORS["coral"], linewidth=1.35, label="Validation")
    if train_std is not None:
        axis.fill_between(
            epochs,
            train_mean - train_std,
            train_mean + train_std,
            color=COLORS["blue_light"],
            alpha=0.42,
            linewidth=0,
        )
        axis.fill_between(
            epochs,
            val_mean - val_std,
            val_mean + val_std,
            color=COLORS["coral_light"],
            alpha=0.42,
            linewidth=0,
        )

    selected_epochs = sorted({run.selected_epoch for run in runs})
    for epoch in selected_epochs:
        axis.axvline(
            epoch,
            color=COLORS["charcoal"],
            linestyle=(0, (3, 2)),
            linewidth=0.75,
            alpha=0.7,
            zorder=0,
        )
    checkpoint_handle = Line2D(
        [0],
        [0],
        color=COLORS["charcoal"],
        linestyle=(0, (3, 2)),
        linewidth=0.75,
        label="Selected checkpoint",
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles + [checkpoint_handle], labels + ["Selected checkpoint"], frameon=False)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Total loss")
    axis.set_title("EPT-Net loss convergence", loc="left")
    axis.set_xlim(1, common_epochs + 0.8)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    figure.subplots_adjust(left=0.18, right=0.97, bottom=0.14, top=0.93)

    report = export_and_check(
        figure,
        output_stem=Path(__file__).with_name(FIGURE_ID),
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
        sources=[run.history_path for run in runs],
        data_summary={
            "model": "EPT-Net (ours)",
            "seeds": sorted(seeds),
            "seed_count": len(seeds),
            "epochs_plotted": int(common_epochs),
            "aggregation": "epoch-wise arithmetic mean across seeds over the common completed-epoch prefix",
            "uncertainty": "sample standard deviation (ddof=1)" if len(seeds) > 1 else "not shown; one seed",
            "selected_checkpoint_rule": "first epoch attaining minimum validation total loss, independently per seed",
            "selected_checkpoint_epochs": [run.selected_epoch for run in runs],
            "smoothing": "none",
        },
    )
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="PATH",
        help="Experiment root or explicit seed_<n> run directory; repeat to combine runs",
    )
    args = parser.parse_args()
    print(json.dumps(generate(run_specs=args.run), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
