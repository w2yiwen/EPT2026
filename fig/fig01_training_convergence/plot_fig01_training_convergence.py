#!/usr/bin/env python3
"""Render loss and validation Event AP@IoU=0.5 convergence curves.

The formal route is deliberately fail-closed: it reads measured training
histories and measured validation metric histories only. It never substitutes
test metrics, interpolates missing evaluations, or smooths trajectories.
"""

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

FIGURE_ID = "fig01_training_convergence"
WIDTH_MM = 190.0
HEIGHT_MM = 90.0
MODEL_ORDER = ("LSTR", "GateHUB", "TeSTra", "EPT-Net (ours)")
MODEL_COLORS = {
    "LSTR": COLORS["slate"],
    "GateHUB": COLORS["teal"],
    "TeSTra": COLORS["orange"],
    "EPT-Net (ours)": COLORS["blue"],
}
DEFAULT_ROOTS = {
    "LSTR": "results/dual4090_seed42_3way/lstr_marlin11_eeg_ppg_video_no_text",
    "GateHUB": "results/dual4090_seed42_3way/gatehub_marlin11_eeg_ppg_video_no_text",
    "TeSTra": "results/dual4090_seed42_3way/testra_marlin11_eeg_ppg_video_no_text",
    "EPT-Net (ours)": "results/dual4090_seed42_3way/eptnet_marlin11_eeg_ppg_video_no_text",
}
METRIC_FILENAME = "validation_epoch_metrics.jsonl"
METRIC_PATH = ("event", "event_ap_iou_0.5")
METRIC_NAME = "event.event_ap_iou_0.5"


@dataclass(frozen=True)
class RunCurve:
    model: str
    seed: int
    run_dir: Path
    history_path: Path
    metric_path: Path
    epochs: np.ndarray
    train_total: np.ndarray
    val_total: np.ndarray
    selected_epoch: int
    metric_epochs: np.ndarray
    metric_values: np.ndarray


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _seed_from_run_dir(path: Path) -> int:
    match = re.fullmatch(r"seed_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Run directory must be named seed_<integer>: {path}")
    return int(match.group(1))


def _metric_from_record(record: dict[str, Any], location: str) -> float:
    split = record.get("split")
    if split is not None and str(split).lower() not in {"val", "validation"}:
        raise ValueError(f"{location}.split must be validation, got {split!r}")

    metric_name = record.get("metric_name", record.get("metric"))
    if "value" in record:
        if metric_name not in {None, METRIC_NAME, "event_ap_iou_0.5"}:
            raise ValueError(
                f"{location} records {metric_name!r}; required metric is {METRIC_NAME!r}"
            )
        value = finite(record["value"], f"{location}.value")
    else:
        node: Any = record.get("metrics", record)
        for key in METRIC_PATH:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(
                    f"{location} must contain value+metric_name or metrics.{METRIC_NAME}"
                )
            node = node[key]
        value = finite(node, f"{location}.{METRIC_NAME}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{location} {METRIC_NAME} must lie in [0, 1], got {value}")
    return value


def _load_metric_records(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing measured validation trajectory: {path}. "
            "Write one JSON object per evaluated epoch as documented in DATA_CONTRACT.md."
        )
    records: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"Validation metric history is empty: {path}")

    points: list[tuple[int, float]] = []
    seen: set[int] = set()
    for index, record in enumerate(records):
        location = f"{path}[{index}]"
        if not isinstance(record, dict):
            raise TypeError(f"{location} must be a JSON object")
        epoch_raw = record.get("epoch")
        if isinstance(epoch_raw, bool) or not isinstance(epoch_raw, (int, float)):
            raise TypeError(f"{location}.epoch must be an integer")
        epoch = int(epoch_raw)
        if float(epoch_raw) != float(epoch) or epoch <= 0:
            raise ValueError(f"{location}.epoch must be a positive integer")
        if epoch in seen:
            raise ValueError(f"Duplicate validation metric for epoch {epoch} in {path}")
        seen.add(epoch)
        points.append((epoch, _metric_from_record(record, location)))
    points.sort()
    return (
        np.asarray([item[0] for item in points], dtype=int),
        np.asarray([item[1] for item in points], dtype=float),
    )


def _load_run(model: str, run_dir: Path) -> RunCurve:
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
            raise TypeError(f"{location} must be an object")
        epoch = int(record.get("epoch", -1))
        if epoch != index + 1:
            raise ValueError(f"{history_path} epochs must be contiguous and one-indexed")
        epochs.append(epoch)
        try:
            train_total.append(finite(record["train"]["total"], f"{location}.train.total"))
            val_total.append(finite(record["val"]["total"], f"{location}.val.total"))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing train/val total loss at {location}") from exc

    val_array = np.asarray(val_total, dtype=float)
    selected_epoch = int(np.argmin(val_array)) + 1
    metric_path = run_dir / METRIC_FILENAME
    metric_epochs, metric_values = _load_metric_records(metric_path)
    if metric_epochs[-1] > epochs[-1]:
        raise ValueError(
            f"Validation metric epoch {metric_epochs[-1]} exceeds final training epoch "
            f"{epochs[-1]} in {run_dir}"
        )
    return RunCurve(
        model=model,
        seed=_seed_from_run_dir(run_dir),
        run_dir=run_dir,
        history_path=history_path,
        metric_path=metric_path,
        epochs=np.asarray(epochs, dtype=int),
        train_total=np.asarray(train_total, dtype=float),
        val_total=val_array,
        selected_epoch=selected_epoch,
        metric_epochs=metric_epochs,
        metric_values=metric_values,
    )


def _parse_specs(specs: Iterable[str] | None) -> dict[str, list[Path]]:
    root = project_root()
    grouped: dict[str, list[Path]] = {model: [] for model in MODEL_ORDER}
    if specs:
        for spec in specs:
            if "=" not in spec:
                raise ValueError(f"Run specification must be MODEL=PATH: {spec!r}")
            model, raw_path = spec.split("=", 1)
            model = model.strip()
            if model not in grouped:
                raise ValueError(f"Unknown model {model!r}; expected one of {MODEL_ORDER}")
            path = Path(raw_path.strip())
            path = path if path.is_absolute() else root / path
            if path.name.startswith("seed_"):
                grouped[model].append(path)
            else:
                grouped[model].extend(sorted(p for p in path.glob("seed_*") if p.is_dir()))
    else:
        for model, relative in DEFAULT_ROOTS.items():
            grouped[model] = sorted(
                p for p in (root / relative).glob("seed_*") if p.is_dir()
            )

    missing = [model for model, paths in grouped.items() if not paths]
    if missing:
        raise FileNotFoundError(f"No run directories found for: {', '.join(missing)}")
    return grouped


def _load_matrix(specs: Iterable[str] | None) -> dict[str, list[RunCurve]]:
    grouped_paths = _parse_specs(specs)
    curves = {
        model: [_load_run(model, path) for path in grouped_paths[model]]
        for model in MODEL_ORDER
    }
    seeds = {model: tuple(sorted(run.seed for run in runs)) for model, runs in curves.items()}
    reference = seeds[MODEL_ORDER[0]]
    mismatched = {model: value for model, value in seeds.items() if value != reference}
    if mismatched:
        raise ValueError(
            "All four models must use an identical matched seed set; "
            f"found {seeds}"
        )

    schedules = {
        (run.model, run.seed): tuple(int(x) for x in run.metric_epochs)
        for runs in curves.values()
        for run in runs
    }
    reference_schedule = next(iter(schedules.values()))
    if any(schedule != reference_schedule for schedule in schedules.values()):
        raise ValueError(
            "Validation AP points must be evaluated at identical epochs for every model/seed; "
            f"found schedules {schedules}"
        )
    return curves


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    mean = np.mean(values, axis=0)
    if values.shape[0] < 2:
        return mean, None
    return mean, np.std(values, axis=0, ddof=1)


def _demo_matrix() -> dict[str, list[RunCurve]]:
    # Synthetic values are confined to the visibly labelled --demo route.
    rng = np.random.default_rng(20261002)
    epochs = np.arange(1, 41)
    metric_epochs = np.asarray([1, 5, 10, 15, 20, 25, 30, 35, 40])
    matrix: dict[str, list[RunCurve]] = {model: [] for model in MODEL_ORDER}
    for model_index, model in enumerate(MODEL_ORDER):
        for seed_index, seed in enumerate((13, 42, 77)):
            train = 1.25 * np.exp(-epochs / 15.0) + 0.18 + rng.normal(0, 0.015, len(epochs))
            val = 1.18 * np.exp(-epochs / 13.0) + 0.28 + 0.002 * np.maximum(epochs - 27, 0) ** 1.45
            val = val + rng.normal(0, 0.02, len(epochs))
            base = 0.36 + 0.035 * model_index
            metric = base + 0.43 * (1 - np.exp(-metric_epochs / 13.0))
            metric = np.clip(metric + rng.normal(0, 0.012, len(metric_epochs)), 0, 1)
            matrix[model].append(
                RunCurve(
                    model=model,
                    seed=seed,
                    run_dir=Path(f"DEMO/{model}/{seed}"),
                    history_path=Path(f"DEMO/{model}/{seed}/history.json"),
                    metric_path=Path(f"DEMO/{model}/{seed}/{METRIC_FILENAME}"),
                    epochs=epochs,
                    train_total=train,
                    val_total=val,
                    selected_epoch=int(np.argmin(val)) + 1,
                    metric_epochs=metric_epochs,
                    metric_values=metric,
                )
            )
    return matrix


def generate(
    *,
    run_specs: Iterable[str] | None = None,
    demo: bool = False,
) -> dict[str, Any]:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    matrix = _demo_matrix() if demo else _load_matrix(run_specs)
    ours = matrix["EPT-Net (ours)"]
    shared_loss_epochs = min(len(run.epochs) for run in ours)
    loss_epochs = np.arange(1, shared_loss_epochs + 1)
    train_values = np.stack([run.train_total[:shared_loss_epochs] for run in ours])
    val_values = np.stack([run.val_total[:shared_loss_epochs] for run in ours])
    train_mean, train_std = _mean_std(train_values)
    val_mean, val_std = _mean_std(val_values)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=canvas_inches(WIDTH_MM, HEIGHT_MM),
        gridspec_kw={"wspace": 0.27},
    )
    loss_axis, metric_axis = axes
    loss_axis.plot(loss_epochs, train_mean, color=COLORS["blue"], linewidth=1.35, label="Training")
    loss_axis.plot(loss_epochs, val_mean, color=COLORS["coral"], linewidth=1.35, label="Validation")
    if train_std is not None:
        loss_axis.fill_between(
            loss_epochs,
            train_mean - train_std,
            train_mean + train_std,
            color=COLORS["blue_light"],
            alpha=0.42,
            linewidth=0,
        )
        loss_axis.fill_between(
            loss_epochs,
            val_mean - val_std,
            val_mean + val_std,
            color=COLORS["coral_light"],
            alpha=0.42,
            linewidth=0,
        )
    selected_epochs = sorted({run.selected_epoch for run in ours})
    for epoch in selected_epochs:
        loss_axis.axvline(
            epoch,
            color=COLORS["charcoal"],
            linestyle=(0, (3, 2)),
            linewidth=0.75,
            alpha=0.6,
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
    handles, labels = loss_axis.get_legend_handles_labels()
    loss_axis.legend(handles + [checkpoint_handle], labels + ["Selected checkpoint"], frameon=False)
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Total loss")
    loss_axis.set_title("(a) Loss convergence", loc="left")
    loss_axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))

    metric_schedule = matrix[MODEL_ORDER[0]][0].metric_epochs
    for model in MODEL_ORDER:
        values = np.stack([run.metric_values for run in matrix[model]])
        mean, std = _mean_std(values)
        color = MODEL_COLORS[model]
        metric_axis.plot(
            metric_schedule,
            mean,
            color=color,
            linewidth=1.35 if model == "EPT-Net (ours)" else 1.05,
            marker="o",
            markersize=2.6,
            markeredgewidth=0,
            label=model,
        )
        if std is not None:
            metric_axis.fill_between(
                metric_schedule,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylabel("Validation event AP@IoU=0.5")
    metric_axis.set_ylim(0, 1)
    metric_axis.set_title("(b) Validation performance", loc="left")
    metric_axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    metric_axis.legend(frameon=False, ncol=2, loc="lower right")

    if demo:
        figure.text(
            0.5,
            0.5,
            "DEMO — SYNTHETIC",
            ha="center",
            va="center",
            fontsize=20,
            color=COLORS["coral"],
            alpha=0.16,
            rotation=18,
            weight="bold",
        )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.92)

    output_name = f"{FIGURE_ID}_demo" if demo else FIGURE_ID
    output_stem = Path(__file__).with_name(output_name)
    sources = [] if demo else [
        path
        for runs in matrix.values()
        for run in runs
        for path in (run.history_path, run.metric_path)
    ]
    seeds = sorted(run.seed for run in matrix[MODEL_ORDER[0]])
    report = export_and_check(
        figure,
        output_stem=output_stem,
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
        sources=sources,
        data_summary={
            "demo": demo,
            "models": list(MODEL_ORDER),
            "seeds": seeds,
            "seed_count": len(seeds),
            "loss_aggregation": "epoch-wise arithmetic mean and sample standard deviation across matched seeds; common prefix only",
            "loss_epochs_plotted": int(shared_loss_epochs),
            "selected_checkpoint_rule": "first epoch attaining minimum validation total loss, independently per seed",
            "selected_checkpoint_epochs": [run.selected_epoch for run in ours],
            "validation_metric": METRIC_NAME,
            "validation_metric_epochs": [int(value) for value in metric_schedule],
            "metric_aggregation": "epoch-wise arithmetic mean and sample standard deviation across matched seeds; no smoothing or interpolation",
            "single_seed_behavior": "no uncertainty band",
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
        metavar="MODEL=PATH",
        help=(
            "Experiment root or seed_<n> run directory. Repeat as needed. "
            "Without this option the four declared result roots are discovered automatically."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Render a visibly watermarked synthetic layout/QA preview only",
    )
    args = parser.parse_args()
    report = generate(run_specs=args.run, demo=args.demo)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
