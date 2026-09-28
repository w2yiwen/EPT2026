from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

LOSS_COMPONENTS = ("classification", "boundary", "offset", "smooth")
OI = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#7A7A7A",
}


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def load_training_history(path: str | Path) -> list[dict[str, Any]]:
    history_path = Path(path)
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Training history must be a non-empty list: {history_path}")
    previous_epoch = 0
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise TypeError(f"history[{index}] must be a mapping")
        epoch = int(_finite_number(raw.get("epoch"), f"history[{index}].epoch"))
        if epoch != previous_epoch + 1:
            raise ValueError("Training history epochs must be contiguous and one-indexed")
        for split in ("train", "val"):
            losses = raw.get(split)
            if not isinstance(losses, dict):
                raise TypeError(f"history[{index}].{split} must be a mapping")
            for name in ("total", *LOSS_COMPONENTS, "elapsed_seconds"):
                _finite_number(losses.get(name), f"history[{index}].{split}.{name}")
        _finite_number(raw.get("learning_rate"), f"history[{index}].learning_rate")
        _finite_number(raw.get("elapsed_seconds"), f"history[{index}].elapsed_seconds")
        records.append(raw)
        previous_epoch = epoch
    return records


def _history_csv(history: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        "epoch",
        "train_total",
        "val_total",
        *(f"train_{name}" for name in LOSS_COMPONENTS),
        *(f"val_{name}" for name in LOSS_COMPONENTS),
        "learning_rate",
        "next_learning_rate",
        "epoch_elapsed_seconds",
        "train_elapsed_seconds",
        "val_elapsed_seconds",
        "max_cuda_memory_bytes",
    ]
    rows: list[dict[str, Any]] = []
    for record in history:
        train = record["train"]
        val = record["val"]
        row: dict[str, Any] = {
            "epoch": int(record["epoch"]),
            "train_total": float(train["total"]),
            "val_total": float(val["total"]),
            "learning_rate": float(record["learning_rate"]),
            "next_learning_rate": float(record["next_learning_rate"]),
            "epoch_elapsed_seconds": float(record["elapsed_seconds"]),
            "train_elapsed_seconds": float(train["elapsed_seconds"]),
            "val_elapsed_seconds": float(val["elapsed_seconds"]),
            "max_cuda_memory_bytes": record.get("max_cuda_memory_bytes", ""),
        }
        for name in LOSS_COMPONENTS:
            row[f"train_{name}"] = float(train[name])
            row[f"val_{name}"] = float(val[name])
        rows.append(row)

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def generate_training_artifacts(run_directory: str | Path) -> dict[str, Any]:
    """Render optimization diagnostics from the immutable JSON history.

    These figures describe optimization only. They do not read the test split and
    must not be presented as held-out model-performance evidence.
    """

    run_dir = Path(run_directory)
    history_path = run_dir / "history.json"
    history = load_training_history(history_path)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / "training_history.csv"
    _atomic_write_text(_history_csv(history), csv_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )

    epochs = [int(record["epoch"]) for record in history]
    mark_every = max(1, len(epochs) // 8)
    figure, axes = plt.subplots(2, 2, figsize=(6.75, 4.8), constrained_layout=True)

    axes[0, 0].plot(
        epochs,
        [float(record["train"]["total"]) for record in history],
        color=OI["blue"],
        marker="o",
        markevery=mark_every,
        label="Train",
    )
    axes[0, 0].plot(
        epochs,
        [float(record["val"]["total"]) for record in history],
        color=OI["vermillion"],
        linestyle="--",
        marker="s",
        markevery=mark_every,
        label="Validation",
    )
    axes[0, 0].set_ylabel("Total objective")
    axes[0, 0].legend()

    component_colors = (OI["blue"], OI["orange"], OI["green"], OI["pink"])
    component_markers = ("o", "s", "^", "D")
    for name, color, marker in zip(
        LOSS_COMPONENTS, component_colors, component_markers, strict=True
    ):
        axes[0, 1].plot(
            epochs,
            [float(record["val"][name]) for record in history],
            color=color,
            marker=marker,
            markevery=mark_every,
            label=name.capitalize(),
        )
    axes[0, 1].set_ylabel("Validation component loss")
    axes[0, 1].legend(ncol=2)

    learning_rates = [float(record["learning_rate"]) for record in history]
    axes[1, 0].plot(
        epochs,
        learning_rates,
        color=OI["green"],
        marker="^",
        markevery=mark_every,
    )
    if all(value > 0 for value in learning_rates):
        axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Learning rate")

    elapsed = [float(record["elapsed_seconds"]) for record in history]
    axes[1, 1].bar(epochs, elapsed, color=OI["sky"], width=0.7, label="Epoch time")
    axes[1, 1].set_ylabel("Epoch time (s)")
    memory_values = [record.get("max_cuda_memory_bytes") for record in history]
    if all(isinstance(value, (int, float)) for value in memory_values):
        memory_axis = axes[1, 1].twinx()
        memory_axis.spines["top"].set_visible(False)
        memory_axis.plot(
            epochs,
            [float(value) / 1024**3 for value in memory_values],
            color=OI["vermillion"],
            marker="D",
            markevery=mark_every,
            label="Peak memory",
        )
        memory_axis.set_ylabel("Peak CUDA memory (GiB)")
        lines, labels = axes[1, 1].get_legend_handles_labels()
        memory_lines, memory_labels = memory_axis.get_legend_handles_labels()
        axes[1, 1].legend(lines + memory_lines, labels + memory_labels, loc="best")
    else:
        axes[1, 1].legend(loc="best")

    for panel_index, axis in enumerate(axes.flat):
        axis.set_xlabel("Epoch")
        if len(epochs) == 1:
            axis.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
            axis.set_xticks(epochs)
        else:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.text(
            0.0,
            1.03,
            f"({chr(97 + panel_index)})",
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    pdf_path = figure_dir / "fig_training_dynamics.pdf"
    png_path = figure_dir / "fig_training_dynamics.png"
    figure.savefig(
        pdf_path,
        metadata={"Creator": "EPT-Net", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(png_path, dpi=300, metadata={"Software": "EPT-Net"})
    plt.close(figure)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": {"history.json": _sha256(history_path)},
        "epochs": len(history),
        "scientific_scope": (
            "Optimization diagnostics only; no test data or held-out performance metric is plotted."
        ),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (csv_path, pdf_path, png_path)
        },
    }
    manifest_path = figure_dir / "training_figure_manifest.json"
    _atomic_write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        manifest_path,
    )
    manifest["manifest_path"] = manifest_path.relative_to(run_dir).as_posix()
    return manifest


def estimate_full_training_runtime(
    history: Sequence[Mapping[str, Any]],
    *,
    observed_train_steps: int,
    observed_val_steps: int,
    full_train_steps: int,
    full_val_steps: int,
    maximum_epochs: int,
    patience: int,
    uncertainty_fraction: float = 0.35,
) -> dict[str, Any]:
    """Extrapolate a transparent range from a one-epoch real-session smoke run."""

    if len(history) != 1:
        raise ValueError("Runtime estimation expects exactly one smoke epoch")
    positive_counts = {
        "observed_train_steps": observed_train_steps,
        "observed_val_steps": observed_val_steps,
        "full_train_steps": full_train_steps,
        "full_val_steps": full_val_steps,
        "maximum_epochs": maximum_epochs,
        "patience": patience,
    }
    invalid = [name for name, value in positive_counts.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"Runtime-estimation counts must be positive: {invalid}")
    if not 0.0 <= uncertainty_fraction < 1.0:
        raise ValueError("uncertainty_fraction must be in [0, 1)")

    record = history[0]
    train_seconds = _finite_number(record["train"]["elapsed_seconds"], "train elapsed")
    val_seconds = _finite_number(record["val"]["elapsed_seconds"], "val elapsed")
    estimated_epoch = train_seconds * full_train_steps / observed_train_steps + val_seconds * (
        full_val_steps / observed_val_steps
    )
    earliest_patience_epoch = min(maximum_epochs, patience + 1)

    def interval(seconds: float) -> dict[str, float]:
        return {
            "point_seconds": seconds,
            "lower_seconds": seconds * (1.0 - uncertainty_fraction),
            "upper_seconds": seconds * (1.0 + uncertainty_fraction),
            "point_hours": seconds / 3600.0,
            "lower_hours": seconds * (1.0 - uncertainty_fraction) / 3600.0,
            "upper_hours": seconds * (1.0 + uncertainty_fraction) / 3600.0,
        }

    return {
        "schema_version": 1,
        "method": "linear step-count extrapolation from one full real train session and one full real validation session",
        "observed": {
            "train_steps": observed_train_steps,
            "validation_steps": observed_val_steps,
            "train_seconds": train_seconds,
            "validation_seconds": val_seconds,
        },
        "full_protocol": {
            "train_steps_per_epoch": full_train_steps,
            "validation_steps_per_epoch": full_val_steps,
            "maximum_epochs": maximum_epochs,
            "patience": patience,
        },
        "estimated_seconds_per_epoch": interval(estimated_epoch),
        "estimated_until_earliest_patience_stop": {
            "epochs": earliest_patience_epoch,
            **interval(estimated_epoch * earliest_patience_epoch),
        },
        "estimated_configured_maximum": {
            "epochs": maximum_epochs,
            **interval(estimated_epoch * maximum_epochs),
        },
        "uncertainty_fraction": uncertainty_fraction,
        "limitations": [
            "The first CUDA epoch includes warm-up and can be slower than later epochs.",
            "Session lengths and missing-modality patterns change per-step cost.",
            "Early stopping time is unknown until validation loss is observed.",
            "Evaluation and feature extraction time are not included.",
        ],
    }
