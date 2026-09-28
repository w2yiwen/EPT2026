from __future__ import annotations

import sys
from pathlib import Path

FIG_ROOT = Path(__file__).resolve().parents[1]
if str(FIG_ROOT) not in sys.path:
    sys.path.insert(0, str(FIG_ROOT))

from _paper import (  # noqa: E402
    COLORS,
    canvas_inches,
    configure_matplotlib,
    export_and_check,
    finite,
    load_json,
    project_root,
)

SOURCE = "results/eptnet_v6_marlin11_4060_windowed_seed42/seed_42/history.json"


def generate():
    configure_matplotlib()
    import matplotlib.pyplot as plt

    raw = load_json(SOURCE)
    if not isinstance(raw, list) or not raw:
        raise ValueError("history.json must contain at least one epoch")
    epochs = [int(row["epoch"]) for row in raw]
    if epochs != list(range(1, len(raw) + 1)):
        raise ValueError("Epochs must be contiguous and one-indexed")

    figure, axes = plt.subplots(2, 1, figsize=canvas_inches(90, 90), sharex=True)
    series = (
        ("Total objective", "total"),
        ("Classification loss", "classification"),
    )
    for axis, (ylabel, key) in zip(axes, series, strict=True):
        axis.plot(epochs, [finite(r["train"][key], key) for r in raw], color=COLORS["blue"], label="Train")
        axis.plot(epochs, [finite(r["val"][key], key) for r in raw], color=COLORS["coral"], linestyle="--", label="Validation")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Epoch")
    figure.subplots_adjust(left=0.19, right=0.96, bottom=0.14, top=0.97, hspace=0.18)
    source = project_root() / SOURCE
    report = export_and_check(
        figure,
        output_stem=Path(__file__).with_name("fig01-training-dynamics"),
        width_mm=90,
        height_mm=90,
        sources=[source],
        data_summary={"epochs": len(raw), "best_validation_epoch": int(epochs[min(range(len(raw)), key=lambda i: raw[i]["val"]["total"])])},
    )
    plt.close(figure)
    return report


if __name__ == "__main__":
    generate()
