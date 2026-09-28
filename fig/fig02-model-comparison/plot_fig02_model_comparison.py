from __future__ import annotations

import sys
from pathlib import Path

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
    load_json,
    project_root,
)

MAIN = "results/eptnet_v6_marlin11_4060_windowed_seed42/seed_42/test_metrics.json"
GRU = "results/baseline_early_fusion_gru_marlin11_4060_windowed_seed42/seed_42/test_metrics.json"
CLASSICAL = "results/baselines/marlin11_classical_seed42.json"


def generate():
    configure_matplotlib()
    import matplotlib.pyplot as plt

    main, gru, classical = load_json(MAIN), load_json(GRU), load_json(CLASSICAL)
    metrics = (("Macro-F1", "macro_f1"), ("Balanced acc.", "balanced_accuracy"), ("AUROC", "auroc"), ("Average precision", "average_precision"))
    models = (
        ("Majority", classical["majority"], COLORS["charcoal"]),
        ("Logistic", classical["logistic_regression"], COLORS["orange"]),
        ("Early-fusion GRU", gru["frame"], COLORS["teal"]),
        ("EPT-Net", main["frame"], COLORS["blue"]),
    )
    values = np.asarray([[finite(payload[key], f"{name}.{key}") for _, key in metrics] for name, payload, _ in models])
    if np.any((values < 0) | (values > 1)):
        raise ValueError("All comparison metrics must lie in [0, 1]")

    figure, axis = plt.subplots(figsize=canvas_inches(90, 90))
    x = np.arange(len(metrics))
    width = 0.19
    for index, (name, _, color) in enumerate(models):
        axis.bar(x + (index - 1.5) * width, values[index], width=width, label=name, color=color)
    axis.set_xticks(x, [name for name, _ in metrics], rotation=22, ha="right")
    axis.set_ylabel("Held-out test score")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    figure.subplots_adjust(left=0.17, right=0.98, bottom=0.25, top=0.94)
    sources = [project_root() / item for item in (MAIN, GRU, CLASSICAL)]
    report = export_and_check(
        figure,
        output_stem=Path(__file__).with_name("fig02-model-comparison"),
        width_mm=90,
        height_mm=90,
        sources=sources,
        data_summary={"models": [name for name, _, _ in models], "metrics": [name for name, _ in metrics]},
    )
    plt.close(figure)
    return report


if __name__ == "__main__":
    generate()
