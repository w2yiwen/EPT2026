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


def generate():
    configure_matplotlib()
    import matplotlib.pyplot as plt

    main, gru = load_json(MAIN), load_json(GRU)
    metric_keys = (
        ("Frame macro-F1", "frame_macro_f1"),
        ("Balanced accuracy", "frame_balanced_accuracy"),
        ("Frame AUROC", "frame_auroc"),
        ("Average precision", "frame_average_precision"),
        ("Boundary macro-F1", "boundary_macro_f1"),
        ("Event F1 @ 0.5", "event_f1_iou_0.5"),
        ("Event mAP", "event_map"),
    )
    figure, axis = plt.subplots(figsize=canvas_inches(190, 90))
    y = np.arange(len(metric_keys))
    for offset, (name, payload, color, marker) in zip(
        (-0.12, 0.12),
        (("EPT-Net", main, COLORS["blue"], "o"), ("Early-fusion GRU", gru, COLORS["teal"], "s")),
        strict=True,
    ):
        summaries = payload["subject_macro"]["metrics"]
        means = np.asarray([finite(summaries[key]["mean"], key) for _, key in metric_keys])
        lower = np.asarray([finite(summaries[key]["bootstrap_ci95_lower"], key) for _, key in metric_keys])
        upper = np.asarray([finite(summaries[key]["bootstrap_ci95_upper"], key) for _, key in metric_keys])
        if np.any(lower > means) or np.any(means > upper):
            raise ValueError(f"Invalid confidence interval ordering for {name}")
        axis.errorbar(means, y + offset, xerr=np.vstack((means - lower, upper - means)), fmt=marker, color=color, capsize=2.2, label=name)
    axis.set_yticks(y, [label for label, _ in metric_keys])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Held-out subject macro score (95% bootstrap CI)")
    axis.legend(frameon=False, ncol=2, loc="lower right")
    figure.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.95)
    sources = [project_root() / MAIN, project_root() / GRU]
    report = export_and_check(
        figure,
        output_stem=Path(__file__).with_name("fig03-heldout-uncertainty"),
        width_mm=190,
        height_mm=90,
        sources=sources,
        data_summary={"held_out_subjects": int(main["subject_macro"]["num_subjects"]), "metrics": [label for label, _ in metric_keys]},
    )
    plt.close(figure)
    return report


if __name__ == "__main__":
    generate()
