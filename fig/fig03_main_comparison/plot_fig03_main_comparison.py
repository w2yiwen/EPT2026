from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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

WIDTH_MM = 190.0
HEIGHT_MM = 68.02
MODELS = ("EPT-Net", "LSTR", "GateHUB", "TeSTra")
METRICS = (
    ("frame.average_precision", "Frame AP", True),
    ("frame.brier_score", "Brier score", False),
    ("event.event_f1_iou_0.5", "Event F1 @ tIoU 0.5", True),
    ("event.event_map", "Event mAP", True),
)
COLORS_BY_MODEL = {
    "EPT-Net": COLORS["blue"],
    "LSTR": COLORS["charcoal"],
    "GateHUB": COLORS["charcoal"],
    "TeSTra": COLORS["charcoal"],
}
MARKERS = {"EPT-Net": "o", "LSTR": "s", "GateHUB": "D", "TeSTra": "^"}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _read_aggregates(
    specifications: Sequence[str],
) -> tuple[dict[tuple[str, str], float], list[Path], dict[str, list[int]]]:
    table: dict[tuple[str, str], float] = {}
    sources: list[Path] = []
    seeds_by_model: dict[str, list[int]] = {}
    reference_signature: dict[str, Any] | None = None
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("--main-aggregate must use MODEL=PATH syntax")
        model, raw_path = specification.split("=", 1)
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
        if model in seeds_by_model:
            raise ValueError(f"Duplicate aggregate for {model}")
        path = _resolve(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregate = payload.get("aggregate") if isinstance(payload, Mapping) else None
        if not isinstance(aggregate, Mapping):
            raise ValueError(f"{path} is not an aggregate.py JSON artifact")
        seeds = payload.get("seeds")
        if not isinstance(seeds, list) or not seeds or not all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
        ):
            raise ValueError(f"{path} has invalid seed metadata")
        signature = {
            "protocol": payload.get("protocol"),
            "data": payload.get("data"),
            "calibration_protocol": payload.get("calibration_protocol"),
            "provenance_sha256": payload.get("provenance_sha256"),
        }
        if any(value is None for value in signature.values()):
            raise ValueError(f"{path} is missing comparison provenance")
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(f"{path} does not share the same evaluation protocol and data")
        for metric, _, _ in METRICS:
            entry = aggregate.get(metric)
            if not isinstance(entry, Mapping) or entry.get("mean") is None:
                raise ValueError(f"{path}: aggregate.{metric} is missing mean")
            value = finite(entry["mean"], f"{model}.{metric}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{model}.{metric} must lie in [0, 1]")
            table[(model, metric)] = value
        sources.append(path)
        seeds_by_model[model] = list(seeds)
    missing = set(MODELS).difference(seeds_by_model)
    if missing:
        raise ValueError(f"Missing main-comparison aggregates: {sorted(missing)}")
    if any(seeds != [42] for seeds in seeds_by_model.values()):
        raise ValueError(f"The declared main comparison requires seed 42: {seeds_by_model}")
    return table, sources, seeds_by_model


def _demo_data() -> tuple[dict[tuple[str, str], float], list[Path], dict[str, list[int]]]:
    demo_dir = FIG_ROOT / "_demo" / "main_comparison"
    demo_dir.mkdir(parents=True, exist_ok=True)
    path = demo_dir / "DEMO_main_comparison.json"
    values = {
        "EPT-Net": (0.86, 0.11, 0.78, 0.61),
        "LSTR": (0.78, 0.16, 0.69, 0.52),
        "GateHUB": (0.80, 0.15, 0.71, 0.54),
        "TeSTra": (0.82, 0.14, 0.73, 0.56),
    }
    path.write_text(
        json.dumps(
            {
                "demo": True,
                "warning": "Synthetic layout values; never cite as results.",
                "values": values,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    table = {
        (model, metric[0]): value
        for model, row in values.items()
        for metric, value in zip(METRICS, row)
    }
    return table, [path], {model: [42] for model in MODELS}


def generate(*, aggregate_specs: Sequence[str] = (), demo: bool = False) -> dict[str, Any]:
    if demo:
        table, sources, seeds_by_model = _demo_data()
    else:
        table, sources, seeds_by_model = _read_aggregates(aggregate_specs)

    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, 4, figsize=canvas_inches(WIDTH_MM, HEIGHT_MM), sharey=True
    )
    y = np.arange(len(MODELS), dtype=float)
    for axis, (metric, label, higher_is_better) in zip(axes, METRICS):
        for row, model in enumerate(MODELS):
            value = table[(model, metric)]
            axis.hlines(
                y[row], 0, value, color=COLORS_BY_MODEL[model], alpha=0.32, linewidth=1.2
            )
            axis.plot(
                value,
                y[row],
                marker=MARKERS[model],
                color=COLORS_BY_MODEL[model],
                markersize=4.4,
                markeredgewidth=0,
                zorder=3,
            )
            axis.annotate(
                f"{value:.3f}",
                (value, y[row]),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=5.2,
            )
        axis.set_xlim(0, 1.08)
        axis.set_xticks([0, 0.5, 1.0])
        axis.set_xlabel("Score")
        axis.set_title(f"{label} {'↑' if higher_is_better else '↓'}", pad=2.0)
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{model} (seed 42)" for model in MODELS])
    axes[0].invert_yaxis()
    axes[0].get_yticklabels()[0].set_color(COLORS["blue_dark"])
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    if demo:
        figure.text(
            0.57,
            0.5,
            "DEMO · SYNTHETIC",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=COLORS["coral"],
            alpha=0.18,
            rotation=22,
        )
    figure.subplots_adjust(left=0.18, right=0.985, bottom=0.18, top=0.86, wspace=0.24)
    output_name = "fig03_main_comparison_demo" if demo else "fig03_main_comparison"
    report = export_and_check(
        figure,
        output_stem=Path(__file__).with_name(output_name),
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
        sources=sources,
        data_summary={
            "demo": demo,
            "models": list(MODELS),
            "metrics": [label for _, label, _ in METRICS],
            "seeds_by_model": seeds_by_model,
            "plotting_aggregation": "none; reads upstream means",
        },
    )
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the four-model main comparison")
    parser.add_argument(
        "--aggregate", action="append", default=[], metavar="MODEL=PATH"
    )
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(aggregate_specs=args.aggregate, demo=args.demo), indent=2))


if __name__ == "__main__":
    main()
