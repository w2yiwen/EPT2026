from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
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
HEIGHT_MM = 120.0
BRANCH_NAMES = ("EEG time", "EEG spectrum", "PPG")
BRANCH_COLORS = (COLORS["blue"], COLORS["purple"], COLORS["teal"])
BRANCH_LINESTYLES = ("-", (0, (4, 1.5)), (0, (1.2, 1.2)))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_predictions(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    sequences: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                raise ValueError(f"{path}:{line_number} must contain a string sample_id")
            sequences.setdefault(row["sample_id"], []).append(row)
    if not sequences:
        raise ValueError(f"No prediction records in {path}")
    for sample_id, rows in sequences.items():
        row_indices = [int(row["row_index"]) for row in rows]
        if any(right <= left for left, right in zip(row_indices, row_indices[1:])):
            raise ValueError(f"row_index must increase strictly within {sample_id!r}")
    return sequences


def _thresholds(metrics: Mapping[str, Any], explicit: float | None) -> tuple[float, float | None, str]:
    calibration = metrics.get("calibration")
    protocol = metrics.get("protocol")
    if explicit is not None:
        frame = finite(explicit, "frame_threshold")
        label = "Frame threshold"
    elif isinstance(calibration, Mapping) and calibration.get("frame_threshold") is not None:
        frame = finite(calibration["frame_threshold"], "calibration.frame_threshold")
        label = (
            "Validation-selected threshold"
            if bool(calibration.get("enabled", True))
            and calibration.get("selection_split", "validation") == "validation"
            else "Frame threshold"
        )
    elif isinstance(protocol, Mapping) and protocol.get("frame_threshold") is not None:
        frame = finite(protocol["frame_threshold"], "protocol.frame_threshold")
        label = "Frame threshold"
    else:
        raise ValueError(
            "No frame threshold found. Pass --metrics from evaluate.py or --frame-threshold."
        )
    if not 0.0 <= frame <= 1.0:
        raise ValueError("frame_threshold must lie in [0, 1]")
    boundary = None
    if isinstance(protocol, Mapping) and protocol.get("boundary_threshold") is not None:
        boundary = finite(protocol["boundary_threshold"], "protocol.boundary_threshold")
        if not 0.0 <= boundary <= 1.0:
            raise ValueError("boundary_threshold must lie in [0, 1]")
    return frame, boundary, label


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive_classes = {int(row["positive_class"]) for row in rows}
    if len(positive_classes) != 1:
        raise ValueError("positive_class must be constant within a sequence")
    if positive_classes != {0}:
        raise ValueError(
            "The frozen EPT protocol requires positive_class=0 "
            "(0=deception, 1=truth)"
        )
    event_probabilities = np.asarray(
        [finite(row["positive_probability"], "positive_probability") for row in rows]
    )
    boundaries = np.asarray(
        [
            [finite(value, "boundary_probability") for value in row["boundary_probabilities"]]
            for row in rows
        ]
    )
    centers = np.asarray(
        [[finite(value, "read_center") for value in row["read_centers"]] for row in rows]
    )
    widths = np.asarray(
        [[finite(value, "read_width") for value in row["read_widths"]] for row in rows]
    )
    if boundaries.shape != (len(rows), 2):
        raise ValueError(f"boundary_probabilities must have shape [T,2], got {boundaries.shape}")
    if centers.shape != (len(rows), 3) or widths.shape != (len(rows), 3):
        raise ValueError(
            "read_centers/read_widths must have shape [T,3] in branch order "
            "(EEG time, EEG spectrum, PPG)"
        )
    for name, values in (
        ("positive_probability", event_probabilities),
        ("boundary_probabilities", boundaries),
        ("read_centers", centers),
        ("read_widths", widths),
    ):
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} must lie in [0,1]")
    target_valid = np.asarray([bool(row["target_valid"]) for row in rows], dtype=bool)
    if not target_valid.any():
        raise ValueError("Selected sequence contains no target-valid steps")
    return {
        "positive_class": next(iter(positive_classes)),
        "event_probability": event_probabilities,
        "boundary_probability": boundaries,
        "centers": centers,
        "widths": widths,
        "target_valid": target_valid,
    }


def _coordinates(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    has_times = [
        "start_time_seconds" in row and "end_time_seconds" in row for row in rows
    ]
    if any(has_times) and not all(has_times):
        raise ValueError("Timestamps must be present for every step or for none")
    if all(has_times):
        starts = np.asarray([finite(row["start_time_seconds"], "start_time_seconds") for row in rows])
        ends = np.asarray([finite(row["end_time_seconds"], "end_time_seconds") for row in rows])
        if np.any(ends < starts) or np.any(starts[1:] < starts[:-1]):
            raise ValueError("Timestamps must be chronological with end >= start")
        origin = float(starts[0])
        return starts - origin, ends - origin, (starts + ends) / 2.0 - origin, "Time (s)"
    starts = np.arange(len(rows), dtype=float)
    ends = starts + 1.0
    return starts, ends, (starts + ends) / 2.0, "Sequence step"


def _event_span(event: Mapping[str, Any], starts: np.ndarray, ends: np.ndarray) -> tuple[float, float]:
    start_step = finite(event["start"], "event.start")
    end_step = finite(event["end"], "event.end")
    if start_step > end_step:
        raise ValueError("event.start cannot exceed event.end")
    maximum = len(starts) - 1
    if start_step < 0 or end_step > maximum:
        raise ValueError(f"Event [{start_step}, {end_step}] exceeds sequence [0, {maximum}]")
    grid = np.arange(len(starts), dtype=float)
    left = float(np.interp(start_step, grid, starts))
    right = float(np.interp(end_step, grid, ends))
    return left, right


def _contiguous_spans(mask: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start_index: int | None = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start_index is None:
            start_index = index
        elif not active and start_index is not None:
            spans.append((float(starts[start_index]), float(ends[index - 1])))
            start_index = None
    return spans


def _write_demo_inputs() -> tuple[Path, Path, Path, str]:
    demo_dir = FIG_ROOT / "_demo" / "dynamic_tracking"
    demo_dir.mkdir(parents=True, exist_ok=True)
    predictions = demo_dir / "DEMO_predictions.jsonl"
    events = demo_dir / "DEMO_events.json"
    metrics = demo_dir / "DEMO_metrics.json"
    sample_id = "DEMO-SYNTHETIC-01"
    steps = 84
    threshold = 0.58
    rows = []
    for step in range(steps):
        target = 17 <= step <= 31 or 52 <= step <= 65
        score = 0.14 + 0.67 * max(
            math.exp(-0.5 * ((step - 24) / 7.2) ** 2),
            math.exp(-0.5 * ((step - 59) / 6.0) ** 2),
        ) + 0.025 * math.sin(step / 2.8)
        score = min(max(score, 0.01), 0.98)
        start_probability = max(
            0.04,
            0.92 * math.exp(-0.5 * ((step - 18) / 1.8) ** 2),
            0.86 * math.exp(-0.5 * ((step - 53) / 2.0) ** 2),
        )
        end_probability = max(
            0.05,
            0.88 * math.exp(-0.5 * ((step - 31) / 1.8) ** 2),
            0.91 * math.exp(-0.5 * ((step - 65) / 1.9) ** 2),
        )
        centers = [
            0.76 - 0.20 * score + 0.025 * math.sin(step / 5),
            0.57 - 0.12 * score + 0.025 * math.cos(step / 7),
            0.38 + 0.18 * score + 0.020 * math.sin(step / 8),
        ]
        widths = [0.18 + 0.18 * score, 0.31 + 0.10 * score, 0.42 - 0.10 * score]
        raw = score >= threshold
        rows.append(
            {
                "sample_id": sample_id,
                "row_index": step,
                "label": 0 if target else 1,
                "positive_class": 0,
                "target_valid": not (40 <= step <= 44),
                "positive_probability": score,
                "raw_predicted_positive": raw,
                "predicted_positive": raw and not (40 <= step <= 44),
                "class_probabilities": [score, 1.0 - score],
                "boundary_probabilities": [start_probability, end_probability],
                "offsets": [5.0, 6.0],
                "start_time_seconds": step * 0.75,
                "end_time_seconds": (step + 1) * 0.75,
                "read_centers": centers,
                "read_widths": widths,
            }
        )
    predictions.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    events.write_text(
        json.dumps(
            {
                "predictions": [[
                    {"start": 18.0, "end": 33.0, "score": 0.80, "emit_step": 19.0},
                    {"start": 53.0, "end": 66.0, "score": 0.82, "emit_step": 54.0},
                ]],
                "targets": [[{"start": 17.0, "end": 31.0}, {"start": 52.0, "end": 65.0}]],
            },
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    metrics.write_text(
        json.dumps(
            {
                "protocol": {"frame_threshold": threshold, "boundary_threshold": 0.5},
                "calibration": {
                    "enabled": True,
                    "selection_split": "validation",
                    "frame_threshold": threshold,
                },
            },
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    return predictions, events, metrics, sample_id


def generate(
    *,
    predictions_path: str | Path | None = None,
    events_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    sample_id: str | None = None,
    frame_threshold: float | None = None,
    demo: bool = False,
) -> dict[str, Any]:
    if demo:
        predictions_file, events_file, metrics_file, sample_id = _write_demo_inputs()
    else:
        if predictions_path is None or events_path is None or sample_id is None:
            raise ValueError(
                "Formal rendering requires predictions_path, events_path, and a predeclared sample_id"
            )
        predictions_file = _resolve(predictions_path)
        events_file = _resolve(events_path)
        metrics_file = _resolve(metrics_path) if metrics_path is not None else None

    sequences = _read_predictions(predictions_file)
    assert sample_id is not None
    if sample_id not in sequences:
        available = ", ".join(list(sequences)[:8])
        raise ValueError(f"Unknown sample_id {sample_id!r}; available examples: {available}")
    sample_index = list(sequences).index(sample_id)
    rows = sequences[sample_id]
    values = _validate_rows(rows)
    starts, ends, x, x_label = _coordinates(rows)

    events_payload = _load_json(events_file)
    if not isinstance(events_payload, Mapping):
        raise ValueError("events JSON must contain an object")
    predictions = events_payload.get("predictions")
    targets = events_payload.get("targets")
    if not isinstance(predictions, list) or not isinstance(targets, list):
        raise ValueError("events JSON must contain predictions and targets lists")
    if len(predictions) != len(sequences) or len(targets) != len(sequences):
        raise ValueError(
            "events sequence count must match the first-appearance sample order in predictions JSONL"
        )
    predicted_events = predictions[sample_index]
    target_events = targets[sample_index]
    if not isinstance(predicted_events, list) or not isinstance(target_events, list):
        raise ValueError("Per-sequence predictions and targets must be lists")

    metrics_payload: Mapping[str, Any] = {}
    if metrics_file is not None:
        loaded = _load_json(metrics_file)
        if not isinstance(loaded, Mapping):
            raise ValueError("metrics JSON must contain an object")
        metrics_payload = loaded
    threshold, boundary_threshold, threshold_label = _thresholds(
        metrics_payload, frame_threshold
    )

    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(
        4,
        1,
        figsize=canvas_inches(WIDTH_MM, HEIGHT_MM),
        sharex=True,
        gridspec_kw={"height_ratios": [0.72, 1.55, 1.25, 1.65]},
    )
    interval_axis, probability_axis, boundary_axis, reader_axis = axes
    target_spans = [_event_span(event, starts, ends) for event in target_events]
    prediction_spans = [_event_span(event, starts, ends) for event in predicted_events]
    invalid_spans = _contiguous_spans(~values["target_valid"], starts, ends)

    for left, right in target_spans:
        interval_axis.broken_barh(
            [(left, right - left)], (0.62, 0.56), facecolor=COLORS["coral"], edgecolor="none"
        )
        probability_axis.axvspan(left, right, color=COLORS["coral_light"], alpha=0.26, lw=0)
    for left, right in prediction_spans:
        interval_axis.broken_barh(
            [(left, right - left)], (-0.18, 0.56), facecolor=COLORS["blue"], edgecolor="none"
        )
        probability_axis.axvspan(left, right, color=COLORS["blue_light"], alpha=0.20, lw=0)
    interval_axis.set_ylim(-0.34, 1.33)
    interval_axis.set_yticks([0.10, 0.90], ["Prediction", "Ground truth"])
    interval_axis.grid(False)
    interval_axis.spines["left"].set_visible(False)
    interval_axis.tick_params(axis="y", length=0)

    probability_axis.plot(
        x,
        values["event_probability"],
        color=COLORS["blue_dark"],
        linewidth=1.25,
        label=r"$p_t^{event}$",
    )
    probability_axis.axhline(
        threshold,
        color=COLORS["charcoal"],
        linestyle=(0, (3, 2)),
        linewidth=0.85,
        label=f"{threshold_label} ({threshold:.2f})",
    )
    probability_axis.set_ylabel("Probability")
    probability_axis.set_ylim(0, 1.02)

    boundary_axis.plot(
        x,
        values["boundary_probability"][:, 0],
        color=COLORS["teal"],
        linewidth=1.05,
        label="Start",
    )
    boundary_axis.plot(
        x,
        values["boundary_probability"][:, 1],
        color=COLORS["orange"],
        linewidth=1.05,
        label="End",
    )
    if boundary_threshold is not None:
        boundary_axis.axhline(
            boundary_threshold,
            color=COLORS["charcoal"],
            linestyle=(0, (2, 2)),
            linewidth=0.65,
        )
    boundary_axis.set_ylabel("Boundary prob.")
    boundary_axis.set_ylim(0, 1.02)

    for branch, (name, color, linestyle) in enumerate(
        zip(BRANCH_NAMES, BRANCH_COLORS, BRANCH_LINESTYLES)
    ):
        center = values["centers"][:, branch]
        half_width = values["widths"][:, branch] / 2.0
        reader_axis.fill_between(
            x,
            np.clip(center - half_width, 0.0, 1.0),
            np.clip(center + half_width, 0.0, 1.0),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        reader_axis.plot(
            x, center, color=color, linestyle=linestyle, linewidth=0.95, label=name
        )
    reader_axis.set_ylabel("Cache position")
    reader_axis.set_xlabel(x_label)
    reader_axis.set_ylim(0, 1.02)
    reader_axis.text(
        0.002,
        0.02,
        "older",
        transform=reader_axis.transAxes,
        fontsize=5.8,
        color=COLORS["charcoal"],
        va="bottom",
    )
    reader_axis.text(
        0.002,
        0.98,
        "recent",
        transform=reader_axis.transAxes,
        fontsize=5.8,
        color=COLORS["charcoal"],
        va="top",
    )

    for axis in axes:
        for left, right in invalid_spans:
            axis.axvspan(left, right, color=COLORS["pale_grey"], alpha=0.90, lw=0, zorder=-10)
        axis.set_xlim(float(starts[0]), float(ends[-1]))
    probability_axis.legend(frameon=False, ncol=2, loc="upper right")
    boundary_axis.legend(frameon=False, ncol=2, loc="upper right")
    reader_axis.legend(frameon=False, ncol=3, loc="upper right")
    if invalid_spans:
        interval_axis.legend(
            handles=[Patch(facecolor=COLORS["pale_grey"], edgecolor="none", label="Non-target interval")],
            frameon=False,
            loc="upper right",
        )

    panel_titles = (
        "a  Localized intervals",
        "b  Continuous target-event tracking",
        "c  Boundary evidence",
        "d  Adaptive physiological history",
    )
    for axis, title in zip(axes, panel_titles):
        axis.set_title(title, loc="left", pad=2.0)
    if demo:
        figure.text(
            0.99,
            0.992,
            "DEMO — SYNTHETIC INPUTS FOR LAYOUT QA ONLY",
            ha="right",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            color=COLORS["coral"],
        )

    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.955, hspace=0.36)
    output_name = "fig04_dynamic_tracking_demo" if demo else "fig04_dynamic_tracking"
    sources = [predictions_file, events_file] + ([metrics_file] if metrics_file is not None else [])
    report = export_and_check(
        figure,
        output_stem=Path(__file__).with_name(output_name),
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
        sources=sources,
        data_summary={
            "demo": demo,
            "sample_id": sample_id,
            "sequence_index": sample_index,
            "steps": len(rows),
            "positive_class": values["positive_class"],
            "frame_threshold": threshold,
            "frame_threshold_source": threshold_label,
            "ground_truth_events": len(target_events),
            "predicted_events": len(predicted_events),
            "reader_branch_order": list(BRANCH_NAMES),
        },
    )
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the paper's dynamic probability and temporal-localization figure"
    )
    parser.add_argument("--predictions", help="evaluate.py *_predictions.jsonl")
    parser.add_argument("--events", help="evaluate.py *_events.json")
    parser.add_argument("--metrics", help="evaluate.py test_metrics.json")
    parser.add_argument(
        "--sample-id", help="Predeclared fixed evaluation-view sample identifier"
    )
    parser.add_argument("--frame-threshold", type=float, help="Explicit threshold if metrics is omitted")
    parser.add_argument("--demo", action="store_true", help="Render clearly labelled synthetic layout QA data")
    args = parser.parse_args()
    report = generate(
        predictions_path=args.predictions,
        events_path=args.events,
        metrics_path=args.metrics,
        sample_id=args.sample_id,
        frame_threshold=args.frame_threshold,
        demo=args.demo,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
