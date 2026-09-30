from __future__ import annotations

import argparse
import csv
import json
import re
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

WIDTH_MM = 90.0
HEIGHT_MM = 60.0
CONFIGURATIONS = ("Full", "Video")
METRICS = ("AP", "Brier", "Event mAP")
CONFIG_COLORS = {
    "Full": COLORS["blue"],
    "Video": COLORS["coral"],
}
CONFIG_MARKERS = {"Full": "o", "Video": "^"}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _configuration(value: Any) -> str:
    normalized = _token(value)
    aliases = {
        "full": "Full",
        "all": "Full",
        "eegppgvideo": "Full",
        "video": "Video",
        "videoonly": "Video",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown configuration {value!r}; expected Full or Video"
        )
    return aliases[normalized]


def _metric(value: Any) -> str:
    normalized = _token(value)
    aliases = {
        "ap": "AP",
        "averageprecision": "AP",
        "frameaverageprecision": "AP",
        "frameap": "AP",
        "brier": "Brier",
        "brierscore": "Brier",
        "framebrier": "Brier",
        "framebrierscore": "Brier",
        "eventmap": "Event mAP",
        "eventeventmap": "Event mAP",
        "eventmeanaverageprecision": "Event mAP",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown metric {value!r}; expected AP, Brier, or Event mAP"
        )
    return aliases[normalized]


def _records_from_combined(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not all(isinstance(row, dict) for row in payload):
            raise ValueError("Combined JSON list must contain objects")
        return list(payload)
    if not isinstance(payload, dict):
        raise ValueError("Combined evidence JSON must be an object or list")
    if isinstance(payload.get("records"), list):
        return list(payload["records"])
    configurations = payload.get("configurations")
    if isinstance(configurations, list):
        records: list[dict[str, Any]] = []
        for configuration in configurations:
            if not isinstance(configuration, Mapping) or not isinstance(
                configuration.get("metrics"), Mapping
            ):
                raise ValueError("Each configuration must contain name and metrics")
            for metric_name, value in configuration["metrics"].items():
                row = {"configuration": configuration.get("name"), "metric": metric_name}
                if isinstance(value, Mapping):
                    row.update(value)
                else:
                    row["value"] = value
                records.append(row)
        return records
    raise ValueError(
        "Combined evidence JSON must contain records[] or configurations[].metrics"
    )


def _records_from_aggregates(
    specifications: Sequence[str],
) -> tuple[list[dict[str, Any]], list[Path], dict[str, list[int]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    sources: list[Path] = []
    seeds_by_configuration: dict[str, list[int]] = {}
    provenance_by_configuration: dict[str, str] = {}
    reference_signature: dict[str, Any] | None = None
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("--aggregate must use LABEL=PATH syntax")
        label, raw_path = specification.split("=", 1)
        configuration = _configuration(label)
        path = _resolve(raw_path)
        sources.append(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregate = payload.get("aggregate") if isinstance(payload, Mapping) else None
        if not isinstance(aggregate, Mapping):
            raise ValueError(f"{path} is not an aggregate.py JSON artifact")
        signature = {
            "protocol": payload.get("protocol"),
            "data": payload.get("data"),
            "calibration_protocol": payload.get("calibration_protocol"),
        }
        if any(value is None for value in signature.values()):
            raise ValueError(
                f"{path} is missing aggregate protocol/data/provenance metadata"
            )
        seeds = payload.get("seeds")
        if not isinstance(seeds, list) or not seeds or not all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
        ):
            raise ValueError(f"{path} has invalid aggregate seed metadata")
        seeds_by_configuration[configuration] = list(seeds)
        provenance_sha256 = payload.get("provenance_sha256")
        if not isinstance(provenance_sha256, str) or len(provenance_sha256) != 64:
            raise ValueError(f"{path} has an invalid provenance_sha256")
        provenance_by_configuration[configuration] = provenance_sha256
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(
                f"{path} does not use the same evaluation protocol, data scope, "
                "and calibration policy as the other configurations"
            )
        for key, value in aggregate.items():
            try:
                metric = _metric(key)
            except ValueError:
                continue
            if not isinstance(value, Mapping) or value.get("mean") is None:
                raise ValueError(f"{path}: aggregate.{key} is missing mean")
            records.append(
                {
                    "configuration": configuration,
                    "metric": metric,
                    "mean": value["mean"],
                    "n": value.get("n"),
                }
            )
    return records, sources, seeds_by_configuration, provenance_by_configuration


def _normalize(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(records):
        if "configuration" not in row or "metric" not in row:
            raise ValueError(f"Record {index} must contain configuration and metric")
        configuration = _configuration(row["configuration"])
        metric = _metric(row["metric"])
        raw_mean = row.get("mean", row.get("value"))
        if raw_mean in (None, ""):
            raise ValueError(f"Record {index} is missing mean/value")
        mean = finite(raw_mean, f"{configuration}.{metric}.mean")
        lower_raw = row.get("lower", row.get("ci_lower"))
        upper_raw = row.get("upper", row.get("ci_upper"))
        if (lower_raw in (None, "")) != (upper_raw in (None, "")):
            raise ValueError(f"{configuration}.{metric} must provide both lower and upper")
        lower = upper = None
        if lower_raw not in (None, ""):
            lower = finite(lower_raw, f"{configuration}.{metric}.lower")
            upper = finite(upper_raw, f"{configuration}.{metric}.upper")
            if not lower <= mean <= upper:
                raise ValueError(f"Invalid interval ordering for {configuration}.{metric}")
        for name, value in (("mean", mean), ("lower", lower), ("upper", upper)):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{configuration}.{metric}.{name} must lie in [0,1]")
        key = (configuration, metric)
        if key in table:
            raise ValueError(f"Duplicate aggregated record for {configuration}.{metric}")
        table[key] = {
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "n": row.get("n"),
        }
    expected = {(configuration, metric) for configuration in CONFIGURATIONS for metric in METRICS}
    missing = sorted(expected - set(table))
    extra = sorted(set(table) - expected)
    if missing or extra:
        raise ValueError(f"Evidence matrix mismatch; missing={missing}, extra={extra}")
    return table


def _write_demo_input() -> Path:
    demo_dir = FIG_ROOT / "_demo" / "modality_evidence"
    demo_dir.mkdir(parents=True, exist_ok=True)
    path = demo_dir / "DEMO_modality_evidence.json"
    payload = {
        "demo": True,
        "warning": "Synthetic values for layout QA only; never cite as experimental results.",
        "records": [
            {"configuration": "Full", "metric": "AP", "mean": 0.76, "lower": 0.71, "upper": 0.81},
            {"configuration": "Video", "metric": "AP", "mean": 0.61, "lower": 0.55, "upper": 0.68},
            {"configuration": "Full", "metric": "Brier", "mean": 0.16, "lower": 0.13, "upper": 0.19},
            {"configuration": "Video", "metric": "Brier", "mean": 0.24, "lower": 0.20, "upper": 0.28},
            {"configuration": "Full", "metric": "Event mAP", "mean": 0.63, "lower": 0.57, "upper": 0.69},
            {"configuration": "Video", "metric": "Event mAP", "mean": 0.46, "lower": 0.40, "upper": 0.52},
        ],
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def generate(
    *,
    evidence_path: str | Path | None = None,
    aggregate_specs: Sequence[str] = (),
    demo: bool = False,
) -> dict[str, Any]:
    seeds_by_configuration: dict[str, list[int]] = {}
    provenance_by_configuration: dict[str, str] = {}
    if demo:
        evidence_file = _write_demo_input()
        records = _records_from_combined(evidence_file)
        sources = [evidence_file]
    elif evidence_path is not None and aggregate_specs:
        raise ValueError("Use either evidence_path or aggregate_specs, not both")
    elif evidence_path is not None:
        evidence_file = _resolve(evidence_path)
        records = _records_from_combined(evidence_file)
        sources = [evidence_file]
    elif aggregate_specs:
        (
            records,
            sources,
            seeds_by_configuration,
            provenance_by_configuration,
        ) = _records_from_aggregates(aggregate_specs)
    else:
        raise ValueError("Formal rendering requires evidence_path or two aggregate_specs")
    table = _normalize(records)

    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        3,
        figsize=canvas_inches(WIDTH_MM, HEIGHT_MM),
        sharey=True,
    )
    y = np.arange(len(CONFIGURATIONS), dtype=float)
    for metric_index, (axis, metric) in enumerate(zip(axes, METRICS)):
        for row_index, configuration in enumerate(CONFIGURATIONS):
            entry = table[(configuration, metric)]
            mean = entry["mean"]
            xerr = None
            if entry["lower"] is not None:
                xerr = np.asarray([[mean - entry["lower"]], [entry["upper"] - mean]])
            axis.errorbar(
                mean,
                y[row_index],
                xerr=xerr,
                fmt=CONFIG_MARKERS[configuration],
                markersize=4.1,
                markeredgewidth=0,
                color=CONFIG_COLORS[configuration],
                ecolor=CONFIG_COLORS[configuration],
                elinewidth=0.9,
                capsize=2.0,
                zorder=3,
            )
        axis.set_xlim(0, 1)
        axis.set_xticks([0, 0.5, 1.0])
        axis.set_xlabel("Score")
        direction = "↑" if metric != "Brier" else "↓"
        axis.set_title(f"{'abc'[metric_index]}  {metric} {direction}", loc="left", pad=2.0)
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
    y_labels = []
    for configuration in CONFIGURATIONS:
        seeds = seeds_by_configuration.get(configuration)
        seed_text = ",".join(str(seed) for seed in seeds) if seeds else None
        y_labels.append(
            f"{configuration}\n(seed {seed_text})" if seed_text else configuration
        )
    axes[0].set_yticks(y, y_labels)
    axes[0].invert_yaxis()
    for tick, configuration in zip(axes[0].get_yticklabels(), CONFIGURATIONS):
        tick.set_color(CONFIG_COLORS[configuration])
    if demo:
        figure.text(
            0.62,
            0.51,
            "DEMO · SYNTHETIC",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["coral"],
            alpha=0.18,
            rotation=24,
        )
    figure.subplots_adjust(left=0.27, right=0.985, bottom=0.15, top=0.91, wspace=0.24)

    output_name = "fig05_modality_evidence_demo" if demo else "fig05_modality_evidence"
    report = export_and_check(
        figure,
        output_stem=Path(__file__).with_name(output_name),
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
        sources=sources,
        data_summary={
            "demo": demo,
            "configurations": list(CONFIGURATIONS),
            "seeds_by_configuration": seeds_by_configuration,
            "provenance_sha256_by_configuration": provenance_by_configuration,
            "seed_policy": "unmatched seeds are allowed and reported, never relabelled",
            "metrics": list(METRICS),
            "uncertainty_intervals_present": all(
                entry["lower"] is not None for entry in table.values()
            ),
            "plotting_aggregation": "none; reads one upstream aggregate per cell",
        },
    )
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the multimodal evidence comparison figure")
    parser.add_argument("--evidence", help="Combined JSON/CSV evidence matrix")
    parser.add_argument(
        "--aggregate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="aggregate.py output; repeat for Full and Video; seeds may differ",
    )
    parser.add_argument("--demo", action="store_true", help="Render clearly labelled synthetic layout QA data")
    args = parser.parse_args()
    report = generate(
        evidence_path=args.evidence,
        aggregate_specs=args.aggregate,
        demo=args.demo,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
