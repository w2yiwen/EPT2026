#!/usr/bin/env python3
"""Render Table 2 from the formal held-out aggregate artifacts.

The script performs no metric recomputation. It validates that every model uses
the same seed set, evaluation protocol, and held-out data counts, then mirrors
the upstream aggregate means (and sample standard deviations when n > 1) into
LaTeX, CSV, and Markdown outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_ORDER = ("LSTR", "GateHUB", "TeSTra", "EPT-Net (ours)")
DEFAULT_AGGREGATES = {
    "LSTR": "results/dual4090_seed42_3way/lstr_marlin11_eeg_ppg_video_no_text/aggregate_seed_42.json",
    "GateHUB": "results/dual4090_seed42_3way/gatehub_marlin11_eeg_ppg_video_no_text/aggregate_seed_42.json",
    "TeSTra": "results/dual4090_seed42_3way/testra_marlin11_eeg_ppg_video_no_text/aggregate_seed_42.json",
    "EPT-Net (ours)": "results/dual4090_seed42_3way/eptnet_marlin11_eeg_ppg_video_no_text/aggregate_seed_42.json",
}
METRICS = (
    ("frame.auprc", "AUPRC", True),
    ("event.event_ap_iou_0.3", "AP@0.3", True),
    ("event.event_ap_iou_0.5", "AP@0.5", True),
    ("event.event_ap_iou_0.7", "AP@0.7", True),
    ("event.event_f1_iou_0.5", "Event-F1@0.5", True),
    ("latency.mean_detection_delay_seconds", "Delay (s)", False),
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_specs(specs: Sequence[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    raw = specs or [f"{model}={path}" for model, path in DEFAULT_AGGREGATES.items()]
    for spec in raw:
        if "=" not in spec:
            raise ValueError(f"Aggregate specification must use MODEL=PATH: {spec!r}")
        model, value = spec.split("=", 1)
        model = model.strip()
        if model not in MODEL_ORDER:
            raise ValueError(f"Unknown model {model!r}; expected one of {MODEL_ORDER}")
        if model in mapping:
            raise ValueError(f"Duplicate aggregate for {model}")
        path = Path(value.strip())
        path = path if path.is_absolute() else project_root() / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mapping[model] = path
    missing = [model for model in MODEL_ORDER if model not in mapping]
    if missing:
        raise ValueError(f"Missing aggregates for: {', '.join(missing)}")
    return mapping


def metric_entry(payload: Mapping[str, Any], metric: str) -> tuple[float, float | None]:
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, Mapping) and isinstance(aggregate.get(metric), Mapping):
        entry = aggregate[metric]
        mean = float(entry["mean"])
        std_raw = entry.get("sample_std")
        std = None if std_raw is None else float(std_raw)
        return mean, std
    runs = payload.get("runs")
    if isinstance(runs, list) and len(runs) == 1:
        metrics = runs[0].get("metrics") if isinstance(runs[0], Mapping) else None
        if isinstance(metrics, Mapping) and metric in metrics:
            return float(metrics[metric]), None
    raise KeyError(f"Aggregate metric is missing: {metric}")


def load_matrix(paths: Mapping[str, Path]) -> tuple[dict[str, dict[str, tuple[float, float | None]]], list[int], dict[str, Any]]:
    matrix: dict[str, dict[str, tuple[float, float | None]]] = {}
    reference_seeds: list[int] | None = None
    reference_signature: dict[str, Any] | None = None
    for model in MODEL_ORDER:
        payload = json.loads(paths[model].read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError(f"Aggregate root must be an object: {paths[model]}")
        seeds = payload.get("seeds")
        if not isinstance(seeds, list) or not seeds or not all(isinstance(x, int) and not isinstance(x, bool) for x in seeds):
            raise ValueError(f"Invalid seed metadata in {paths[model]}")
        signature = {
            "protocol": payload.get("protocol"),
            "data": payload.get("data"),
            "calibration_protocol": payload.get("calibration_protocol"),
        }
        if any(value is None for value in signature.values()):
            raise ValueError(f"Missing protocol/data metadata in {paths[model]}")
        if reference_seeds is None:
            reference_seeds = list(seeds)
            reference_signature = signature
        elif list(seeds) != reference_seeds:
            raise ValueError(f"Models do not share a matched seed set: {model} has {seeds}, expected {reference_seeds}")
        elif signature != reference_signature:
            raise ValueError(f"Models do not share the same held-out protocol/data: {model}")
        matrix[model] = {metric: metric_entry(payload, metric) for metric, _, _ in METRICS}
    assert reference_seeds is not None and reference_signature is not None
    return matrix, reference_seeds, reference_signature


def winners(matrix: Mapping[str, Mapping[str, tuple[float, float | None]]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for metric, _, higher in METRICS:
        values = {model: matrix[model][metric][0] for model in MODEL_ORDER}
        result[metric] = (max if higher else min)(values, key=values.get)
    return result


def numeric_text(value: float, std: float | None, n: int) -> str:
    if n > 1:
        if std is None:
            raise ValueError("Multi-run aggregate is missing sample_std")
        return f"{value:.3f} ± {std:.3f}"
    return f"{value:.3f}"


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def render(paths: Mapping[str, Path], output_dir: Path) -> dict[str, Any]:
    matrix, seeds, signature = load_matrix(paths)
    best = winners(matrix)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "table02_performance.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", *[label for _, label, _ in METRICS]])
        for model in MODEL_ORDER:
            writer.writerow([model, *[numeric_text(*matrix[model][metric], len(seeds)) for metric, _, _ in METRICS]])

    header = "| Method | " + " | ".join(f"{label} {'↑' if higher else '↓'}" for _, label, higher in METRICS) + " |"
    alignment = "|:--|" + "|".join(":--:" for _ in METRICS) + "|"
    markdown_rows = [header, alignment]
    for model in MODEL_ORDER:
        values = []
        for metric, _, _ in METRICS:
            text = numeric_text(*matrix[model][metric], len(seeds))
            values.append(f"**{text}**" if best[metric] == model else text)
        label = "**EPT-Net (ours)**" if model == "EPT-Net (ours)" else model
        markdown_rows.append(f"| {label} | " + " | ".join(values) + " |")
    note = (
        f"Mean ± sample standard deviation across {len(seeds)} matched seeds ({', '.join(map(str, seeds))})."
        if len(seeds) > 1
        else f"Single-run result (seed {seeds[0]}); between-run uncertainty is not available."
    )
    md_path = output_dir / "table02_performance.md"
    md_path.write_text("\n".join(markdown_rows) + f"\n\n{note}\n", encoding="utf-8")

    column_header = " & ".join(
        ["Method", *[f"{latex_escape(label)} $\\{'uparrow' if higher else 'downarrow'}$" for _, label, higher in METRICS]]
    ) + r" \\" 
    latex_rows = []
    for model in MODEL_ORDER:
        cells = []
        for metric, _, _ in METRICS:
            text = numeric_text(*matrix[model][metric], len(seeds)).replace("±", r"$\pm$")
            cells.append(r"\textbf{" + text + "}" if best[metric] == model else text)
        model_label = r"\textbf{EPT-Net (ours)}" if model == "EPT-Net (ours)" else latex_escape(model)
        latex_rows.append(model_label + " & " + " & ".join(cells) + " \\\\")
    tex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\caption{Performance comparison on the held-out test set. Best values are bolded independently for each metric.}",
            r"\label{tab:main-performance}",
            r"\centering",
            r"\setlength{\tabcolsep}{4.2pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            column_header,
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}",
            r"\parbox{0.98\textwidth}{\footnotesize " + latex_escape(note) + r" Event-F1 uses one-to-one matching at tIoU $\geq 0.5$; the confidence threshold is selected on the validation set.}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = output_dir / "table02_performance.tex"
    tex_path.write_text(tex, encoding="utf-8")

    manifest = {
        "table_id": "table02_performance",
        "generated_files": [p.name for p in (tex_path, csv_path, md_path)],
        "models": list(MODEL_ORDER),
        "seeds": seeds,
        "uncertainty": "sample standard deviation across matched seeds" if len(seeds) > 1 else "unavailable for a single run",
        "best_by_metric": best,
        "sources": {model: {"path": str(path.relative_to(project_root())), "sha256": sha256(path)} for model, path in paths.items()},
        "protocol": signature,
        "checks": {"matched_seeds": "pass", "matched_protocol_and_data": "pass", "no_recomputation": "pass", "write_boundary": "pass"},
    }
    manifest_path = output_dir / "table02_performance.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def generate(*, aggregate_specs: Sequence[str] = ()) -> dict[str, Any]:
    return render(parse_specs(aggregate_specs), Path(__file__).resolve().parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", action="append", default=[], metavar="MODEL=PATH")
    args = parser.parse_args()
    report = generate(aggregate_specs=args.aggregate)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
