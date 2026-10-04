#!/usr/bin/env python3
"""Generate Table 2 from measured test metrics with protocol checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN_ROOT = Path("results/dual4090_seed42_3way")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    experiment: str
    group: str


MODEL_SPECS = (
    ModelSpec("LSTR", "lstr_marlin11_eeg_ppg_video_no_text", "main"),
    ModelSpec("GateHUB", "gatehub_marlin11_eeg_ppg_video_no_text", "main"),
    ModelSpec("TeSTra", "testra_marlin11_eeg_ppg_video_no_text", "main"),
    ModelSpec("EPT-Net (ours)", "eptnet_marlin11_eeg_ppg_video_no_text", "main"),
    ModelSpec("Fixed reader", "eptnet_marlin11_fixed_reader_no_text", "diagnostic"),
    ModelSpec("No persistent state", "eptnet_marlin11_no_persistent_no_text", "diagnostic"),
)

METRICS = (
    ("auprc", ("frame", "auprc"), "AUPRC $\\uparrow$"),
    ("ap03", ("event", "event_ap_iou_0.3"), "AP@0.3 $\\uparrow$"),
    ("ap05", ("event", "event_ap_iou_0.5"), "AP@0.5 $\\uparrow$"),
    ("ap07", ("event", "event_ap_iou_0.7"), "AP@0.7 $\\uparrow$"),
    ("event_f1_05", ("event", "event_f1_iou_0.5"), "Event-F1@0.5 $\\uparrow$"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _nested(payload: dict[str, Any], keys: tuple[str, ...], location: str) -> float:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing {'.'.join(keys)} in {location}")
        node = node[key]
    value = float(node)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{location}:{'.'.join(keys)} must be finite and in [0, 1]")
    return value


def _protocol_signature(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    protocol = payload.get("protocol")
    data = payload.get("data")
    if not isinstance(protocol, dict) or not isinstance(data, dict):
        raise ValueError(f"Missing protocol/data object in {path}")
    event_ap = protocol.get("event_ap_protocol")
    event_f1 = protocol.get("event_f1_protocol")
    if not isinstance(event_ap, dict) or event_ap.get("score_threshold") is not None:
        raise ValueError(f"Event AP must be threshold independent in {path}")
    if not isinstance(event_f1, dict) or event_f1.get("uses_frame_threshold") is not True:
        raise ValueError(f"Event F1 must use the validation-selected frame threshold in {path}")
    return {
        "evaluation_unit": protocol.get("evaluation_unit"),
        "positive_class": protocol.get("positive_class"),
        "class_encoding": protocol.get("class_encoding"),
        "boundary_threshold": protocol.get("boundary_threshold"),
        "boundary_tolerance_steps": protocol.get("boundary_tolerance_steps"),
        "event_iou_thresholds": protocol.get("event_iou_thresholds"),
        "target_mask_definition": protocol.get("target_mask_definition"),
        "event_ap_protocol": event_ap,
        "data": {
            key: data.get(key)
            for key in (
                "num_sequences",
                "num_unique_steps",
                "num_evaluated_target_steps",
                "num_target_events",
                "num_dense_event_proposals",
            )
        },
    }


def _seed_from_dir(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Run directory must be named seed_<integer>: {path}") from exc


def _collect(campaign_root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[Path], str]:
    records: dict[str, list[dict[str, Any]]] = {}
    sources: list[Path] = []
    reference_protocol: dict[str, Any] | None = None
    reference_provenance: str | None = None
    reference_seeds: tuple[int, ...] | None = None

    for spec in MODEL_SPECS:
        model_records: list[dict[str, Any]] = []
        for run_dir in sorted((campaign_root / spec.experiment).glob("seed_*")):
            metrics_path = run_dir / "test_metrics.json"
            if not metrics_path.is_file():
                continue
            payload = _read_json(metrics_path)
            checkpoint = payload.get("checkpoint")
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Missing checkpoint metadata in {metrics_path}")
            if checkpoint.get("experiment") != spec.experiment:
                raise ValueError(f"Experiment identity mismatch in {metrics_path}")
            seed = _seed_from_dir(run_dir)
            if checkpoint.get("seed") != seed:
                raise ValueError(f"Seed metadata mismatch in {metrics_path}")
            provenance = checkpoint.get("provenance_sha256")
            if not isinstance(provenance, str) or len(provenance) != 64:
                raise ValueError(f"Missing checkpoint provenance in {metrics_path}")
            if reference_provenance is None:
                reference_provenance = provenance
            elif provenance != reference_provenance:
                raise ValueError(
                    f"All rows must share one provenance fingerprint; {metrics_path} differs"
                )
            signature = _protocol_signature(payload, metrics_path)
            if reference_protocol is None:
                reference_protocol = signature
            elif signature != reference_protocol:
                raise ValueError(f"Protocol or evaluation data mismatch in {metrics_path}")
            values = {
                key: _nested(payload, path, str(metrics_path))
                for key, path, _ in METRICS
            }
            model_records.append({"seed": seed, "path": metrics_path, "values": values})
            sources.append(metrics_path)
        if not model_records:
            raise FileNotFoundError(
                f"No completed test_metrics.json runs for {spec.label} under "
                f"{campaign_root / spec.experiment}"
            )
        seeds = tuple(sorted(record["seed"] for record in model_records))
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seeds for {spec.label}: {seeds}")
        if reference_seeds is None:
            reference_seeds = seeds
        elif seeds != reference_seeds:
            raise ValueError(
                f"Every main/diagnostic row must use the same seed set; "
                f"{spec.label} has {seeds}, expected {reference_seeds}"
            )
        records[spec.label] = model_records

    assert reference_provenance is not None
    return records, sources, reference_provenance


def _summarize(records: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for spec in MODEL_SPECS:
        runs = records[spec.label]
        row: dict[str, Any] = {
            "group": spec.group,
            "n": len(runs),
            "seeds": sorted(run["seed"] for run in runs),
        }
        for key, _, _ in METRICS:
            values = [run["values"][key] for run in runs]
            row[key] = {
                "mean": mean(values),
                "sample_std": stdev(values) if len(values) > 1 else None,
            }
        summary[spec.label] = row
    return summary


def _display(metric: dict[str, float | None]) -> str:
    central = 100.0 * float(metric["mean"])
    spread = metric["sample_std"]
    if spread is None:
        return f"{central:.2f}"
    return f"{central:.2f} ± {100.0 * float(spread):.2f}"


def _main_best(summary: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        key: max(
            float(summary[spec.label][key]["mean"])
            for spec in MODEL_SPECS
            if spec.group == "main"
        )
        for key, _, _ in METRICS
    }


def _markdown(summary: dict[str, dict[str, Any]], best: dict[str, float]) -> str:
    headings = ["Method", "AUPRC ↑", "AP@0.3 ↑", "AP@0.5 ↑", "AP@0.7 ↑", "Event-F1@0.5 ↑"]
    lines = [
        "*Table 2. Performance comparison on the fixed test evaluation view (%).*",
        "",
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join(["---"] + ["---:"] * 5) + " |",
    ]
    for spec in MODEL_SPECS:
        if spec.label == "Fixed reader":
            lines.append("| *Mechanism diagnostics* |  |  |  |  |  |")
        label = f"**{spec.label}**" if spec.label == "EPT-Net (ours)" else spec.label
        cells = [label]
        for key, _, _ in METRICS:
            value = _display(summary[spec.label][key])
            is_best = spec.group == "main" and math.isclose(
                float(summary[spec.label][key]["mean"]), best[key], rel_tol=0.0, abs_tol=1e-12
            )
            cells.append(f"**{value}**" if is_best else value)
        lines.append("| " + " | ".join(cells) + " |")
    seeds = summary[MODEL_SPECS[0].label]["seeds"]
    n = summary[MODEL_SPECS[0].label]["n"]
    uncertainty = (
        "Values are mean ± sample standard deviation across seeds."
        if n > 1
        else "One run is available, so values are point estimates without a standard deviation."
    )
    lines.extend(
        [
            "",
            f"Seeds: {', '.join(str(seed) for seed in seeds)} (n={n}). {uncertainty}",
            "Bold marks the best value within the four-model main comparison only. ",
            "Fixed reader and No persistent state are mechanism diagnostics, not additional main baselines. ",
            "The project's validation and test manifests are fixed training-included evaluation views, not held-out splits.",
            "",
        ]
    )
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def _latex(summary: dict[str, dict[str, Any]], best: dict[str, float]) -> str:
    rows = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Performance comparison on the fixed test evaluation view (\%).}",
        r"\label{tab:overall-performance}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & AUPRC $\uparrow$ & AP@0.3 $\uparrow$ & AP@0.5 $\uparrow$ & AP@0.7 $\uparrow$ & Event-F1@0.5 $\uparrow$ \\",
        r"\midrule",
    ]
    for spec in MODEL_SPECS:
        if spec.label == "Fixed reader":
            rows.extend([r"\midrule", r"\multicolumn{6}{l}{\textit{Mechanism diagnostics}} \\"])
        label = _latex_escape(spec.label)
        if spec.label == "EPT-Net (ours)":
            label = r"\textbf{" + label + "}"
        cells = [label]
        for key, _, _ in METRICS:
            value = _display(summary[spec.label][key]).replace("±", r"$\pm$")
            is_best = spec.group == "main" and math.isclose(
                float(summary[spec.label][key]["mean"]), best[key], rel_tol=0.0, abs_tol=1e-12
            )
            cells.append((r"\textbf{" + value + "}") if is_best else value)
        rows.append(" & ".join(cells) + r" \\")
    first = summary[MODEL_SPECS[0].label]
    seeds = ", ".join(str(seed) for seed in first["seeds"])
    uncertainty = (
        r"Values are mean $\pm$ sample standard deviation across seeds."
        if first["n"] > 1
        else "One run is available; values are point estimates without a standard deviation."
    )
    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\textwidth}\footnotesize",
            f"Seeds: {seeds} ($n={first['n']}$). {uncertainty} "
            "Bold marks the best result within the four-model main comparison. "
            "Diagnostic variants are shown separately. The validation/test manifests are "
            "fixed training-included evaluation views, not held-out splits.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(rows)


def generate(*, campaign_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(campaign_root) if campaign_root is not None else DEFAULT_CAMPAIGN_ROOT
    root = root if root.is_absolute() else PROJECT_ROOT / root
    root = root.resolve()
    records, sources, provenance = _collect(root)
    summary = _summarize(records)
    best = _main_best(summary)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    markdown_path = TABLE_DIR / "table02_overall_performance.md"
    latex_path = TABLE_DIR / "table02_overall_performance.tex"
    csv_path = TABLE_DIR / "table02_overall_performance.csv"
    manifest_path = TABLE_DIR / "table02_overall_performance.manifest.json"
    markdown_path.write_text(_markdown(summary, best), encoding="utf-8")
    latex_path.write_text(_latex(summary, best), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["group", "method", "n", "seeds"]
        for key, _, _ in METRICS:
            header.extend([f"{key}_mean", f"{key}_sample_std"])
        writer.writerow(header)
        for spec in MODEL_SPECS:
            row = summary[spec.label]
            values: list[Any] = [spec.group, spec.label, row["n"], ";".join(map(str, row["seeds"]))]
            for key, _, _ in METRICS:
                values.extend([row[key]["mean"], row[key]["sample_std"]])
            writer.writerow(values)

    manifest = {
        "table_id": "table02_overall_performance",
        "campaign_root": str(root.relative_to(PROJECT_ROOT)),
        "display_scale": "percentage (100 x stored proportion)",
        "groups": {
            "main_comparison": [spec.label for spec in MODEL_SPECS if spec.group == "main"],
            "mechanism_diagnostics": [
                spec.label for spec in MODEL_SPECS if spec.group == "diagnostic"
            ],
        },
        "summary": summary,
        "main_comparison_best": best,
        "provenance_sha256": provenance,
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in sources
        },
        "checks": {
            "files": "pass",
            "protocol_match": "pass",
            "evaluation_scope_match": "pass",
            "provenance_match": "pass",
            "seed_match": "pass",
            "missing_metrics": "pass",
            "held_out_claim": "not-applicable; explicitly labelled training-included evaluation view",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "outputs": [str(path.relative_to(PROJECT_ROOT)) for path in (markdown_path, latex_path, csv_path, manifest_path)],
        "seeds": summary[MODEL_SPECS[0].label]["seeds"],
        "seed_count": summary[MODEL_SPECS[0].label]["n"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        default=str(DEFAULT_CAMPAIGN_ROOT),
        help="Root containing the protocol-matched experiment directories",
    )
    args = parser.parse_args()
    print(json.dumps(generate(campaign_root=args.campaign_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
