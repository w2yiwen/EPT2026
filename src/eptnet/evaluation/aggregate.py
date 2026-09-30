from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping
from glob import glob
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from eptnet.provenance import portable_path

METRIC_SECTIONS = (
    "frame",
    "boundary",
    "event",
    "early_detection",
    "latency",
    "efficiency",
    "subject_macro",
    "fixed_threshold_0_5_sensitivity",
)


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            flattened[prefix] = number
    return flattened


def _metric_values(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for section in METRIC_SECTIONS:
        if section in payload:
            metrics.update(_flatten_numeric(payload[section], section))
    return metrics


def _run_identity(payload: Mapping[str, Any], path: Path) -> tuple[str, int, str, str]:
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{path} is missing checkpoint metadata")
    experiment = checkpoint.get("experiment")
    seed = checkpoint.get("seed")
    fingerprint = checkpoint.get("training_config_sha256")
    if not isinstance(experiment, str) or not experiment:
        raise ValueError(f"{path} has an invalid checkpoint experiment name")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{path} has an invalid checkpoint seed")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(f"{path} has an invalid training configuration fingerprint")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{path} is missing provenance metadata")
    provenance_sha256 = provenance.get("provenance_sha256")
    if not isinstance(provenance_sha256, str) or len(provenance_sha256) != 64:
        raise ValueError(f"{path} has an invalid provenance fingerprint")
    if checkpoint.get("provenance_sha256") != provenance_sha256:
        raise ValueError(f"{path} checkpoint and evaluation provenance disagree")
    return experiment, seed, fingerprint, provenance_sha256


def _comparison_signature(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    protocol = payload.get("protocol")
    data = payload.get("data")
    if not isinstance(protocol, Mapping) or not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain protocol and data metadata")
    # A validation-calibrated frame threshold may legitimately differ by seed.
    protocol_signature = {key: value for key, value in protocol.items() if key != "frame_threshold"}
    data_signature = {
        key: data.get(key)
        for key in (
            "num_sequences",
            "num_unique_steps",
            "num_evaluated_target_steps",
            "num_target_events",
            "num_dense_event_proposals",
        )
    }
    calibration = payload.get("calibration")
    if calibration is None:
        calibration_signature: dict[str, Any] = {
            "enabled": False,
            "method": "fixed_threshold",
            "selection_split": None,
            "calibration_min_threshold": None,
            "calibration_max_threshold": None,
            "calibration_steps": None,
            "frame_threshold": protocol.get("frame_threshold"),
        }
    elif isinstance(calibration, Mapping):
        # Selected thresholds and validation scores vary by seed; the selection
        # procedure itself must remain identical across runs.
        calibration_signature = {
            "enabled": bool(calibration.get("enabled", True)),
            "method": calibration.get("method", "validation_subject_macro_f1_grid"),
            "selection_split": calibration.get("selection_split"),
            "calibration_min_threshold": calibration.get("calibration_min_threshold"),
            "calibration_max_threshold": calibration.get("calibration_max_threshold"),
            "calibration_steps": calibration.get("calibration_steps"),
        }
        if not calibration_signature["enabled"]:
            calibration_signature["frame_threshold"] = calibration.get(
                "frame_threshold", protocol.get("frame_threshold")
            )
    else:
        raise ValueError(f"{path} has invalid calibration metadata")
    return {
        "protocol": protocol_signature,
        "data": data_signature,
        "calibration": calibration_signature,
    }


def aggregate_result_files(paths: Iterable[Path]) -> dict[str, Any]:
    records = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        experiment, seed, fingerprint, provenance_sha256 = _run_identity(payload, path)
        metrics = _metric_values(payload)
        if not metrics:
            raise ValueError(f"{path} contains no numeric evaluation metrics")
        records.append(
            {
                "path": portable_path(path),
                "experiment": experiment,
                "seed": seed,
                "training_config_sha256": fingerprint,
                "provenance_sha256": provenance_sha256,
                "signature": _comparison_signature(payload, path),
                "metrics": metrics,
            }
        )
    if not records:
        raise ValueError("No result JSON files were provided")

    experiments = {record["experiment"] for record in records}
    if len(experiments) != 1:
        raise ValueError(f"Cannot aggregate different experiments: {sorted(experiments)}")
    fingerprints = {record["training_config_sha256"] for record in records}
    if len(fingerprints) != 1:
        raise ValueError("Cannot aggregate runs trained with different configurations")
    provenance_fingerprints = {record["provenance_sha256"] for record in records}
    if len(provenance_fingerprints) != 1:
        raise ValueError("Cannot aggregate runs with different source or prepared-data provenance")
    seeds = [record["seed"] for record in records]
    duplicate_seeds = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if duplicate_seeds:
        raise ValueError(f"Duplicate evaluation seeds are not independent runs: {duplicate_seeds}")
    reference_signature = records[0]["signature"]
    incompatible = [
        record["path"] for record in records[1:] if record["signature"] != reference_signature
    ]
    if incompatible:
        raise ValueError(
            "Evaluation protocol/data mismatch across runs: " + ", ".join(incompatible)
        )

    metric_keys = set().union(*(set(record["metrics"]) for record in records))
    aggregate = {}
    incomplete_metrics = []
    for key in sorted(metric_keys):
        values = [record["metrics"][key] for record in records if key in record["metrics"]]
        missing = len(records) - len(values)
        aggregate[key] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "n": len(values),
            "missing": missing,
        }
        if missing:
            incomplete_metrics.append(key)
    for record in records:
        record.pop("signature")
    return {
        "experiment": next(iter(experiments)),
        "num_runs": len(records),
        "seeds": sorted(seeds),
        "training_config_sha256": next(iter(fingerprints)),
        "provenance_sha256": next(iter(provenance_fingerprints)),
        "protocol": reference_signature["protocol"],
        "data": reference_signature["data"],
        "calibration_protocol": reference_signature["calibration"],
        "incomplete_metrics": incomplete_metrics,
        "runs": records,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated EPT-Net evaluations")
    parser.add_argument("inputs", nargs="+", help="Evaluation JSON paths or glob patterns")
    parser.add_argument("--output", required=True, help="Aggregate JSON output path")
    parser.add_argument("--csv", help="Optional flat CSV output path")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the JSON or CSV output already exists",
    )
    args = parser.parse_args()

    paths = []
    for pattern in args.inputs:
        candidate = Path(pattern)
        if candidate.is_file():
            paths.append(candidate)
        else:
            paths.extend(Path(match) for match in glob(pattern, recursive=True))
    # Preserve one record per concrete file even if glob patterns overlap.
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    summary = aggregate_result_files(paths)

    output = Path(args.output)
    csv_path = Path(args.csv) if args.csv else None
    if args.no_overwrite:
        existing = [
            path
            for path in (output, csv_path)
            if path is not None and (path.exists() or path.is_symlink())
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite aggregate output(s): "
                + ", ".join(str(path) for path in existing)
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_overwrite:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
            handle.write("\n")
    else:
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
            newline="\n",
        )
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "x" if args.no_overwrite else "w"
        with csv_path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["metric", "mean", "std", "min", "max", "n", "missing"],
                lineterminator="\n",
            )
            writer.writeheader()
            for metric, values in summary["aggregate"].items():
                writer.writerow({"metric": metric, **values})
    print(json.dumps({"num_runs": summary["num_runs"], "output": str(output)}))


if __name__ == "__main__":
    main()
