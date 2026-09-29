"""Read-only identity and structure checks for reusable aligned11 results."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPTNET_SOURCE = PROJECT_ROOT / "src" / "eptnet"
CLASS_ENCODING = {"0": "deception", "1": "truth"}
POSITIVE_CLASS = 0
CSV_FIELDS = ("metric", "mean", "std", "min", "max", "n", "missing")


def _load_lightweight_module(name: str):
    """Load non-model EPT modules without executing ``eptnet.__init__``."""

    package_name = "_eptnet_shortpaper_verifier"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(EPTNET_SOURCE)]
        sys.modules[package_name] = package
    qualified = f"{package_name}.{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    source = EPTNET_SOURCE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(qualified, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


provenance_module = _load_lightweight_module("provenance")
config_module = _load_lightweight_module("config")
aggregate_module = _load_lightweight_module("aggregate")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _require_regular_file(value: str | Path, label: str) -> Path:
    path = _resolve(value)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def _read_json_object(value: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = _require_regular_file(value, label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return path, payload


def _sha256_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 identity")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value.lower()


def _preflight_identity(value: str | Path) -> tuple[Path, dict[str, Any]]:
    path, payload = _read_json_object(value, "preflight record")
    if payload.get("status") != "passed" or payload.get("warnings") not in ([], None):
        raise ValueError("Reusable results require a strict warning-free preflight record")
    if payload.get("positive_class") != POSITIVE_CLASS:
        raise ValueError("Preflight positive_class is not 0")
    if payload.get("label_mapping") != CLASS_ENCODING:
        raise ValueError("Preflight class encoding is not 0=deception, 1=truth")
    runtime = payload.get("runtime_provenance")
    if not isinstance(runtime, Mapping):
        raise ValueError("Preflight record lacks strict runtime_provenance")
    _sha256_identity(runtime.get("source_tree_sha256"), "preflight source_tree_sha256")
    _sha256_identity(runtime.get("provenance_sha256"), "preflight provenance_sha256")
    return path, payload


def _required_count(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"metrics.data.{name} must be a non-negative integer")
    return value


def _finite_probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and lie in [0, 1]")
    return number


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _verify_predictions(path: Path, data: Mapping[str, Any]) -> dict[str, int]:
    row_count = 0
    target_valid_count = 0
    sample_order: list[str] = []
    closed_samples: set[str] = set()
    previous_sample: str | None = None
    previous_row_index: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank prediction record: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"Prediction record is not an object: {path}:{line_number}")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Invalid prediction sample_id: {path}:{line_number}")
            row_index = row.get("row_index")
            if isinstance(row_index, bool) or not isinstance(row_index, int):
                raise ValueError(f"Invalid prediction row_index: {path}:{line_number}")
            if sample_id != previous_sample:
                if previous_sample is not None:
                    closed_samples.add(previous_sample)
                if sample_id in closed_samples:
                    raise ValueError(f"Prediction sample is not contiguous: {sample_id}")
                sample_order.append(sample_id)
                previous_sample = sample_id
                previous_row_index = None
            if previous_row_index is not None and row_index <= previous_row_index:
                raise ValueError(f"Prediction row_index is not increasing for {sample_id}")
            previous_row_index = row_index
            if row.get("positive_class") != POSITIVE_CLASS:
                raise ValueError(f"Prediction positive_class is not 0: {path}:{line_number}")
            _finite_probability(
                row.get("positive_probability"),
                f"prediction positive_probability at {path}:{line_number}",
            )
            target_valid = row.get("target_valid")
            if not isinstance(target_valid, bool):
                raise ValueError(f"Prediction target_valid is not boolean: {path}:{line_number}")
            label = row.get("label")
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError(f"Prediction label is not 0/1: {path}:{line_number}")
            class_probabilities = row.get("class_probabilities")
            if not isinstance(class_probabilities, list) or len(class_probabilities) != 2:
                raise ValueError(
                    f"Prediction class_probabilities must contain two values: {path}:{line_number}"
                )
            probabilities = [
                _finite_probability(value, f"class probability at {path}:{line_number}")
                for value in class_probabilities
            ]
            if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Prediction class probabilities do not sum to one: {path}:{line_number}")
            if not math.isclose(
                probabilities[POSITIVE_CLASS],
                float(row["positive_probability"]),
                rel_tol=1e-7,
                abs_tol=1e-7,
            ):
                raise ValueError(f"Prediction positive probability has the wrong class: {path}:{line_number}")
            boundary_probabilities = row.get("boundary_probabilities")
            if not isinstance(boundary_probabilities, list) or len(boundary_probabilities) != 2:
                raise ValueError(
                    "Prediction boundary_probabilities must contain two values: "
                    f"{path}:{line_number}"
                )
            for value in boundary_probabilities:
                _finite_probability(
                    value, f"boundary probability at {path}:{line_number}"
                )
            offsets = row.get("offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(
                    f"Prediction offsets must contain two values: {path}:{line_number}"
                )
            for value in offsets:
                _finite_nonnegative(value, f"offset at {path}:{line_number}")
            row_count += 1
            target_valid_count += int(target_valid)

    if row_count == 0:
        raise ValueError(f"Predictions file is empty: {path}")
    expected_rows = _required_count(data, "num_unique_steps")
    expected_valid = _required_count(data, "num_evaluated_target_steps")
    expected_sequences = _required_count(data, "num_sequences")
    if row_count != expected_rows:
        raise ValueError(f"Prediction row count {row_count} != metrics {expected_rows}")
    if target_valid_count != expected_valid:
        raise ValueError(
            f"Prediction target-valid count {target_valid_count} != metrics {expected_valid}"
        )
    if len(sample_order) != expected_sequences:
        raise ValueError(
            f"Prediction sequence count {len(sample_order)} != metrics {expected_sequences}"
        )
    return {
        "rows": row_count,
        "target_valid_rows": target_valid_count,
        "sequences": len(sample_order),
    }


def _verify_event_list(events: Any, label: str) -> int:
    if not isinstance(events, list):
        raise ValueError(f"{label} must be a list")
    count = 0
    for sequence_index, sequence in enumerate(events):
        if not isinstance(sequence, list):
            raise ValueError(f"{label}[{sequence_index}] must be a list")
        for event_index, event in enumerate(sequence):
            if not isinstance(event, Mapping):
                raise ValueError(f"{label}[{sequence_index}][{event_index}] must be an object")
            start = float(event.get("start", math.nan))
            end = float(event.get("end", math.nan))
            if not math.isfinite(start) or not math.isfinite(end) or start > end:
                raise ValueError(f"Invalid interval in {label}[{sequence_index}][{event_index}]")
            count += 1
    return count


def _verify_events(path: Path, data: Mapping[str, Any]) -> dict[str, int]:
    _, payload = _read_json_object(path, "decoded events")
    predictions = payload.get("predictions")
    targets = payload.get("targets")
    predicted_count = _verify_event_list(predictions, "events.predictions")
    target_count = _verify_event_list(targets, "events.targets")
    expected_sequences = _required_count(data, "num_sequences")
    if len(predictions) != expected_sequences or len(targets) != expected_sequences:
        raise ValueError("Event sequence counts do not match metrics.data.num_sequences")
    expected_predicted = _required_count(data, "num_predicted_events")
    expected_targets = _required_count(data, "num_target_events")
    if predicted_count != expected_predicted:
        raise ValueError(
            f"Decoded event count {predicted_count} != metrics {expected_predicted}"
        )
    if target_count != expected_targets:
        raise ValueError(f"Target event count {target_count} != metrics {expected_targets}")
    return {
        "predicted_events": predicted_count,
        "target_events": target_count,
        "sequences": expected_sequences,
    }


def verify_completed_run(
    *,
    config_path: str | Path,
    seed: int,
    metrics_path: str | Path,
    predictions_path: str | Path,
    events_path: str | Path,
    preflight_path: str | Path,
) -> dict[str, Any]:
    """Verify that a completed run belongs to the current frozen execution identity."""

    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    config_file = _require_regular_file(config_path, "configuration")
    metrics_file, metrics = _read_json_object(metrics_path, "test metrics")
    predictions_file = _require_regular_file(predictions_path, "test predictions")
    events_file = _require_regular_file(events_path, "test events")
    if len({metrics_file.parent, predictions_file.parent, events_file.parent}) != 1:
        raise ValueError("Metrics, predictions, and events must share one seed directory")
    preflight_file, preflight = _preflight_identity(preflight_path)
    runtime = preflight["runtime_provenance"]
    expected_provenance = _sha256_identity(
        runtime.get("provenance_sha256"), "preflight provenance_sha256"
    )

    config = config_module.load_config(str(config_file))
    if int(config["evaluation"]["positive_class"]) != POSITIVE_CLASS:
        raise ValueError("Configuration positive_class is not 0")
    expected_experiment = str(config["experiment"]["name"])
    expected_training = provenance_module.training_config_fingerprint(config)

    protocol = metrics.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Metrics are missing protocol metadata")
    if protocol.get("positive_class") != POSITIVE_CLASS:
        raise ValueError("Metrics protocol positive_class is not 0")
    if protocol.get("class_encoding") != CLASS_ENCODING:
        raise ValueError("Metrics protocol class encoding is not 0=deception, 1=truth")
    checkpoint = metrics.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Metrics are missing checkpoint metadata")
    if checkpoint.get("experiment") != expected_experiment:
        raise ValueError("Completed run experiment does not match the current configuration")
    if checkpoint.get("seed") != seed:
        raise ValueError("Completed run seed does not match the requested seed")
    if checkpoint.get("training_config_sha256") != expected_training:
        raise ValueError("Completed run training configuration fingerprint is stale")
    checkpoint_provenance = _sha256_identity(
        checkpoint.get("provenance_sha256"), "checkpoint provenance_sha256"
    )
    metrics_provenance = metrics.get("provenance")
    if not isinstance(metrics_provenance, Mapping):
        raise ValueError("Metrics are missing runtime provenance")
    metrics_identity = _sha256_identity(
        metrics_provenance.get("provenance_sha256"), "metrics provenance_sha256"
    )
    if checkpoint_provenance != expected_provenance or metrics_identity != expected_provenance:
        raise ValueError("Completed run provenance does not match the current strict preflight")
    source_tree = metrics_provenance.get("source_tree")
    if not isinstance(source_tree, Mapping) or source_tree.get("sha256") != runtime.get(
        "source_tree_sha256"
    ):
        raise ValueError("Completed run source-tree identity does not match preflight")

    data = metrics.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Metrics are missing data counts")
    prediction_summary = _verify_predictions(predictions_file, data)
    event_summary = _verify_events(events_file, data)
    return {
        "status": "passed",
        "mode": "read_only_completed_run_verification",
        "config": provenance_module.portable_path(config_file, root=PROJECT_ROOT),
        "experiment": expected_experiment,
        "seed": seed,
        "positive_class": POSITIVE_CLASS,
        "training_config_sha256": expected_training,
        "provenance_sha256": expected_provenance,
        "preflight": provenance_module.portable_path(preflight_file, root=PROJECT_ROOT),
        "run_directory": provenance_module.portable_path(
            metrics_file.parent, root=PROJECT_ROOT
        ),
        "predictions": prediction_summary,
        "events": event_summary,
    }


def _verify_aggregate_csv(path: Path, expected: Mapping[str, Any]) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"Aggregate CSV has unexpected columns: {reader.fieldnames}")
        rows = list(reader)
    expected_rows = []
    for metric, values in expected["aggregate"].items():
        expected_rows.append(
            {
                "metric": metric,
                **{name: str(values[name]) for name in CSV_FIELDS[1:]},
            }
        )
    if rows != expected_rows:
        raise ValueError("Aggregate CSV does not match the current metric inputs")
    return len(rows)


def verify_aggregate(
    *,
    metrics_paths: Sequence[str | Path],
    aggregate_json_path: str | Path,
    aggregate_csv_path: str | Path,
    preflight_path: str | Path,
) -> dict[str, Any]:
    """Recompute an aggregate in memory and compare it without writing outputs."""

    if not metrics_paths:
        raise ValueError("At least one metrics file is required")
    _, preflight = _preflight_identity(preflight_path)
    expected_provenance = _sha256_identity(
        preflight["runtime_provenance"].get("provenance_sha256"),
        "preflight provenance_sha256",
    )
    metrics_files = [_require_regular_file(path, "aggregate input metrics") for path in metrics_paths]
    expected = aggregate_module.aggregate_result_files(metrics_files)
    if expected.get("provenance_sha256") != expected_provenance:
        raise ValueError("Aggregate inputs do not match the current strict preflight provenance")
    aggregate_json_file, actual = _read_json_object(
        aggregate_json_path, "aggregate JSON"
    )
    aggregate_csv_file = _require_regular_file(aggregate_csv_path, "aggregate CSV")
    if actual != expected:
        raise ValueError("Aggregate JSON does not match a read-only recomputation")
    csv_rows = _verify_aggregate_csv(aggregate_csv_file, expected)
    return {
        "status": "passed",
        "mode": "read_only_aggregate_verification",
        "experiment": expected["experiment"],
        "seeds": expected["seeds"],
        "provenance_sha256": expected_provenance,
        "metrics_files": len(metrics_files),
        "aggregate_metrics": csv_rows,
        "aggregate_json": provenance_module.portable_path(
            aggregate_json_file, root=PROJECT_ROOT
        ),
        "aggregate_csv": provenance_module.portable_path(
            aggregate_csv_file, root=PROJECT_ROOT
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Verify one completed seed run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--seed", required=True, type=int)
    run_parser.add_argument("--metrics", required=True)
    run_parser.add_argument("--predictions", required=True)
    run_parser.add_argument("--events", required=True)
    run_parser.add_argument("--preflight", required=True)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Verify an existing aggregate against current metric inputs"
    )
    aggregate_parser.add_argument("metrics", nargs="+")
    aggregate_parser.add_argument("--aggregate-json", required=True)
    aggregate_parser.add_argument("--aggregate-csv", required=True)
    aggregate_parser.add_argument("--preflight", required=True)
    args = parser.parse_args()

    if args.command == "run":
        payload = verify_completed_run(
            config_path=args.config,
            seed=args.seed,
            metrics_path=args.metrics,
            predictions_path=args.predictions,
            events_path=args.events,
            preflight_path=args.preflight,
        )
    else:
        payload = verify_aggregate(
            metrics_paths=args.metrics,
            aggregate_json_path=args.aggregate_json,
            aggregate_csv_path=args.aggregate_csv,
            preflight_path=args.preflight,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
