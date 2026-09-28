from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

DATASET_NAME = "bci_truth_deception_v1"
SESSION_ID = "session_001"
SPLIT_ORDER = ("train", "val", "test")
MAX_INPUT_BYTES = 64 * 1024 * 1024
MODEL_MODALITIES = (
    "eeg_time",
    "eeg_spectral",
    "physiology",
    "video",
    "audio",
    "text",
)


def _regular_file(path: Path) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    path = path.resolve()
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"Input exceeds the {MAX_INPUT_BYTES}-byte audit limit: {path}")
    return path


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _strict_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _read_json(path: Path) -> Any:
    path = _regular_file(path)
    return _strict_json(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = _regular_file(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = _strict_json(line)
            if not isinstance(record, dict):
                raise ValueError(f"Manifest line {line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_label_rows(path: Path) -> list[tuple[int, float, float]]:
    path = _regular_file(path)
    rows: list[tuple[int, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Start_Time", "End_Time", "Label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Primary label CSV is missing columns {sorted(required)}")
        for line_number, record in enumerate(reader, 2):
            try:
                label_value = float(record["Label"])
                label = int(label_value)
                start = float(record["Start_Time"])
                end = float(record["End_Time"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric metadata at CSV line {line_number}") from error
            if label_value != label or label not in {0, 1}:
                raise ValueError(f"Invalid binary label at CSV line {line_number}")
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError(f"Invalid timestamp at CSV line {line_number}")
            if rows and start < rows[-1][1]:
                raise ValueError(f"Non-monotonic start time at CSV line {line_number}")
            rows.append((label, start, end))
    if not rows:
        raise ValueError("Primary label CSV is empty")
    return rows


def _parse_feature_value(value: str) -> float:
    text = (value or "").strip()
    if not text or text.lower() in {"nan", "na", "none", "null"}:
        return float("nan")
    bracketed = re.fullmatch(r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]", text)
    if bracketed:
        text = bracketed.group(1)
    parsed = float(text)
    return parsed if math.isfinite(parsed) else float("nan")


def _read_model_feature_matrix(
    path: Path,
    model_mapping: Mapping[str, Sequence[str]],
    expected_labels: Sequence[int],
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    path = _regular_file(path)
    feature_names: list[str] = []
    modality_indices: dict[str, list[int]] = {}
    for modality in MODEL_MODALITIES:
        names = [str(name) for name in model_mapping[modality]]
        start = len(feature_names)
        feature_names.extend(names)
        modality_indices[modality] = list(range(start, start + len(names)))
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Model feature mapping contains duplicate column names")

    rows: list[list[float]] = []
    observed_labels: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Feature table has no header")
        missing = sorted(set(feature_names) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Feature table is missing {len(missing)} model columns")
        for line_number, record in enumerate(reader, 2):
            try:
                label_value = float(record["Label"])
                label = int(label_value)
                values = [_parse_feature_value(record[name]) for name in feature_names]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid model feature at CSV line {line_number}") from error
            if label_value != label:
                raise ValueError(f"Non-integral label at feature CSV line {line_number}")
            observed_labels.append(label)
            rows.append(values)
    if observed_labels != list(expected_labels):
        raise ValueError("Feature-table labels do not match the primary label table")
    return np.asarray(rows, dtype=np.float64), feature_names, modality_indices


def _fit_train_preprocessing(
    matrix: np.ndarray, train_start: int, train_end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = matrix[train_start:train_end]
    median = np.nanmedian(train, axis=0)
    if np.isnan(median).any():
        raise ValueError("At least one model feature is entirely missing in the train split")
    filled = np.where(np.isnan(matrix), median[None, :], matrix)
    mean = filled[train_start:train_end].mean(axis=0)
    scale = filled[train_start:train_end].std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (filled - mean[None, :]) / scale[None, :]
    return standardized, median, scale


def _point_biserial(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    class_zero = labels == 0
    class_one = labels == 1
    if not class_zero.any() or not class_one.any():
        return np.zeros(matrix.shape[1], dtype=np.float64)
    mean_zero = matrix[class_zero].mean(axis=0)
    mean_one = matrix[class_one].mean(axis=0)
    scale = matrix.std(axis=0)
    factor = math.sqrt(float(class_zero.sum() * class_one.sum())) / len(labels)
    correlation = np.divide(
        (mean_one - mean_zero) * factor,
        scale,
        out=np.zeros_like(scale),
        where=scale > 1e-12,
    )
    return np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    class_zero = labels == 0
    class_one = labels == 1
    if not class_zero.any() or not class_one.any():
        return float("nan")
    specificity = float((predictions[class_zero] == 0).mean())
    sensitivity = float((predictions[class_one] == 1).mean())
    return 0.5 * (specificity + sensitivity)


def _fit_single_feature_threshold(
    values: np.ndarray, labels: np.ndarray
) -> tuple[float, str, float]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order]
    total_zero = int((labels == 0).sum())
    total_one = int((labels == 1).sum())
    if total_zero == 0 or total_one == 0:
        return 0.0, "high_is_one", float("nan")
    cumulative_one = np.concatenate(([0], np.cumsum(sorted_labels == 1)))
    cumulative_zero = np.concatenate(([0], np.cumsum(sorted_labels == 0)))
    valid_cuts = np.ones(len(values) + 1, dtype=bool)
    if len(values) > 1:
        valid_cuts[1:-1] = sorted_values[:-1] < sorted_values[1:]

    true_negative_high = cumulative_zero
    true_positive_high = total_one - cumulative_one
    balanced_high = 0.5 * (true_negative_high / total_zero + true_positive_high / total_one)
    true_positive_low = cumulative_one
    true_negative_low = total_zero - cumulative_zero
    balanced_low = 0.5 * (true_negative_low / total_zero + true_positive_low / total_one)
    balanced_high[~valid_cuts] = -1.0
    balanced_low[~valid_cuts] = -1.0

    high_index = int(np.argmax(balanced_high))
    low_index = int(np.argmax(balanced_low))
    if balanced_high[high_index] >= balanced_low[low_index]:
        cut = high_index
        direction = "high_is_one"
        score = float(balanced_high[high_index])
    else:
        cut = low_index
        direction = "low_is_one"
        score = float(balanced_low[low_index])
    if cut == 0:
        threshold = float(np.nextafter(sorted_values[0], -np.inf))
    elif cut == len(values):
        threshold = float(np.nextafter(sorted_values[-1], np.inf))
    else:
        threshold = float((sorted_values[cut - 1] + sorted_values[cut]) / 2.0)
    return threshold, direction, score


def _threshold_predictions(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "high_is_one":
        return (values > threshold).astype(np.int64)
    return (values <= threshold).astype(np.int64)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[cursor]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _lag_one_column_correlations(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < 3:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    first = matrix[:-1]
    second = matrix[1:]
    first_centered = first - first.mean(axis=0)
    second_centered = second - second.mean(axis=0)
    denominator = np.sqrt(
        np.square(first_centered).sum(axis=0) * np.square(second_centered).sum(axis=0)
    )
    correlations = np.divide(
        (first_centered * second_centered).sum(axis=0),
        denominator,
        out=np.zeros(matrix.shape[1], dtype=np.float64),
        where=denominator > 1e-12,
    )
    return np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)


def _cross_split_similarity(
    reference: np.ndarray,
    query: np.ndarray,
    reference_labels: np.ndarray | None = None,
    query_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    reference_rows: dict[bytes, list[int]] = {}
    for index, row in enumerate(reference.astype(np.float32)):
        key = np.ascontiguousarray(row).tobytes()
        if reference_labels is not None:
            reference_rows.setdefault(key, []).append(int(reference_labels[index]))
        else:
            reference_rows.setdefault(key, [])
    exact_keys = [np.ascontiguousarray(row).tobytes() for row in query.astype(np.float32)]
    exact_matches = sum(key in reference_rows for key in exact_keys)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    normalized_reference = np.divide(
        reference,
        reference_norm,
        out=np.zeros_like(reference),
        where=reference_norm > 1e-12,
    )
    normalized_query = np.divide(
        query,
        query_norm,
        out=np.zeros_like(query),
        where=query_norm > 1e-12,
    )
    maximum_cosine = np.max(normalized_query @ normalized_reference.T, axis=1)
    exact_label_accuracy = None
    exact_lookup_balanced_accuracy = None
    exact_lookup_accuracy = None
    if reference_labels is not None and query_labels is not None and exact_matches:
        correct = 0
        reference_counts = Counter(int(value) for value in reference_labels.tolist())
        fallback = min(reference_counts, key=lambda label: (-reference_counts[label], label))
        lookup_predictions = np.full(len(query_labels), fallback, dtype=np.int64)
        for index, key in enumerate(exact_keys):
            if key not in reference_rows:
                continue
            counts = Counter(reference_rows[key])
            predicted = min(counts, key=lambda label: (-counts[label], label))
            lookup_predictions[index] = predicted
            correct += int(predicted == int(query_labels[index]))
        exact_label_accuracy = correct / exact_matches
        exact_lookup_accuracy = float((lookup_predictions == query_labels).mean())
        exact_lookup_balanced_accuracy = _balanced_accuracy(query_labels, lookup_predictions)
    return {
        "query_steps": len(query),
        "exact_row_matches": int(exact_matches),
        "exact_row_match_rate": float(exact_matches / len(query)),
        "exact_match_train_majority_label_accuracy": exact_label_accuracy,
        "exact_lookup_accuracy_with_train_majority_fallback": exact_lookup_accuracy,
        "exact_lookup_balanced_accuracy_with_train_majority_fallback": (
            exact_lookup_balanced_accuracy
        ),
        "nearest_train_cosine": {
            "median": float(np.median(maximum_cosine)),
            "q95": float(np.quantile(maximum_cosine, 0.95)),
            "max": float(maximum_cosine.max()),
        },
        "near_match_rate_cosine_ge_0_999": float((maximum_cosine >= 0.999).mean()),
        "near_match_rate_cosine_ge_0_9999": float((maximum_cosine >= 0.9999).mean()),
    }


def _modality_leakage_screen(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    modality_indices: Mapping[str, Sequence[int]],
    labels: Sequence[int],
    split_ranges: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int64)
    train_start, train_end = (int(value) for value in split_ranges["train"])
    val_start, val_end = (int(value) for value in split_ranges["val"])
    test_start, test_end = (int(value) for value in split_ranges["test"])
    standardized, _, _ = _fit_train_preprocessing(matrix, train_start, train_end)
    train_labels = label_array[train_start:train_end]
    val_labels = label_array[val_start:val_end]
    test_labels = label_array[test_start:test_end]

    output: dict[str, Any] = {}
    for modality in MODEL_MODALITIES:
        indices = np.asarray(modality_indices[modality], dtype=np.int64)
        train = standardized[train_start:train_end][:, indices]
        val = standardized[val_start:val_end][:, indices]
        test = standardized[test_start:test_end][:, indices]
        train_correlations = _point_biserial(train, train_labels)
        test_correlations = _point_biserial(test, test_labels)
        top_train_corr = int(np.argmax(np.abs(train_correlations)))
        top_test_corr = int(np.argmax(np.abs(test_correlations)))

        best_feature = 0
        best_threshold = 0.0
        best_direction = "high_is_one"
        best_train_balanced = -1.0
        for local_index in range(train.shape[1]):
            threshold, direction, balanced = _fit_single_feature_threshold(
                train[:, local_index], train_labels
            )
            if balanced > best_train_balanced:
                best_feature = local_index
                best_threshold = threshold
                best_direction = direction
                best_train_balanced = balanced
        oriented_train = (
            train[:, best_feature] if best_direction == "high_is_one" else -train[:, best_feature]
        )
        oriented_val = (
            val[:, best_feature] if best_direction == "high_is_one" else -val[:, best_feature]
        )
        oriented_test = (
            test[:, best_feature] if best_direction == "high_is_one" else -test[:, best_feature]
        )
        lag = _lag_one_column_correlations(test)
        output[modality] = {
            "feature_count": len(indices),
            "max_abs_point_biserial_train": {
                "feature": feature_names[int(indices[top_train_corr])],
                "value": float(abs(train_correlations[top_train_corr])),
            },
            "max_abs_point_biserial_test_posthoc": {
                "feature": feature_names[int(indices[top_test_corr])],
                "value": float(abs(test_correlations[top_test_corr])),
            },
            "best_train_selected_single_feature_threshold": {
                "feature": feature_names[int(indices[best_feature])],
                "direction": best_direction,
                "threshold_standardized": best_threshold,
                "train_balanced_accuracy": best_train_balanced,
                "validation_balanced_accuracy": _balanced_accuracy(
                    val_labels,
                    _threshold_predictions(val[:, best_feature], best_threshold, best_direction),
                ),
                "test_balanced_accuracy": _balanced_accuracy(
                    test_labels,
                    _threshold_predictions(test[:, best_feature], best_threshold, best_direction),
                ),
                "train_auroc": _roc_auc(train_labels, oriented_train),
                "validation_auroc": _roc_auc(val_labels, oriented_val),
                "test_auroc": _roc_auc(test_labels, oriented_test),
            },
            "test_lag1_feature_correlation": {
                "median": float(np.median(lag)),
                "q90": float(np.quantile(lag, 0.90)),
                "max": float(lag.max()),
            },
            "cross_split_similarity": {
                "train_to_validation": _cross_split_similarity(
                    train, val, train_labels, val_labels
                ),
                "train_to_test": _cross_split_similarity(train, test, train_labels, test_labels),
            },
        }
    output["combined_model_input"] = {
        "feature_count": standardized.shape[1],
        "cross_split_similarity": {
            "train_to_validation": _cross_split_similarity(
                standardized[train_start:train_end],
                standardized[val_start:val_end],
                train_labels,
                val_labels,
            ),
            "train_to_test": _cross_split_similarity(
                standardized[train_start:train_end],
                standardized[test_start:test_end],
                train_labels,
                test_labels,
            ),
        },
    }
    return output


def _label_temporal_statistics(
    labels: Sequence[int], split_ranges: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int64)
    output: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        start, end = (int(value) for value in split_ranges[split])
        values = label_array[start:end]
        first = values[:-1].astype(np.float64)
        second = values[1:].astype(np.float64)
        first -= first.mean()
        second -= second.mean()
        denominator = math.sqrt(float(np.square(first).sum() * np.square(second).sum()))
        lag_correlation = (
            float((first * second).sum() / denominator) if denominator > 1e-12 else 0.0
        )
        previous_predictions = values[:-1]
        output[split] = {
            "lag1_label_correlation": lag_correlation,
            "adjacent_transition_rate": float((values[1:] != values[:-1]).mean()),
            "previous_label_balanced_accuracy": _balanced_accuracy(
                values[1:], previous_predictions
            ),
        }
    return output


def _feature_provenance(raw_root: Path) -> dict[str, Any]:
    code_suffixes = {".py", ".ipynb", ".r", ".m", ".jl"}
    extractor_files = [
        path
        for path in raw_root.iterdir()
        if path.is_file() and path.suffix.lower() in code_suffixes
    ]
    return {
        "raw_video_present": (raw_root / f"{SESSION_ID}_video.mp4").is_file(),
        "raw_audio_present": (raw_root / f"{SESSION_ID}_audio.wav").is_file(),
        "word_timestamps_present": (
            raw_root / f"{SESSION_ID}_word_timestamps_refined.csv"
        ).is_file(),
        "precomputed_feature_table_present": (
            raw_root / f"{SESSION_ID}_multimodal_features.csv"
        ).is_file(),
        "extractor_code_files_in_staged_bundle": len(extractor_files),
        "text": {
            "source_trace": "word timestamps and a 768-dimensional table are present",
            "derivation_traceable": False,
            "missing_evidence": "embedding model/version, token context, causal context policy, and fit scope",
        },
        "video": {
            "source_trace": "raw video and landmark/action-unit columns are present",
            "derivation_traceable": False,
            "missing_evidence": "extractor code/version, windowing, calibration, and fit scope",
        },
        "audio": {
            "source_trace": "raw audio and acoustic columns are present",
            "derivation_traceable": False,
            "missing_evidence": "extractor code/version, windowing, and fit scope",
        },
    }


def _event_runs(labels: Sequence[int], event_label: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label == event_label and start is None:
            start = index
        elif label != event_label and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(labels) - 1))
    return runs


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "q25": _quantile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "q75": _quantile(values, 0.75),
        "max": max(values),
    }


def _event_safe_boundary(labels: Sequence[int], index: int, event_label: int) -> bool:
    return index in {0, len(labels)} or not (
        labels[index - 1] == event_label and labels[index] == event_label
    )


def _window_statistics(
    records: Sequence[Mapping[str, Any]], split_start: int, split_end: int
) -> dict[str, Any]:
    coverage: Counter[int] = Counter()
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Every manifest record needs metadata")
        start = int(metadata["source_row_start"])
        end = int(metadata["source_row_end_exclusive"])
        if start < split_start or end > split_end or start >= end:
            raise ValueError(
                f"Window [{start}, {end}) lies outside split [{split_start}, {split_end})"
            )
        coverage.update(range(start, end))
    expected = set(range(split_start, split_end))
    observed = set(coverage)
    if observed != expected:
        missing = len(expected - observed)
        outside = len(observed - expected)
        raise ValueError(f"Window coverage mismatch: {missing} missing, {outside} outside")
    total_observations = sum(coverage.values())
    unique_steps = len(coverage)
    duplicate_observations = total_observations - unique_steps
    repeated_unique_steps = sum(count > 1 for count in coverage.values())
    return {
        "window_count": len(records),
        "window_observations": total_observations,
        "unique_steps": unique_steps,
        "duplicate_observations": duplicate_observations,
        "duplicate_observation_rate": duplicate_observations / total_observations,
        "repeated_unique_steps": repeated_unique_steps,
        "repeated_unique_step_rate": repeated_unique_steps / unique_steps,
        "mean_coverage_multiplicity": total_observations / unique_steps,
        "max_coverage_multiplicity": max(coverage.values()),
        "coverage_multiplicity_histogram": {
            str(key): value for key, value in sorted(Counter(coverage.values()).items())
        },
    }


def build_data_audit(
    data_root: str | Path, results_root: str | Path | None = None
) -> dict[str, Any]:
    """Build a read-only aggregate audit without loading pickle-backed tensor files."""
    root = Path(data_root).resolve()
    processed = root / "processed" / DATASET_NAME
    raw = root / "raw" / DATASET_NAME / SESSION_ID
    summary = _read_json(processed / "dataset_summary.json")
    schema = _read_json(processed / "feature_schema.json")
    label_path = raw / f"{SESSION_ID}_labels_primary.csv"
    label_hash = _sha256(label_path)
    rows = _read_label_rows(label_path)
    labels = [row[0] for row in rows]
    event_label = int(schema["event_positive_label"])

    if int(summary["num_steps"]) != len(rows):
        raise ValueError("dataset_summary num_steps does not match the primary label table")
    computed_label_counts = {str(key): value for key, value in sorted(Counter(labels).items())}
    if summary.get("label_counts") != computed_label_counts:
        raise ValueError("dataset_summary label_counts does not match the primary label table")

    split_ranges = summary.get("actual_split_ranges", summary.get("split_ranges"))
    if not isinstance(split_ranges, dict):
        raise ValueError("dataset_summary is missing split ranges")
    previous_end = 0
    split_audits: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        start, end = (int(value) for value in split_ranges[split])
        if start != previous_end or end <= start or end > len(rows):
            raise ValueError(f"Invalid or non-contiguous {split} range [{start}, {end})")
        previous_end = end
        split_rows = rows[start:end]
        split_labels = labels[start:end]
        local_runs = _event_runs(split_labels, event_label)
        run_lengths = [run_end - run_start + 1 for run_start, run_end in local_runs]
        run_durations = [
            split_rows[run_end][2] - split_rows[run_start][1] for run_start, run_end in local_runs
        ]
        manifest_records = _read_jsonl(processed / "manifests" / f"{split}.jsonl")
        window_stats = _window_statistics(manifest_records, start, end)
        expected_windows = int(summary["windows_per_split"][split])
        if window_stats["window_count"] != expected_windows:
            raise ValueError(f"Manifest count for {split} does not match dataset_summary")
        split_audits[split] = {
            "range": [start, end],
            "unique_steps": end - start,
            "class_counts": {
                str(key): value for key, value in sorted(Counter(split_labels).items())
            },
            "event_runs": len(local_runs),
            "event_length_steps": _distribution(run_lengths),
            "event_duration_seconds": _distribution(run_durations),
            "boundary_counts_from_unchanged_labels": {
                "start": len(local_runs),
                "end": len(local_runs),
            },
            "time": {
                "start_seconds": split_rows[0][1],
                "end_seconds": split_rows[-1][2],
                "span_seconds": split_rows[-1][2] - split_rows[0][1],
            },
            "event_safe_start": _event_safe_boundary(labels, start, event_label),
            "event_safe_end": _event_safe_boundary(labels, end, event_label),
            "windows": window_stats,
        }
    if previous_end != len(rows):
        raise ValueError("Split ranges do not cover every primary-label row")

    full_runs = _event_runs(labels, event_label)
    full_run_lengths = [end - start + 1 for start, end in full_runs]
    full_run_durations = [rows[end][2] - rows[start][1] for start, end in full_runs]
    all_cuts_safe = all(
        split_audits[split]["event_safe_start"] and split_audits[split]["event_safe_end"]
        for split in SPLIT_ORDER
    )
    feature_matrix, feature_names, modality_indices = _read_model_feature_matrix(
        raw / f"{SESSION_ID}_multimodal_features.csv",
        schema["model_mapping"],
        labels,
    )
    leakage_screen = _modality_leakage_screen(
        feature_matrix,
        feature_names,
        modality_indices,
        labels,
        split_ranges,
    )
    temporal_statistics = _label_temporal_statistics(labels, split_ranges)

    observed_result = None
    if results_root is not None:
        metrics_path = Path(results_root) / "eptnet_default" / "seed_42" / "test_metrics.json"
        if metrics_path.is_file():
            metrics = _read_json(metrics_path)
            observed_result = {
                "experiment": "eptnet_default",
                "seed": 42,
                "test_unique_steps": metrics.get("data", {}).get("num_unique_steps"),
                "test_event_runs": metrics.get("data", {}).get("num_target_events"),
                "test_auroc": metrics.get("frame", {}).get("auroc"),
                "test_macro_f1": metrics.get("frame", {}).get("macro_f1"),
                "checkpoint_epoch": metrics.get("checkpoint", {}).get("epoch"),
            }

    strongest_single = max(
        (
            (
                modality,
                values["best_train_selected_single_feature_threshold"]["test_auroc"],
                values["best_train_selected_single_feature_threshold"]["test_balanced_accuracy"],
            )
            for modality, values in ((name, leakage_screen[name]) for name in MODEL_MODALITIES)
        ),
        key=lambda item: item[1],
    )
    result_interpretation = {
        "can_auroc_0_943_be_inflated": True,
        "assessment": (
            "The score is a within-session, one-seed pilot and may be inflated by "
            "participant/session leakage, temporal dependence, high-dimensional feature "
            "selection, and unverified upstream feature extraction. The audit does not prove "
            "that a label column was copied into the inputs."
        ),
        "strongest_train_selected_single_feature_on_test": {
            "modality": strongest_single[0],
            "auroc": strongest_single[1],
            "balanced_accuracy": strongest_single[2],
        },
        "direct_leakage_caveat": (
            "Low single-feature separability or zero exact row matches cannot exclude "
            "multivariate leakage or identity/session memorization. Post-hoc test maxima are "
            "diagnostics only."
        ),
    }

    return {
        "audit_version": 1,
        "audit_date": date.today().isoformat(),
        "dataset": summary["dataset"],
        "session_count": 1,
        "scope": {
            "complete_primary_label_rows_scanned": len(rows),
            "all_manifest_records_scanned": True,
            "tensor_files_loaded": False,
            "boundary_count_basis": "deterministic runs recomputed from unchanged labels",
        },
        "label_integrity": {
            "labels_modified": False,
            "source_label_sha256": label_hash,
            "statement": "This audit is read-only and does not smooth, merge, relabel, or overwrite labels.",
        },
        "feature_dimensions": summary.get("feature_dimensions", {}),
        "nominal_split_ranges": summary.get("nominal_split_ranges"),
        "actual_split_ranges": {
            split: [int(value) for value in split_ranges[split]] for split in SPLIT_ORDER
        },
        "all_split_boundaries_event_safe": all_cuts_safe,
        "overall": {
            "unique_steps": len(rows),
            "class_counts": computed_label_counts,
            "event_runs": len(full_runs),
            "event_length_steps": _distribution(full_run_lengths),
            "event_duration_seconds": _distribution(full_run_durations),
            "boundary_counts_from_unchanged_labels": {
                "start": len(full_runs),
                "end": len(full_runs),
            },
            "time": {
                "start_seconds": rows[0][1],
                "end_seconds": rows[-1][2],
                "span_seconds": rows[-1][2] - rows[0][1],
            },
        },
        "splits": split_audits,
        "temporal_dependence": temporal_statistics,
        "feature_leakage_screen": leakage_screen,
        "feature_provenance": _feature_provenance(raw),
        "observed_seed42_result": observed_result,
        "seed42_result_interpretation": result_interpretation,
        "limitations": summary.get("limitations", []),
    }


def _format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Data Audit",
        "",
        f"- Audit date: {report['audit_date']}",
        f"- Dataset: `{report['dataset']}`",
        "- Label handling: **read-only; no smoothing, merging, or relabeling**",
        f"- Event-safe split boundaries: **{'yes' if report['all_split_boundaries_event_safe'] else 'no'}**",
        "",
        "## Split statistics",
        "",
        "| Split | Unique steps | Label 0 / 1 | Event runs | Event length steps (min/median/mean/max) | Boundaries start/end | Time span (s) | Windows | Duplicate observation rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLIT_ORDER:
        values = report["splits"][split]
        lengths = values["event_length_steps"]
        boundaries = values["boundary_counts_from_unchanged_labels"]
        windows = values["windows"]
        length_text = "/".join(
            _format_number(lengths[key]) for key in ("min", "median", "mean", "max")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(values["unique_steps"]),
                    f"{values['class_counts'].get('0', 0)} / {values['class_counts'].get('1', 0)}",
                    str(values["event_runs"]),
                    length_text,
                    f"{boundaries['start']} / {boundaries['end']}",
                    _format_number(values["time"]["span_seconds"]),
                    str(windows["window_count"]),
                    f"{100.0 * windows['duplicate_observation_rate']:.2f}%",
                ]
            )
            + " |"
        )

    overall = report["overall"]
    lengths = overall["event_length_steps"]
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- {overall['unique_steps']} unique steps over {overall['time']['span_seconds']:.2f} seconds.",
            f"- Class counts: label 0 = {overall['class_counts'].get('0', 0)}, label 1 = {overall['class_counts'].get('1', 0)}.",
            f"- {overall['event_runs']} event runs; length min/median/mean/max = "
            f"{_format_number(lengths['min'])}/{_format_number(lengths['median'])}/"
            f"{_format_number(lengths['mean'])}/{_format_number(lengths['max'])} steps.",
            f"- Derived boundary counts: {overall['boundary_counts_from_unchanged_labels']['start']} starts and "
            f"{overall['boundary_counts_from_unchanged_labels']['end']} ends.",
            "",
            "## Leakage and temporal-dependence screen",
            "",
            "| Modality | Max |r| train | Max |r| test (post-hoc) | Train-selected 1-feature BA train/test | Train-selected 1-feature test AUROC | Exact test rows in train | Test nearest-train cosine q95 | Test feature lag-1 median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for modality in MODEL_MODALITIES:
        values = report["feature_leakage_screen"][modality]
        threshold = values["best_train_selected_single_feature_threshold"]
        similarity = values["cross_split_similarity"]["train_to_test"]
        lines.append(
            "| "
            + " | ".join(
                [
                    modality,
                    f"{values['max_abs_point_biserial_train']['value']:.3f}",
                    f"{values['max_abs_point_biserial_test_posthoc']['value']:.3f}",
                    f"{threshold['train_balanced_accuracy']:.3f}/{threshold['test_balanced_accuracy']:.3f}",
                    f"{threshold['test_auroc']:.3f}",
                    str(similarity["exact_row_matches"]),
                    f"{similarity['nearest_train_cosine']['q95']:.4f}",
                    f"{values['test_lag1_feature_correlation']['median']:.3f}",
                ]
            )
            + " |"
        )

    observed = report.get("observed_seed42_result")
    interpretation = report["seed42_result_interpretation"]
    combined_similarity = report["feature_leakage_screen"]["combined_model_input"][
        "cross_split_similarity"
    ]["train_to_test"]
    text_similarity = report["feature_leakage_screen"]["text"]["cross_split_similarity"][
        "train_to_test"
    ]
    text_match_accuracy = text_similarity["exact_match_train_majority_label_accuracy"]
    text_match_accuracy_text = (
        f"{100.0 * text_match_accuracy:.1f}%" if text_match_accuracy is not None else "n/a"
    )
    text_lookup_balanced = text_similarity[
        "exact_lookup_balanced_accuracy_with_train_majority_fallback"
    ]
    text_lookup_balanced_text = (
        f"{text_lookup_balanced:.3f}" if text_lookup_balanced is not None else "n/a"
    )
    test_temporal = report["temporal_dependence"]["test"]
    lines.extend(
        [
            "",
            "## Seed-42 result interpretation",
            "",
            (
                f"The observed EPT-Net test AUROC is **{observed['test_auroc']:.3f}** "
                f"on {observed['test_unique_steps']} unique steps from one session."
                if observed is not None
                else "No completed seed-42 metric file was available during this audit."
            ),
            "",
            interpretation["assessment"],
            "",
            f"Across all model inputs jointly, {combined_similarity['exact_row_matches']} test rows "
            "exactly match a training row. For text alone, "
            f"{text_similarity['exact_row_matches']}/{text_similarity['query_steps']} test rows "
            "exactly match a training embedding; the training-majority label agrees on "
            f"{text_match_accuracy_text} of those matches. An exact-text lookup with the "
            f"training-majority fallback reaches test balanced accuracy {text_lookup_balanced_text}.",
            "",
            f"Test-label lag-1 correlation is {test_temporal['lag1_label_correlation']:.3f}; "
            f"a previous-label predictor reaches balanced accuracy "
            f"{test_temporal['previous_label_balanced_accuracy']:.3f}. Thus label persistence is "
            "present but does not by itself account for AUROC 0.943; same-session feature and "
            "lexical reuse remain important confounds.",
            "",
            "The strongest train-selected single feature on the test split came from "
            f"`{interpretation['strongest_train_selected_single_feature_on_test']['modality']}` "
            "with AUROC "
            f"{interpretation['strongest_train_selected_single_feature_on_test']['auroc']:.3f} "
            "and balanced accuracy "
            f"{interpretation['strongest_train_selected_single_feature_on_test']['balanced_accuracy']:.3f}. "
            + interpretation["direct_leakage_caveat"],
            "",
            "## Provenance and definitions",
            "",
            "Raw audio/video and word timestamps are present, but the staged bundle contains no extractor code. "
            "The 768-dimensional text embedding, video landmark/action-unit features, and acoustic features "
            "therefore cannot be reproduced or checked for causal context and fit scope from this bundle alone.",
            "",
            "Boundary counts were deterministically recomputed from contiguous label-0 runs. "
            "Window duplication is reported as `(window observations - unique steps) / window observations`; "
            "it describes storage/sampling overlap, not additional independent data.",
            "",
            "The observational unit remains one session. Split rows and event runs are temporally correlated and "
            "must not be interpreted as independent participants.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, text: str, force: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing report without --force: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary_path, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prepared BCI split and event statistics")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-json", default="results/data_audit.json")
    parser.add_argument("--output-md", default="results/data_audit.md")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    for path in (output_json, output_md):
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing report without --force: {path}")
    report = build_data_audit(args.data_root, results_root=args.results_root)
    _atomic_write(
        output_json,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        force=args.force,
    )
    _atomic_write(output_md, render_markdown(report), force=args.force)
    print(json.dumps({"json": str(output_json), "markdown": str(output_md)}))


if __name__ == "__main__":
    main()
