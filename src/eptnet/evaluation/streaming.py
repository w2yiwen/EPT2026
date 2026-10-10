"""Native-stream inference, evidence traces and the existing event decoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from eptnet.data.streaming import StreamManifestDataset, StreamSession
from eptnet.models import load_model
from eptnet.training.trainer import resolve_device
from .decoding import decode_events
from .evaluator import (evaluate_sequences, calibrate_frame_threshold,
                        _padded_boundary_arrays, _subject_summary)
from .metrics import boundary_metrics, event_metrics, frame_metrics


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {name: _json_value(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


@torch.no_grad()
def predict_session(model, session, trace_handle=None, trace=False):
    model.eval()
    runtime = model.initial_state()
    fields = {name: [] for name in ("class_logits", "boundary_logits", "offsets",
                                    "read_lags", "read_centers", "read_widths", "write_gates", "anchors")}
    chunk = int(model.config["training"]["chunk_steps"])
    positive = int(model.config["evaluation"]["positive_class"])
    for start in range(0, len(session), chunk):
        stop = min(start + chunk, len(session))
        outputs = model(session, runtime, start, stop, trace=trace)
        for name in fields:
            fields[name].append(outputs[name][0].cpu())
        if trace_handle is not None:
            probabilities = torch.softmax(outputs["class_logits"], dim=-1)[0, :, positive]
            boundaries = torch.sigmoid(outputs["boundary_logits"])[0]
            for local, index in enumerate(range(start, stop)):
                record = {"sample_id": session.sample_id, "step": index, "time_seconds": float(session.times[index]),
                    "sequence_valid": bool(session.sequence_mask[index]), "target_valid": bool(session.target_mask[index]),
                    "positive_probability": float(probabilities[local]),
                    "boundary_probabilities": boundaries[local], "offsets": outputs["offsets"][0, local],
                    "anchors": outputs["anchors"][0, local], "read_lags": outputs["read_lags"][0, local],
                    "read_centers": outputs["read_centers"][0, local],
                    "read_widths": outputs["read_widths"][0, local], "write_gates": outputs["write_gates"][0, local]}
                if trace:
                    record["evidence"] = outputs["read_details"][local]
                trace_handle.write(json.dumps(_json_value(record), allow_nan=False) + "\n")
        runtime.detach()
    sequence = {name: torch.cat(values, dim=0) for name, values in fields.items()}
    sequence.update({"sample_id": session.sample_id, "metadata": session.metadata,
        "labels": session.labels, "boundaries": session.boundaries,
        "target_mask": session.target_mask & session.sequence_mask,
        "row_indices": torch.arange(len(session)),
        "timestamps": torch.stack((session.times, session.times), dim=-1)})
    return sequence


def _participant_metrics(sequences, predictions, targets, config):
    evaluation = config["evaluation"]
    groups = {}
    for index, sequence in enumerate(sequences):
        groups.setdefault(str(sequence["metadata"]["subject_id"]), []).append(index)
    rows = []
    for subject, indices in groups.items():
        selected = [sequences[index] for index in indices]
        probability = np.concatenate([torch.softmax(item["class_logits"], dim=-1)[:, 0].numpy() for item in selected])
        labels = np.concatenate([item["labels"].numpy() for item in selected])
        mask = np.concatenate([item["target_mask"].numpy() for item in selected])
        frame = frame_metrics(probability, labels, mask, threshold=evaluation["frame_threshold"], positive_class=0)
        scores, boundaries, valid = _padded_boundary_arrays(selected)
        boundary = boundary_metrics(scores, boundaries, mask=valid,
            threshold=evaluation["boundary_threshold"], tolerance=evaluation["boundary_tolerance_steps"])
        event = event_metrics([predictions[index] for index in indices], [targets[index] for index in indices],
                              evaluation["event_iou_thresholds"])
        rows.append({"subject_id": subject, "frame_average_precision": frame["average_precision"],
                     "frame_macro_f1": frame["macro_f1"], "boundary_macro_f1": boundary["boundary_macro_f1"],
                     "event_f1_iou_0.5": event["event_f1_iou_0.5"]})
    return {"unit": "participant; recordings pooled within participant", "num_subjects": len(rows),
            "subjects": rows, "metrics": {key: _subject_summary([row[key] for row in rows])
                for key in ("frame_average_precision", "frame_macro_f1", "boundary_macro_f1", "event_f1_iou_0.5")}}


def evaluate(config, checkpoint, output, predictions_output=None, events_output=None, trace=False,
             split="test", no_calibration=False):
    device = resolve_device(config["training"]["device"])
    model = load_model(checkpoint, str(device))
    for section in ("model", "encoders"):
        if model.config[section] != config[section]:
            raise ValueError(f"Checkpoint {section} differs from evaluation configuration")
    provenance = model.checkpoint_metadata.get("provenance")
    if provenance is None:
        raise ValueError("The streaming checkpoint must contain its training-participant record")
    training_subjects = set(provenance["datasets"]["train"]["subjects"])
    evaluation = dict(config["evaluation"])
    if evaluation.get("calibrate_frame_threshold", False) and not no_calibration:
        validation = StreamManifestDataset(config["data"]["val_manifest"])
        if validation.subjects & training_subjects:
            raise ValueError("Calibration participants overlap checkpoint training participants")
        val_sequences = [predict_session(model, validation[index]) for index in range(len(validation))]
        grouped = {}
        for sequence in val_sequences:
            grouped.setdefault(str(sequence["metadata"]["subject_id"]), []).append(sequence)
        participant_sequences = [{key: torch.cat([item[key] for item in recordings], dim=0)
                                  for key in ("class_logits", "labels", "target_mask")}
                                 for recordings in grouped.values()]
        calibration = {"enabled": True, "selection_split": "validation",
                       **calibrate_frame_threshold(participant_sequences, evaluation)}
        evaluation["frame_threshold"] = calibration["frame_threshold"]
    else:
        calibration = {"enabled": False, "frame_threshold": evaluation["frame_threshold"]}
    requested = {**config, "evaluation": evaluation}
    paths = [Path(output), Path(predictions_output) if predictions_output else Path(output).with_name("test_predictions.jsonl"),
             Path(events_output) if events_output else Path(output).with_name("test_events.json")]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Evaluation output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    dataset = StreamManifestDataset(config["data"][f"{split}_manifest"])
    train = StreamManifestDataset(config["data"]["train_manifest"])
    if dataset.subjects & (train.subjects | training_subjects):
        raise ValueError("Evaluation participants overlap the training manifest")
    if split == "test" and evaluation.get("calibrate_frame_threshold", False) and not no_calibration:
        if dataset.subjects & validation.subjects:
            raise ValueError("Test participants overlap threshold-calibration participants")
    sequences = []
    with paths[1].open("x", encoding="utf-8") as handle:
        for index in range(len(dataset)):
            sequences.append(predict_session(model, dataset[index], handle, trace))
    metrics, predictions, targets = evaluate_sequences(sequences, requested, evaluation["frame_threshold"])
    metrics["subject_macro"] = _participant_metrics(sequences, predictions, targets, requested)
    metrics["protocol"].update({"evaluation_unit": "0.5-second decisions in independent continuous recordings",
        "input_mode": "native_streams", "decision_seconds": 0.5,
        "latency_definition": "non-negative decoder emit time minus target onset on the 0.5-second grid",
        "target_mask_definition": "annotated, observed decision steps; invalid segments close candidates without resetting state"})
    metrics["checkpoint"] = str(checkpoint)
    metrics["training_provenance"] = provenance
    metrics["calibration"] = calibration
    with paths[0].open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    with paths[2].open("x", encoding="utf-8") as handle:
        json.dump({"sample_ids": [item["sample_id"] for item in sequences], "predictions": predictions,
                   "targets": targets, "decision_seconds": 0.5}, handle, indent=2, allow_nan=False)
    return metrics


def infer(checkpoint, source, output, device="cpu", trace=False):
    model = load_model(checkpoint, str(resolve_device(device)))
    session = StreamSession.from_payload(torch.load(source, map_location="cpu", weights_only=False))
    path = Path(output)
    event_path = path.with_name(path.stem + "_events.json")
    if path.exists() or event_path.exists():
        raise FileExistsError("Inference outputs already exist; select a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        sequence = predict_session(model, session, handle, trace)
    settings = model.config["evaluation"]
    event_outputs = {name: sequence[name][None] for name in ("class_logits", "boundary_logits", "offsets")}
    event_outputs["sequence_mask"] = torch.ones(1, len(session), dtype=torch.bool)
    # Inference needs observation validity, never ground-truth annotation availability.
    event_outputs["target_mask"] = session.sequence_mask[None]
    events = decode_events(event_outputs, positive_class=0, frame_threshold=settings["frame_threshold"],
                           boundary_threshold=settings["boundary_threshold"])[0]
    with event_path.open("x", encoding="utf-8") as handle:
        json.dump({"sample_id": session.sample_id, "events": events, "decision_seconds": 0.5}, handle, indent=2)
    return {"predictions": str(path), "events": str(event_path), "decisions": len(session)}
