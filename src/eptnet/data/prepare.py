"""Prepare native-time tensors from synchronized recordings and annotations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from eptnet.models.pretrained import FrozenBehaviorEncoders
from .physiology import prepare_physiology


def event_targets(times, events, target_mask, positive_class=0):
    times = torch.as_tensor(times, dtype=torch.float64)
    valid = torch.as_tensor(target_mask, dtype=torch.bool)
    positive = torch.zeros(len(times), dtype=torch.bool)
    for start, end in events:
        positive |= (times >= float(start)) & (times <= float(end))
    positive &= valid
    labels = torch.full((len(times),), 1 - positive_class, dtype=torch.long)
    labels[positive] = positive_class
    boundaries = torch.zeros(len(times), 2)
    offsets = torch.zeros(len(times), 2)
    indices = torch.nonzero(positive).flatten().tolist()
    runs = []
    for index in indices:
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    for run in runs:
        start, end = run[0], run[-1]
        boundaries[start, 0], boundaries[end, 1] = 1, 1
        positions = torch.arange(start, end + 1)
        offsets[start:end + 1, 0] = positions - start
        offsets[start:end + 1, 1] = end - positions
    return {"labels": labels, "boundaries": boundaries, "offsets": offsets, "positive_mask": positive}


def _segment_mask(times, segments):
    mask = torch.zeros(len(times), dtype=torch.bool)
    for start, end in segments:
        mask |= (times >= float(start)) & (times <= float(end))
    return mask


def prepare_recording(raw, config, encoders=None, device="cpu"):
    """Offsets use session_time = device_time + marker_offset - timeline_origin.

    Facial crops and 16-kHz mono audio enter with their own absolute timestamps.
    Annotation/transcript timestamps share the reference clock and are shifted
    by the same origin; marker offsets correct the independently acquired clocks.
    """
    origin = float(raw.get("timeline_origin", 0.0))
    offsets = raw.get("time_offsets", {})
    def clock(name):
        return torch.as_tensor(raw.get(f"{name}_times", []), dtype=torch.float64) + float(offsets.get(name, 0)) - origin
    eeg_times, ppg_times, frame_times, audio_times = (clock(name) for name in ("eeg", "ppg", "frame", "audio"))
    start = float(raw.get("session_start", origin)) - origin
    end = float(raw["session_end"]) - origin
    # theta_t = t * Delta: retain only decision-grid locations inside this recording.
    first, last = int(np.ceil(start / 0.5)), int(np.floor(end / 0.5))
    times = torch.arange(first, last + 1, dtype=torch.float64) * 0.5
    if not times.numel():
        raise ValueError("The recording contains no 0.5-second decision")
    streams = prepare_physiology(
        np.asarray(raw.get("eeg", np.empty((0, 8)))), eeg_times.numpy(),
        np.asarray(raw.get("ppg", np.empty(0))), ppg_times.numpy(), times.numpy())
    encoders = FrozenBehaviorEncoders(config["encoders"], device) if encoders is None else encoders
    frames = torch.as_tensor(raw.get("frames", torch.empty(0, 3, 224, 224)))
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    else:
        frames = frames.float()
    audio = torch.as_tensor(raw.get("audio", []), dtype=torch.float32)
    if audio.ndim != 1:
        raise ValueError("Supply synchronized 16-kHz mono audio before prefix extraction")
    words = [{**word, "start": float(word["start"]) - origin, "end": float(word["end"]) - origin}
             for word in raw.get("words", [])]
    words.sort(key=lambda word: (word["end"], word["start"]))
    behavior = [encoders.prefix(float(now), frames=frames, frame_times=frame_times,
                               audio=audio, audio_times=audio_times, words=words) for now in times]
    def segments(name):
        return [[float(left) - origin, float(right) - origin] for left, right in raw[name]]
    sequence = (_segment_mask(times, segments("observation_segments"))
                if "observation_segments" in raw else torch.ones(len(times), dtype=torch.bool))
    if "annotated_segments" in raw:
        target = _segment_mask(times, segments("annotated_segments")) & sequence
    else:
        target = sequence.clone() if "events" in raw else torch.zeros_like(sequence)
    events = segments("events") if "events" in raw else []
    payload = {"sample_id": raw.get("sample_id", "session"),
               "metadata": {"subject_id": str(raw["subject_id"]), "timeline_origin": origin,
                            "time_offsets": offsets, "text_protocol": "reference_transcript_prefix",
                            "encoders": config["encoders"], "encoder_weights": encoders.provenance,
                            "decision_seconds": 0.5,
                            "spectral_order": "channel_major: delta,theta,alpha,beta,gamma",
                            "preprocessing": "causal Butterworth; trailing Welch; NeuroKit2 0.2.11 Elgendi"},
               "times": times, "streams": {name: {key: torch.as_tensor(value) for key, value in stream.items()}
                                             for name, stream in streams.items()},
               "sequence_mask": sequence, "target_mask": target,
               **event_targets(times, events, target)}
    for name in ("video", "audio", "text", "modality_mask"):
        payload[name] = torch.stack([step[name].detach().cpu() for step in behavior])
    return payload


def prepare_file(source, destination, config, device="cpu"):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Prepared recording already exists: {destination}")
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
    else:
        raw = torch.load(source, map_location="cpu", weights_only=False)
    payload = prepare_recording(raw, config, device=device)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    payload["metadata"]["raw_sha256"] = digest.hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        torch.save(payload, handle)
    return {"output": str(destination), "decisions": len(payload["times"])}
