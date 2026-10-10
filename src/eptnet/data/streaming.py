"""Recording tensors on the 0.5-second grid with independent native-rate streams."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from eptnet.models.memory import BRANCHES, ObservationBatch


@dataclass
class StreamSession:
    sample_id: str
    metadata: dict[str, Any]
    times: Tensor
    streams: dict[str, ObservationBatch]
    video: Tensor
    audio: Tensor
    text: Tensor
    modality_mask: Tensor
    sequence_mask: Tensor
    target_mask: Tensor
    labels: Tensor
    boundaries: Tensor
    offsets: Tensor
    positive_mask: Tensor
    physiology_mask: Tensor

    def __len__(self) -> int:
        return self.times.numel()

    @classmethod
    def from_payload(cls, payload: dict, sample_id: str = "session",
                     metadata: dict | None = None) -> StreamSession:
        times = torch.as_tensor(payload["times"], dtype=torch.float64)
        n = times.numel()
        if (times.ndim != 1 or n == 0 or not bool(torch.isfinite(times).all())
                or n > 1 and not bool(torch.allclose(times[1:] - times[:-1], times.new_full((n - 1,), 0.5)))):
            raise ValueError("times must be a nonempty chronological 0.5-second decision grid")
        streams = {}
        for name, width in (("time", 8), ("spec", 40), ("hr", 1)):
            stream = payload.get("streams", {}).get(name, {})
            values = torch.as_tensor(stream.get("values", torch.empty(0, width)), dtype=torch.float32)
            size = values.shape[0]
            support = torch.as_tensor(stream.get("support", torch.empty(0, 2)), dtype=torch.float64)
            available = torch.as_tensor(stream.get("available", torch.empty(0)), dtype=torch.float64)
            if values.shape != (size, width) or support.shape != (size, 2) or available.shape != (size,):
                raise ValueError(f"{name} must contain [N,{width}] values, [N,2] support and [N] availability")
            if size and (bool((available[1:] < available[:-1]).any())
                         or bool((support[1:, 1] < support[:-1, 1]).any())
                         or bool((support[:, 0] > support[:, 1]).any())
                         or not bool(torch.isfinite(support).all() & torch.isfinite(available).all())
                         or bool((support[:, 1] > available).any())):
                raise ValueError(f"{name} must be availability-ordered and available after its support ends")
            streams[name] = ObservationBatch(values, support, available,
                torch.arange(size), torch.as_tensor(stream.get("mask", torch.ones(size)), dtype=torch.bool))
        sequence = torch.as_tensor(payload.get("sequence_mask", torch.ones(n)), dtype=torch.bool)
        default_target = sequence if "labels" in payload else torch.zeros_like(sequence)
        target = torch.as_tensor(payload.get("target_mask", default_target), dtype=torch.bool)
        labels = torch.as_tensor(payload.get("labels", torch.ones(n)), dtype=torch.long)
        return cls(payload.get("sample_id", sample_id), {**(metadata or {}), **payload.get("metadata", {})},
            times, streams, *(torch.as_tensor(payload.get(name, torch.zeros(n, width)), dtype=torch.float32)
                             for name, width in (("video", 384), ("audio", 768), ("text", 768))),
            torch.as_tensor(payload.get("modality_mask", torch.zeros(n, 3)), dtype=torch.bool),
            sequence, target, labels,
            torch.as_tensor(payload.get("boundaries", torch.zeros(n, 2)), dtype=torch.float32),
            torch.as_tensor(payload.get("offsets", torch.zeros(n, 2)), dtype=torch.float32),
            torch.as_tensor(payload.get("positive_mask", labels.eq(0)), dtype=torch.bool),
            torch.as_tensor(payload.get("physiology_mask", torch.ones(n, 3)), dtype=torch.bool))

    def decision(self, index: int, device="cpu") -> dict[str, Any]:
        now = float(self.times[index])
        previous = float(self.times[index - 1]) if index else -float("inf")
        new = {}
        for name in BRANCHES:
            stream = self.streams[name]
            left = int(torch.searchsorted(stream.available, previous, right=True))
            right = int(torch.searchsorted(stream.available, now, right=True))
            new[name] = ObservationBatch(*(value[left:right].to(device) for value in (
                stream.values, stream.support, stream.available, stream.indices, stream.mask)))
        return {"time": now, "streams": new,
                **{name: getattr(self, name)[index].to(device) for name in ("video", "audio", "text")},
                "modality_mask": self.modality_mask[index].to(device),
                "physiology_mask": self.physiology_mask[index],
                "sequence_mask": bool(self.sequence_mask[index])}

    def targets(self, start: int, stop: int, device="cpu") -> dict[str, Tensor]:
        return {name: getattr(self, name)[start:stop].to(device)[None] for name in (
            "sequence_mask", "target_mask", "labels", "boundaries", "offsets", "positive_mask")}


class StreamManifestDataset:
    def __init__(self, manifest: str | Path) -> None:
        self.path = Path(manifest)
        with self.path.open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]

    @property
    def subjects(self) -> set[str]:
        return {str(record["subject_id"]) for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def recording_hashes(self) -> list[str]:
        hashes = []
        for record in self.records:
            path = Path(record["path"])
            path = path if path.is_absolute() else self.path.parent / path
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            hashes.append(digest.hexdigest())
        return hashes

    def __getitem__(self, index: int) -> StreamSession:
        record = self.records[index]
        path = Path(record["path"])
        path = path if path.is_absolute() else self.path.parent / path
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload_subject = payload.get("metadata", {}).get("subject_id")
        if payload_subject is not None and str(payload_subject) != str(record["subject_id"]):
            raise ValueError(f"Participant identity differs between manifest and recording: {path}")
        return StreamSession.from_payload(payload, record.get("sample_id", path.parent.name),
            {"subject_id": str(record["subject_id"]), "source": str(path), **record.get("metadata", {})})


def normalization_statistics(dataset: StreamManifestDataset) -> dict[str, Any]:
    """Fit population statistics once, using only the training manifest's rows."""
    moments = {name: [0, torch.zeros(width, dtype=torch.float64),
                      torch.zeros(width, dtype=torch.float64)]
               for name, width in (("time", 8), ("spec", 40), ("hr", 1))}
    for index in range(len(dataset)):
        for name, stream in dataset[index].streams.items():
            valid = stream.mask & torch.isfinite(stream.values).all(dim=-1)
            values = stream.values[valid].double()
            count, mean, m2 = moments[name]
            if not values.shape[0]:
                continue
            added = values.shape[0]
            new_mean = values.mean(dim=0)
            delta = new_mean - mean
            total = count + added
            new_m2 = (values - new_mean).square().sum(dim=0)
            moments[name] = [total, mean + delta * added / total,
                             m2 + new_m2 + delta.square() * count * added / total]
    result = {"subjects": sorted(dataset.subjects), "manifest": str(dataset.path),
              "manifest_sha256": hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
              "recording_sha256": dataset.recording_hashes()}
    for name, (count, mean, m2) in moments.items():
        scale = (m2 / max(count, 1)).clamp_min(0).sqrt().clamp_min(1e-6)
        if not count:
            scale = torch.ones_like(scale)
        result[name] = {"count": count, "mean": mean.tolist(), "scale": scale.tolist()}
    return result
