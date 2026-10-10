"""Native-rate physiological histories for the read-update-write protocol."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

BRANCHES = ("time", "spec", "hr")


@dataclass
class ObservationBatch:
    """New observations: [N,F], [N,2] support, [N] availability and stable IDs.

    All times are seconds on the session timeline. IDs increase within a stream;
    re-sending an already consumed ID does not create a second memory entry.
    """

    values: Tensor
    support: Tensor
    available: Tensor
    indices: Tensor
    mask: Tensor

    def to(self, device) -> ObservationBatch:
        target = torch.device(device)
        time_dtype = torch.float32 if target.type == "mps" else torch.float64
        return ObservationBatch(self.values.to(target), self.support.to(target, dtype=time_dtype),
            self.available.to(target, dtype=time_dtype), self.indices.to(target), self.mask.to(target))


@dataclass
class MemoryEntries:
    values: Tensor
    support: Tensor
    available: Tensor
    indices: Tensor

    def detach(self) -> MemoryEntries:
        return MemoryEntries(self.values.detach(), self.support, self.available, self.indices)


class RepresentationMemory:
    """HPRM for one recording; caches are runtime state, never model buffers."""

    def __init__(self, horizons: dict[str, float], reference: Tensor) -> None:
        self.horizons = dict(horizons)
        self.entries: dict[str, MemoryEntries] = {}
        self.latest = {name: torch.zeros_like(reference) for name in BRANCHES}
        self.latest_valid = {name: False for name in BRANCHES}
        self.last_index = {name: -1 for name in BRANCHES}

    def pending(self, name: str, observations: ObservationBatch, now: float) -> Tensor:
        ready = observations.available <= now
        # A future observation is deferred, rather than consumed by its ID.
        ready &= observations.support[:, 1] <= now
        return ready & (observations.indices > self.last_index[name])

    def prune(self, now: float) -> None:
        for name, entries in self.entries.items():
            keep = entries.support[:, 1] >= now - self.horizons[name]
            self.entries[name] = MemoryEntries(
                entries.values[keep], entries.support[keep],
                entries.available[keep], entries.indices[keep]
            )

    def history(self, name: str, now: float) -> MemoryEntries | None:
        entries = self.entries.get(name)
        if entries is None:
            return None
        keep = (entries.available <= now) & (entries.support[:, 1] <= now)
        return MemoryEntries(entries.values[keep], entries.support[keep],
                             entries.available[keep], entries.indices[keep])

    def commit(self, name: str, values: Tensor, observations: ObservationBatch,
               pending: Tensor, now: float) -> None:
        if not bool(pending.any()):
            return
        self.last_index[name] = int(observations.indices[pending].max())
        keep = pending & observations.mask.bool()
        keep &= torch.isfinite(observations.values).all(dim=-1)
        if not bool(keep.any()):
            return
        # The latest path can use this batch at the present decision. It becomes
        # retrievable only after this method is called at the end of that decision.
        self.latest[name] = values[keep][-1:]
        self.latest_valid[name] = True
        keep &= observations.support[:, 1] >= now - self.horizons[name]
        if not bool(keep.any()):
            return
        added = MemoryEntries(values[keep], observations.support[keep],
                              observations.available[keep], observations.indices[keep])
        previous = self.entries.get(name)
        if previous is not None:
            added = MemoryEntries(*(torch.cat((old, new), dim=0) for old, new in zip(
                (previous.values, previous.support, previous.available, previous.indices),
                (added.values, added.support, added.available, added.indices), strict=True
            )))
        self.entries[name] = added

    def detach(self) -> None:
        self.entries = {name: entries.detach() for name, entries in self.entries.items()}
        self.latest = {name: value.detach() for name, value in self.latest.items()}
