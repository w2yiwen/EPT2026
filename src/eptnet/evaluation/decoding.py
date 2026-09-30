from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


class CausalEventDecoder:
    """Incremental threshold decoder that never revisits a closed event.

    ``step`` consumes exactly one newly available model output. It returns a
    completed event when the positive-class score falls below threshold. An
    event's ``emit_step`` is the first observed threshold crossing, while its
    final interval is computed only from outputs observed by its close step.
    """

    def __init__(self, frame_threshold: float = 0.5, boundary_threshold: float = 0.5) -> None:
        self.frame_threshold = float(frame_threshold)
        self.boundary_threshold = float(boundary_threshold)
        if (
            not math.isfinite(self.frame_threshold)
            or not math.isfinite(self.boundary_threshold)
            or not 0.0 <= self.frame_threshold <= 1.0
            or not 0.0 <= self.boundary_threshold <= 1.0
        ):
            raise ValueError("decoder thresholds must be finite probabilities within [0, 1]")
        self._active: dict[str, float] | None = None
        self._last_step = -1

    @property
    def is_active(self) -> bool:
        return self._active is not None

    @property
    def emit_step(self) -> int | None:
        if self._active is None:
            return None
        return int(self._active["emit_step"])

    def _open(
        self,
        step_index: int,
        segment_start: int,
        class_probability: float,
        boundary_probabilities: Sequence[float],
        offsets: Sequence[float],
    ) -> None:
        self._active = {
            "segment_start": float(segment_start),
            "emit_step": float(step_index),
            "last_active_step": float(step_index),
            "score": float(class_probability),
            "peak_step": float(step_index),
            "peak_left_offset": max(float(offsets[0]), 0.0),
            "peak_right_offset": max(float(offsets[1]), 0.0),
            "best_start_probability": float(boundary_probabilities[0]),
            "best_start_step": float(step_index),
            "best_end_probability": float(boundary_probabilities[1]),
            "best_end_step": float(step_index),
        }

    def _observe_active(
        self,
        step_index: int,
        class_probability: float,
        boundary_probabilities: Sequence[float],
        offsets: Sequence[float],
    ) -> None:
        assert self._active is not None
        active = self._active
        active["last_active_step"] = float(step_index)
        if class_probability > active["score"]:
            active["score"] = float(class_probability)
            active["peak_step"] = float(step_index)
            active["peak_left_offset"] = max(float(offsets[0]), 0.0)
            active["peak_right_offset"] = max(float(offsets[1]), 0.0)
        if boundary_probabilities[0] > active["best_start_probability"]:
            active["best_start_probability"] = float(boundary_probabilities[0])
            active["best_start_step"] = float(step_index)
        if boundary_probabilities[1] > active["best_end_probability"]:
            active["best_end_probability"] = float(boundary_probabilities[1])
            active["best_end_step"] = float(step_index)

    def _close(self, close_step: int) -> dict[str, float]:
        assert self._active is not None
        active = self._active
        segment_start = int(active["segment_start"])
        last_active = int(active["last_active_step"])
        peak = int(active["peak_step"])
        offset_start = peak - int(round(active["peak_left_offset"]))
        offset_end = peak + int(round(active["peak_right_offset"]))
        offset_start = max(segment_start, min(offset_start, last_active))
        offset_end = max(segment_start, min(offset_end, last_active))

        if active["best_start_probability"] >= self.boundary_threshold:
            start = int(round((active["best_start_step"] + offset_start) / 2.0))
        else:
            start = offset_start
        if active["best_end_probability"] >= self.boundary_threshold:
            end = int(round((active["best_end_step"] + offset_end) / 2.0))
        else:
            end = offset_end
        start = max(segment_start, min(start, last_active))
        end = max(start, min(end, last_active))

        event = {
            "start": float(start),
            "end": float(end),
            "score": float(active["score"]),
            "emit_step": float(active["emit_step"]),
            "peak_step": float(active["peak_step"]),
            "close_step": float(close_step),
        }
        self._active = None
        return event

    def step(
        self,
        class_probability: float,
        boundary_probabilities: Sequence[float],
        offsets: Sequence[float],
        step_index: int | None = None,
        segment_start: int | None = None,
    ) -> list[dict[str, float]]:
        """Consume one time step and return any event completed at this step."""
        if step_index is None:
            step_index = self._last_step + 1
        step_index = int(step_index)
        segment_start = 0 if segment_start is None else int(segment_start)
        if step_index <= self._last_step:
            raise ValueError("step_index must increase monotonically")
        if segment_start < 0 or segment_start > step_index:
            raise ValueError("segment_start must be within [0, step_index]")
        if len(boundary_probabilities) != 2 or len(offsets) != 2:
            raise ValueError("boundary_probabilities and offsets must each contain two values")
        values = [
            float(class_probability),
            *map(float, boundary_probabilities),
            *map(float, offsets),
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("decoder inputs must be finite")
        self._last_step = step_index

        if class_probability >= self.frame_threshold:
            if self._active is None:
                self._open(
                    step_index,
                    segment_start,
                    float(class_probability),
                    boundary_probabilities,
                    offsets,
                )
            else:
                self._observe_active(
                    step_index,
                    float(class_probability),
                    boundary_probabilities,
                    offsets,
                )
            return []
        if self._active is None:
            return []
        return [self._close(close_step=step_index)]

    def finalize(self) -> list[dict[str, float]]:
        """Close an event that remains active at end of stream."""
        if self._active is None:
            return []
        return [self._close(close_step=self._last_step)]


@torch.no_grad()
def decode_events(
    outputs: dict[str, Tensor],
    positive_class: int = 0,
    frame_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
) -> list[list[dict[str, float]]]:
    """Decode model outputs by replaying each valid sequence one step at a time."""
    class_prob = torch.softmax(outputs["class_logits"], dim=-1)[..., positive_class]
    boundary_prob = torch.sigmoid(outputs["boundary_logits"])
    offsets = outputs["offsets"]
    mask = outputs["sequence_mask"].bool()
    decision_mask = outputs.get("target_mask", mask).bool() & mask
    decoded: list[list[dict[str, float]]] = []

    for batch_index in range(class_prob.shape[0]):
        valid_length = int(mask[batch_index].sum().item())
        state_machine = CausalEventDecoder(frame_threshold, boundary_threshold)
        events: list[dict[str, float]] = []
        segment_start = 0
        for step_index in range(valid_length):
            if not bool(decision_mask[batch_index, step_index].item()):
                # Invalid target-speaker intervals are chronological separators.
                # A zero score closes an active event and can never open one;
                # using the original wall-clock step preserves event timing.
                events.extend(
                    state_machine.step(
                        0.0,
                        (0.0, 0.0),
                        (0.0, 0.0),
                        step_index=step_index,
                        segment_start=segment_start,
                    )
                )
                segment_start = step_index + 1
                continue
            events.extend(
                state_machine.step(
                    float(class_prob[batch_index, step_index].item()),
                    boundary_prob[batch_index, step_index].tolist(),
                    offsets[batch_index, step_index].tolist(),
                    step_index=step_index,
                    segment_start=segment_start,
                )
            )
        events.extend(state_machine.finalize())
        decoded.append(events)
    return decoded
