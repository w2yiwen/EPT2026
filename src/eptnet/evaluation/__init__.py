"""Evaluation metrics, decoding, aggregation, and artifact generation."""

from .decoding import CausalEventDecoder, decode_events
from .metrics import (
    PROBABILITY_EPSILON,
    boundary_metrics,
    early_detection_recall,
    event_metrics,
    frame_metrics,
    interval_iou,
)

__all__ = [
    "CausalEventDecoder",
    "PROBABILITY_EPSILON",
    "boundary_metrics",
    "decode_events",
    "early_detection_recall",
    "event_metrics",
    "frame_metrics",
    "interval_iou",
]
