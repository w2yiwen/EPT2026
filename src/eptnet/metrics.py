"""Compatibility facade; use :mod:`eptnet.evaluation.metrics`."""

from .evaluation import metrics as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
